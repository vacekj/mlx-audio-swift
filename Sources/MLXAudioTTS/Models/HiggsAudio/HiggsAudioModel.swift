//
//  HiggsAudioModel.swift
//  MLXAudio
//
//  Higgs Audio v3 TTS (bosonai/higgs-tts-3-4b) for MLX. A Qwen3-4B decoder
//  with a fused multi-codebook (8 x 1026) audio head, a delay pattern, and a
//  bundled DAC-style acoustic decoder. Generation builds a prompt of
//  text/voice tokens plus delayed audio codes, runs the backbone on the
//  embedded prompt, then autoregressively samples delayed 8-codebook rows
//  which are reverse-delayed and vocoded to a 24 kHz waveform.
//
//  Ported from mlx_audio/tts/models/higgs_audio_v3 (Python). Zero-shot
//  synthesis is fully supported; the reference-audio encode path (voice
//  cloning) is not yet ported.
//

import Foundation
@preconcurrency import MLX
import MLXNN
import HuggingFace
import Tokenizers
@preconcurrency import MLXLMCommon
import MLXAudioCore

// MARK: - Backbone (Qwen3, accepts continuous input embeddings)

final class HiggsAttention: Module {
    let args: HiggsAudioTextConfig
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(_ args: HiggsAudioTextConfig) {
        self.args = args
        let dim = args.hiddenSize
        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)
        self._wq.wrappedValue = Linear(dim, args.attentionHeads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, args.kvHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, args.kvHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(args.attentionHeads * headDim, dim, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        self.rope = RoPE(dimensions: headDim, traditional: false, base: args.ropeTheta, scale: 1)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return wo(output)
    }
}

final class HiggsMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dim: Int, hidden: Int) {
        self._gate.wrappedValue = Linear(dim, hidden, bias: false)
        self._down.wrappedValue = Linear(hidden, dim, bias: false)
        self._up.wrappedValue = Linear(dim, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class HiggsTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: HiggsAttention
    @ModuleInfo(key: "mlp") var mlp: HiggsMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: HiggsAudioTextConfig) {
        self._attention.wrappedValue = HiggsAttention(args)
        self._mlp.wrappedValue = HiggsMLP(dim: args.hiddenSize, hidden: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}

/// Qwen3 backbone that runs on precomputed embeddings (text tokens are
/// embedded separately via ``embedTokenIds`` and concatenated with audio-code
/// embeddings before the forward pass).
final class HiggsBackbone: Module {
    let hiddenSize: Int
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [HiggsTransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ args: HiggsAudioTextConfig) {
        self.hiddenSize = args.hiddenSize
        self._embedTokens.wrappedValue = Embedding(embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        self.layers = (0..<args.hiddenLayers).map { _ in HiggsTransformerBlock(args) }
        // The Higgs checkpoint omits the final norm; it is loaded non-strictly
        // (weight stays at ones) to match the Python reference exactly.
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    /// Forward on continuous embeddings ``[B, L, hidden]``.
    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let h = embeddings
        let mask = createAttentionMask(h: h, cache: cache?.first)
        var out = h
        for (i, layer) in layers.enumerated() {
            out = layer(out, mask: mask, cache: cache?[i])
        }
        return norm(out)
    }

    /// Embed a flat list of text token ids -> ``[L, hidden]``.
    func embedTokenIds(_ ids: [Int]) -> MLXArray {
        if ids.isEmpty {
            return MLXArray.zeros([0, hiddenSize])
        }
        let idx = MLXArray(ids) // [L], int32
        return embedTokens(idx) // [L, hidden]
    }
}

// MARK: - Model

public final class HiggsAudioModel: Module, SpeechGenerationModel, @unchecked Sendable {
    /// RNG seed for Higgs generation. Set per request so the zero-shot voice
    /// stays consistent across chunks of one reading, while letting the app
    /// pick/randomize/save a specific voice timbre (see ``generateSamples``).
    public var generationSeed: UInt64 = 42

    let configuration: HiggsAudioConfig

    @ModuleInfo(key: "backbone") var backbone: HiggsBackbone
    @ModuleInfo(key: "multimodal_embedding") var multimodalEmbedding: Embedding

    var codec: HiggsAudioCodec
    var tokenizer: Tokenizers.Tokenizer?

    // Special token ids resolved from the tokenizer's added vocab.
    private(set) var ttsTokenId: Int = 0
    private(set) var refAudioTokenId: Int = 0
    private(set) var textTokenId: Int = 0
    private(set) var audioTokenId: Int = 0
    private(set) var refTextTokenId: Int? = nil

    public init(_ config: HiggsAudioConfig) {
        self.configuration = config
        self._backbone.wrappedValue = HiggsBackbone(config.text)
        self._multimodalEmbedding.wrappedValue = Embedding(
            embeddingCount: config.numCodebooks * config.codebookSize,
            dimensions: config.text.hiddenSize
        )
        self.codec = HiggsAudioCodec(config)
    }

    // MARK: SpeechGenerationModel

    public var sampleRate: Int { configuration.sampleRate }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(maxTokens: 2048, temperature: 0.8, topP: 1.0, topK: 50)
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        let samples = try await generateSamples(
            text: text,
            temperature: generationParameters.temperature,
            topK: generationParameters.topK > 0 ? generationParameters.topK : nil,
            maxTokens: generationParameters.maxTokens ?? 2048,
            refAudio: refAudio,
            refText: refText
        )
        return MLXArray(samples)
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let temp = generationParameters.temperature
        let topK = generationParameters.topK > 0 ? generationParameters.topK : nil
        let maxTokens = generationParameters.maxTokens ?? 2048
        let task = Task { @Sendable [weak self, continuation] in
            guard let self else { return }
            do {
                let samples = try await self.generateSamples(
                    text: text, temperature: temp, topK: topK, maxTokens: maxTokens,
                    refAudio: refAudio, refText: refText
                )
                continuation.yield(.audio(MLXArray(samples)))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }

    // MARK: - Embedding helpers

    /// Sum the fused multi-codebook embeddings for delayed codes ``[..., N]``
    /// -> ``[..., hidden]``.
    private func embedAudioCodes(_ codes: MLXArray) -> MLXArray {
        let was1D = (codes.ndim == 1)
        var c = codes
        if was1D { c = c.expandedDimensions(axis: 0) } // [1, N]
        let n = configuration.numCodebooks
        let offsets = MLXArray(Array(0..<n)) * configuration.codebookSize // [N] int32
        let fused = c + offsets // [..., N]
        let emb = multimodalEmbedding(fused) // [..., N, hidden]
        return MLX.sum(emb, axes: [-2], keepDims: false) // [..., hidden]
    }

    /// Project hidden states to per-codebook logits ``[..., hidden]`` ->
    /// ``[..., N, V]``.
    private func audioLogits(_ hidden: MLXArray) -> MLXArray {
        let flat = multimodalEmbedding.asLinear(hidden) // [..., N*V]
        let v = configuration.codebookSize
        let leading = Array(hidden.shape.dropLast())
        return flat.reshaped(leading + [configuration.numCodebooks, v])
    }

    /// A reference-audio turn: delayed codes ``[L, N]`` plus an optional
    /// transcript, spliced into the prompt ahead of the main ``<|text|>`` turn.
    private struct HiggsReference {
        let delayedCodes: MLXArray
        let text: String?
    }

    /// Encoded reference codes memoized by the source MLXArray identity, so the
    /// same reference clip is encoded once and reused across all chunks of a
    /// reading (mirrors the Qwen3-TTS caching strategy).
    private var referenceCodesCache: [ObjectIdentifier: MLXArray] = [:]

    /// Encode a reference waveform into delay-patterned Higgs codes
    /// ``[T + N - 1, N]``. ``refAudio`` is a 1-D waveform at 24 kHz.
    private func encodeReference(_ refAudio: MLXArray) -> MLXArray {
        if let cached = referenceCodesCache[ObjectIdentifier(refAudio)] {
            return cached
        }
        var wav = refAudio.asArray(Float.self)
        if wav.count < configuration.sampleRate {
            wav += [Float](repeating: 0, count: configuration.sampleRate - wav.count)
        }
        let waveform = MLXArray(wav).reshaped([1, -1, 1]).asType(.float32) // [1, T, 1]
        let codes = codec.encode(waveform)[0].asType(.int32) // [T', 8]
        eval(codes)
        let delayed = higgsApplyDelayPattern(
            codes,
            bocId: configuration.bocTokenId,
            eocId: configuration.eocTokenId
        ) // [T' + N - 1, 8]
        eval(delayed)
        referenceCodesCache[ObjectIdentifier(refAudio)] = delayed
        return delayed
    }

    /// Build the prompt embedding sequence ``[1, L, hidden]`` for ``text``,
    /// optionally preceded by one or more reference turns. Layout mirrors the
    /// Python ``HiggsAudioV3PromptBuilder.build_prompt`` + ``_build_prompt_embeddings``:
    /// ``<|tts|>`` then, per reference, ``[<|ref_text|> refText]? <|ref_audio|>``
    /// followed by the delayed reference codes, then ``<|text|>`` + text + ``<|audio|>``.
    private func buildPromptEmbeddings(text: String, references: [HiggsReference]) -> MLXArray {
        let placeholder = configuration.audioTokenId // -100
        var ids: [Int] = [ttsTokenId]
        var segments: [(start: Int, codes: MLXArray)] = []

        for ref in references {
            if let rid = refTextTokenId, let refText = ref.text, !refText.isEmpty {
                ids.append(rid)
                ids.append(contentsOf: tokenizer!.encode(text: refText, addSpecialTokens: false))
            }
            ids.append(refAudioTokenId)
            let start = ids.count
            let numRows = ref.delayedCodes.dim(0)
            ids.append(contentsOf: Array(repeating: placeholder, count: numRows))
            segments.append((start, ref.delayedCodes))
        }

        ids.append(textTokenId)
        ids.append(contentsOf: tokenizer!.encode(text: text, addSpecialTokens: false))
        ids.append(audioTokenId)

        // Walk the id list, emitting text-token embeddings between audio segments
        // and fused multi-codebook embeddings for the reference code spans. The
        // -100 placeholders are never embedded: they only mark the code span.
        var pieces: [MLXArray] = []
        var cursor = 0
        for (start, codes) in segments {
            if start > cursor {
                pieces.append(backbone.embedTokenIds(Array(ids[cursor..<start])))
            }
            pieces.append(embedAudioCodes(codes))
            cursor = start + codes.dim(0)
        }
        if cursor < ids.count {
            pieces.append(backbone.embedTokenIds(Array(ids[cursor..<ids.count])))
        }

        let nonEmpty = pieces.filter { $0.dim(0) > 0 }
        let full = MLX.concatenated(nonEmpty, axis: 0) // [L, hidden]
        return full.expandedDimensions(axis: 0) // [1, L, hidden]
    }

    /// Codec round-trip for validation: encode a reference waveform to codes,
    /// then decode it back to a waveform. Used to verify the encode path is
    /// faithful (decode is already exercised by zero-shot synthesis).
    public func reconstructReference(_ refAudio: MLXArray) -> MLXArray {
        let delayed = encodeReference(refAudio) // [L, N]
        let raw = higgsReverseDelayPattern(delayed) // [T, N]
        return codec.decode(raw) // [T*960]
    }

    // MARK: - Generation core

    private func generateSamples(text: String, temperature: Float, topK: Int?, maxTokens: Int, refAudio: MLXArray?, refText: String?) async throws -> [Float] {
        guard tokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }

        // Fix the RNG seed per chunk so every chunk of a generation draws from
        // the same random state — otherwise zero-shot Higgs drifts to a
        // different voice/timbre between chunks. Same text+seed is reproducible.
        MLXRandom.seed(generationSeed)

        var references: [HiggsReference] = []
        if let refAudio {
            references.append(HiggsReference(delayedCodes: encodeReference(refAudio), text: refText))
        }
        let promptEmbeds = buildPromptEmbeddings(text: text, references: references)
        eval(promptEmbeds)

        let cache: [KVCache] = (0..<configuration.text.hiddenLayers).map { _ in KVCacheSimple() }

        // Prefill on the embedded prompt; take the last position's hidden state.
        var hidden = backbone(promptEmbeds, cache: cache) // [1, L, hidden]
        var lastHidden = hidden[0..., -1, 0...] // [1, hidden]
        eval(lastHidden)

        var state = HiggsSamplerState(numCodebooks: configuration.numCodebooks)
        var delayedRows: [MLXArray] = []
        let limit = maxTokens

        for _ in 0..<limit {
            try Task.checkCancellation()
            let logits = audioLogits(lastHidden)[0] // [N, V]
            let codes = higgsSamplerStep(
                logits: logits,
                state: &state,
                temperature: temperature,
                topK: topK,
                bocId: configuration.bocTokenId,
                eocId: configuration.eocTokenId
            ) // [N]
            delayedRows.append(codes)
            if state.generationDone { break }

            var nextEmbed = embedAudioCodes(codes) // [1, hidden]
            nextEmbed = nextEmbed.expandedDimensions(axis: 0) // [1, 1, hidden]
            hidden = backbone(nextEmbed, cache: cache)
            lastHidden = hidden[0..., -1, 0...]
            eval(lastHidden)
        }

        Memory.clearCache()

        guard delayedRows.count >= configuration.numCodebooks else {
            throw AudioGenerationError.generationFailed("Higgs produced too few frames (\(delayedRows.count))")
        }

        let delayed = MLX.stacked(delayedRows, axis: 0).asType(.int32) // [L, N]
        eval(delayed)
        let rawCodes = higgsReverseDelayPattern(delayed) // [L-N+1, N]
        var audio = codec.decode(rawCodes) // [T*960]
        audio = applyFades(audio)
        eval(audio)
        Memory.clearCache()
        return audio.asArray(Float.self)
    }

    private func applyFades(_ audio: MLXArray, fadeInMs: Float = 30, fadeOutMs: Float = 15) -> MLXArray {
        var samples = audio.asArray(Float.self)
        let sr = configuration.sampleRate
        let nIn = Int(fadeInMs * Float(sr) / 1000.0)
        let nOut = Int(fadeOutMs * Float(sr) / 1000.0)
        if nIn > 0, samples.count > nIn {
            for i in 0..<nIn { samples[i] *= Float(i) / Float(nIn) }
        }
        if nOut > 0, samples.count > nOut {
            let start = samples.count - nOut
            for i in 0..<nOut { samples[start + i] *= 1.0 - Float(i) / Float(nOut) }
        }
        return MLXArray(samples)
    }

    // MARK: - Weight loading

    /// Map the Higgs checkpoint layout onto this module's parameters.
    func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("tied.embedding.text_embedding.") {
                let rest = String(key.dropFirst("tied.embedding.text_embedding.".count))
                out["backbone.embed_tokens." + rest] = value
            } else if key.hasPrefix("body.layers.") {
                out["backbone.layers." + String(key.dropFirst("body.layers.".count))] = value
            } else if key.hasPrefix("body.norm.") {
                out["backbone.norm." + String(key.dropFirst("body.norm.".count))] = value
            } else if key.hasPrefix("tied.embedding.modality_embeddings.0.embedding.") {
                out["multimodal_embedding." + String(key.dropFirst("tied.embedding.modality_embeddings.0.embedding.".count))] = value
            } else if key.hasPrefix("tied.embedding.modality_embeddings.0.model.") {
                continue // codec tensors — loaded separately
            } else if key.hasPrefix("tied.head.") {
                continue
            } else {
                out[key] = value
            }
        }
        return out
    }

    /// Resolve the special Higgs token ids from the tokenizer's added vocab.
    private func resolveSpecialTokens() throws {
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }
        func id(_ token: String) throws -> Int {
            guard let v = tokenizer.convertTokenToId(token) else {
                throw AudioGenerationError.modelNotInitialized("Tokenizer missing Higgs token: \(token)")
            }
            return v
        }
        ttsTokenId = try id("<|tts|>")
        refAudioTokenId = try id("<|ref_audio|>")
        textTokenId = try id("<|text|>")
        audioTokenId = try id("<|audio|>")
        refTextTokenId = tokenizer.convertTokenToId("<|ref_text|>")
    }

    // MARK: - Loading

    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws -> HiggsAudioModel {
        let hfToken = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(modelRepo)")
        }
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID, requiredExtension: ".safetensors", hfToken: hfToken, cache: cache
        )
        return try await fromModelDirectory(modelDir, cache: cache)
    }

    public static func fromModelDirectory(_ modelDir: URL, cache: HubCache = .default) async throws -> HiggsAudioModel {
        // Config
        let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        let configJSON = try JSONSerialization.jsonObject(with: configData) as? [String: Any] ?? [:]
        let config = HiggsAudioConfig(configJSON: configJSON)

        let model = HiggsAudioModel(config)

        // Weights (single combined safetensors shard for this checkpoint).
        let allWeights = try loadHiggsWeights(from: modelDir)

        // TTS model weights (backbone + multimodal embedding + final norm).
        var sanitized = model.sanitize(allWeights)

        // Codec weights (decode path), extracted from the same shard. The codec
        // is a child module of the model, so its keys are loaded together with
        // the rest under the ``codec.`` prefix.
        let codecPrefix = "tied.embedding.modality_embeddings.0.model."
        var codecRaw: [String: MLXArray] = [:]
        for (k, v) in allWeights where k.hasPrefix(codecPrefix) {
            codecRaw[String(k.dropFirst(codecPrefix.count))] = v
        }
        for (k, v) in HiggsAudioCodec.sanitize(codecRaw) {
            sanitized["codec." + k] = v
        }

        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        eval(model)

        // Tokenizer + special tokens.
        model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        try model.resolveSpecialTokens()

        return model
    }
}

private func loadHiggsWeights(from directory: URL) throws -> [String: MLXArray] {
    let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
    var weights: [String: MLXArray] = [:]
    for file in files where file.pathExtension == "safetensors" {
        let w = try MLX.loadArrays(url: file)
        weights.merge(w) { _, new in new }
    }
    return weights
}
