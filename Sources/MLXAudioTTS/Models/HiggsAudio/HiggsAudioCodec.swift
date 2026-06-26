//
//  HiggsAudioCodec.swift
//  MLXAudio
//
//  Decode-only port of the Higgs Audio tokenizer bundled inside
//  bosonai/higgs-tts-3-4b (stored under
//  ``tied.embedding.modality_embeddings.0.model.*``). The decode path turns
//  8-codebook discrete tokens back into a 24 kHz waveform:
//
//      codes [B,T,8] -> RVQ decode -> [B,T,1024] -> fc2 -> [B,T,256]
//                    -> acoustic decoder (DAC/BigVGAN-style) -> [B,T*960,1]
//
//  The encode path (HuBERT + acoustic encoder, needed only for voice cloning)
//  is intentionally omitted. Zero-shot synthesis does not require it.
//

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Snake activation

/// Snake activation for 1D signals. ``alpha`` has shape ``[1, 1, channels]``.
final class HiggsSnake1d: Module {
    @ModuleInfo(key: "alpha") var alpha: MLXArray

    init(channels: Int) {
        self._alpha.wrappedValue = MLXArray.ones([1, 1, channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Compute in float32 (matches the Python reference): alpha near zero in
        // lower precision makes 1/(alpha+eps) overflow.
        let x32 = x.asType(.float32)
        let a32 = alpha.asType(.float32)
        let recip = 1.0 / (a32 + 1e-9)
        let s = MLX.sin(a32 * x32)
        let y = x32 + recip * (s * s)
        return y.asType(x.dtype)
    }
}

// MARK: - Weight-normalization-free 1D convolution

/// Plain (non-weight-normed) 1D convolution matching the Higgs codec's
/// ``WNConv1d(..., norm="none")`` usage. Stores a single ``.weight`` tensor of
/// shape ``[out, kernel, in]`` (MLX channels-last convention) and applies
/// symmetric padding ``(kernel - stride) * dilation / 2``.
final class HiggsConv1d: Module {
    let kernelSize: Int
    let dilation: Int
    let stride: Int
    let padding: Int

    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, dilation: Int = 1, bias: Bool = true, paddingOverride: Int? = nil) {
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.stride = stride
        self.padding = paddingOverride ?? (kernelSize - stride) * dilation / 2

        let scale = (1.0 / Float(inChannels * kernelSize)).squareRoot()
        self._weight.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [outChannels, kernelSize, inChannels])
        self._bias.wrappedValue = bias ? MLXArray.zeros([outChannels]) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = MLX.conv1d(x, weight, stride: stride, padding: padding, dilation: dilation)
        if let bias { y = y + bias }
        return y
    }
}

// MARK: - Residual unit

/// Dilated residual unit: snake -> conv(k=7, dil) -> snake -> conv(k=1) + skip.
/// Keys: ``snake1``, ``conv1``, ``snake2``, ``conv2``.
final class HiggsResidualUnit: Module {
    @ModuleInfo(key: "snake1") var snake1: HiggsSnake1d
    @ModuleInfo(key: "conv1") var conv1: HiggsConv1d
    @ModuleInfo(key: "snake2") var snake2: HiggsSnake1d
    @ModuleInfo(key: "conv2") var conv2: HiggsConv1d

    init(dim: Int, dilation: Int) {
        self._snake1.wrappedValue = HiggsSnake1d(channels: dim)
        self._conv1.wrappedValue = HiggsConv1d(inChannels: dim, outChannels: dim, kernelSize: 7, dilation: dilation)
        self._snake2.wrappedValue = HiggsSnake1d(channels: dim)
        self._conv2.wrappedValue = HiggsConv1d(inChannels: dim, outChannels: dim, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = snake1(x)
        y = conv1(y)
        y = snake2(y)
        y = conv2(y)
        // Trim skip to match y if lengths differ (no-op for the kernel/dilation
        // combos used here, but keeps the reference contract).
        let pad = (x.dim(1) - y.dim(1)) / 2
        var skip = x
        if pad > 0 {
            skip = x[0..., pad..<(x.dim(1) - pad), 0...]
        }
        return skip + y
    }
}

// MARK: - Decoder block

/// Acoustic decoder block: snake -> transposed-conv upsample -> 3 residual units.
/// Keys: ``snake1``, ``conv_t1``, ``res_unit1..3``.
final class HiggsAcousticDecoderBlock: Module {
    let stride: Int
    @ModuleInfo(key: "snake1") var snake1: HiggsSnake1d
    @ModuleInfo(key: "conv_t1") var convT1: HiggsConvTranspose1d
    @ModuleInfo(key: "res_unit1") var resUnit1: HiggsResidualUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: HiggsResidualUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: HiggsResidualUnit

    init(inDim: Int, outDim: Int, stride: Int) {
        self.stride = stride
        self._snake1.wrappedValue = HiggsSnake1d(channels: inDim)
        self._convT1.wrappedValue = HiggsConvTranspose1d(
            inChannels: inDim, outChannels: outDim,
            kernelSize: 2 * stride, stride: stride, padding: stride / 2
        )
        self._resUnit1.wrappedValue = HiggsResidualUnit(dim: outDim, dilation: 1)
        self._resUnit2.wrappedValue = HiggsResidualUnit(dim: outDim, dilation: 3)
        self._resUnit3.wrappedValue = HiggsResidualUnit(dim: outDim, dilation: 9)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let tIn = x.dim(1)
        var h = snake1(x)
        h = convT1(h)
        // Odd strides produce one extra sample at the tail; trim to tIn*stride.
        let expected = tIn * stride
        if h.dim(1) > expected {
            h = h[0..., 0..<expected, 0...]
        }
        h = resUnit1(h)
        h = resUnit2(h)
        h = resUnit3(h)
        return h
    }
}

/// Transposed 1D convolution matching Higgs' ``nn.ConvTranspose1d(in, out,
/// kernel=2*stride, stride, padding=stride/2)``. Stores ``.weight`` as
/// ``[out, kernel, in]``.
final class HiggsConvTranspose1d: Module {
    let stride: Int
    let padding: Int
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, padding: Int) {
        self.stride = stride
        self.padding = padding
        let scale = (1.0 / Float(inChannels * kernelSize)).squareRoot()
        self._weight.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [outChannels, kernelSize, inChannels])
        self._bias.wrappedValue = MLXArray.zeros([outChannels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = MLX.convTransposed1d(x, weight, stride: stride, padding: padding, dilation: 1)
        if let bias { y = y + bias }
        return y
    }
}

// MARK: - Acoustic decoder

/// DAC-style acoustic decoder: latent ``[B, T, 256]`` -> waveform ``[B, T*960, 1]``.
/// Strides ``[8, 5, 4, 2, 3]`` (product 960), channels 256 -> 1024 -> 512 -> 256
/// -> 128 -> 64 -> 32 -> 1.
final class HiggsAcousticDecoder: Module {
    @ModuleInfo(key: "conv1") var conv1: HiggsConv1d
    @ModuleInfo(key: "block") var blocks: [HiggsAcousticDecoderBlock]
    @ModuleInfo(key: "snake1") var snake1: HiggsSnake1d
    @ModuleInfo(key: "conv2") var conv2: HiggsConv1d

    private static let strides = [8, 5, 4, 2, 3]
    private static let inChannels = [1024, 512, 256, 128, 64]
    private static let outChannels = [512, 256, 128, 64, 32]

    override init() {
        self._conv1.wrappedValue = HiggsConv1d(inChannels: 256, outChannels: 1024, kernelSize: 7)
        var blocks: [HiggsAcousticDecoderBlock] = []
        for i in 0..<Self.strides.count {
            blocks.append(HiggsAcousticDecoderBlock(
                inDim: Self.inChannels[i], outDim: Self.outChannels[i], stride: Self.strides[i]
            ))
        }
        self._blocks.wrappedValue = blocks
        self._snake1.wrappedValue = HiggsSnake1d(channels: 32)
        self._conv2.wrappedValue = HiggsConv1d(inChannels: 32, outChannels: 1, kernelSize: 7)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        for b in blocks { h = b(h) }
        h = snake1(h)
        h = conv2(h)
        return h
    }
}

// MARK: - Acoustic encoder

/// Acoustic encoder block: 3 residual units + snake + strided downsampling conv.
/// Order (mirror of the decoder block): ``res_unit1..3`` -> ``snake1`` ->
/// ``conv1``. Residual units run at ``inDim``; ``conv1`` downsamples
/// ``inDim`` -> ``outDim`` with ``padding = ceil(stride / 2)``.
/// Keys: ``block.N.res_unit1/2/3``, ``block.N.snake1``, ``block.N.conv1``.
final class HiggsAcousticEncoderBlock: Module {
    let stride: Int
    @ModuleInfo(key: "res_unit1") var resUnit1: HiggsResidualUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: HiggsResidualUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: HiggsResidualUnit
    @ModuleInfo(key: "snake1") var snake1: HiggsSnake1d
    @ModuleInfo(key: "conv1") var conv1: HiggsConv1d

    init(inDim: Int, outDim: Int, stride: Int) {
        self.stride = stride
        self._resUnit1.wrappedValue = HiggsResidualUnit(dim: inDim, dilation: 1)
        self._resUnit2.wrappedValue = HiggsResidualUnit(dim: inDim, dilation: 3)
        self._resUnit3.wrappedValue = HiggsResidualUnit(dim: inDim, dilation: 9)
        self._snake1.wrappedValue = HiggsSnake1d(channels: inDim)
        let pad = (stride + 1) / 2 // ceil(stride / 2)
        self._conv1.wrappedValue = HiggsConv1d(
            inChannels: inDim, outChannels: outDim,
            kernelSize: 2 * stride, stride: stride, bias: true, paddingOverride: pad
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = resUnit1(x)
        h = resUnit2(h)
        h = resUnit3(h)
        h = snake1(h)
        h = conv1(h)
        return h
    }
}

/// DAC-style acoustic encoder: waveform ``[B, T, 1]`` -> latent ``[B, T//960, 256]``.
/// Strides ``[8, 5, 4, 2, 3]`` (product 960), channels 1 -> 64 -> 128 -> 256 ->
/// 512 -> 1024 -> 2048 -> 256. Keys: ``conv1``, ``block``, ``snake1``, ``conv2``.
final class HiggsAcousticEncoder: Module {
    @ModuleInfo(key: "conv1") var conv1: HiggsConv1d
    @ModuleInfo(key: "block") var blocks: [HiggsAcousticEncoderBlock]
    @ModuleInfo(key: "snake1") var snake1: HiggsSnake1d
    @ModuleInfo(key: "conv2") var conv2: HiggsConv1d

    private static let strides = [8, 5, 4, 2, 3]
    private static let channels = [64, 128, 256, 512, 1024, 2048]

    override init() {
        self._conv1.wrappedValue = HiggsConv1d(inChannels: 1, outChannels: 64, kernelSize: 7)
        var blocks: [HiggsAcousticEncoderBlock] = []
        for i in 0..<Self.strides.count {
            blocks.append(HiggsAcousticEncoderBlock(
                inDim: Self.channels[i], outDim: Self.channels[i + 1], stride: Self.strides[i]
            ))
        }
        self._blocks.wrappedValue = blocks
        self._snake1.wrappedValue = HiggsSnake1d(channels: 2048)
        self._conv2.wrappedValue = HiggsConv1d(inChannels: 2048, outChannels: 256, kernelSize: 3)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        for b in blocks { h = b(h) }
        h = snake1(h)
        return conv2(h)
    }
}

// MARK: - Residual vector quantizer

/// Single VQ codebook. Keys: ``project_in``, ``codebook``, ``project_out``.
final class HiggsVectorQuantizer: Module {
    @ModuleInfo(key: "project_in") var projectIn: Linear
    @ModuleInfo(key: "codebook") var codebook: Embedding
    @ModuleInfo(key: "project_out") var projectOut: Linear

    init(latentDim: Int = 1024, codebookSize: Int = 1024, codebookDim: Int = 64) {
        self._projectIn.wrappedValue = Linear(latentDim, codebookDim, bias: true)
        self._codebook.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: codebookDim)
        self._projectOut.wrappedValue = Linear(codebookDim, latentDim, bias: true)
    }

    /// ``codes [B, T]`` -> ``[B, T, latentDim]``.
    func decodeCodes(_ codes: MLXArray) -> MLXArray {
        projectOut(codebook(codes))
    }

    /// ``z [., latentDim]`` -> ``[.]`` int32 nearest-neighbor indices.
    func encode(_ z: MLXArray) -> MLXArray {
        let zq = projectIn(z) // [., codebookDim]
        let cw = codebook.weight // [codebookSize, codebookDim]
        let zqSq = MLX.sum(zq * zq, axes: [-1], keepDims: true) // [., 1]
        let cwSq = MLX.sum(cw * cw, axes: [-1]) // [codebookSize]
        let cross = matmul(zq, cw.transposed(-1, -2)) // [., codebookSize]
        let dists = zqSq + cwSq - 2 * cross
        return dists.argMin(axis: -1).asType(.int32)
    }
}

/// 8-codebook residual vector quantizer.
final class HiggsResidualVectorQuantizer: Module {
    @ModuleInfo(key: "quantizers") var quantizers: [HiggsVectorQuantizer]
    let nCodebooks: Int

    init(nCodebooks: Int = 8, latentDim: Int = 1024, codebookSize: Int = 1024, codebookDim: Int = 64) {
        self.nCodebooks = nCodebooks
        self._quantizers.wrappedValue = (0..<nCodebooks).map { _ in
            HiggsVectorQuantizer(latentDim: latentDim, codebookSize: codebookSize, codebookDim: codebookDim)
        }
    }

    /// ``codes [B, T, nCodebooks]`` -> ``[B, T, latentDim]`` (sum of codebooks).
    func decode(_ codes: MLXArray) -> MLXArray {
        // codes: [B, T, N]
        let b = codes.dim(0), t = codes.dim(1)
        var z = MLXArray.zeros([b, t, 1024])
        for i in 0..<nCodebooks {
            let ci = codes[0..., 0..., i] // [B, T]
            z = z + quantizers[i].decodeCodes(ci)
        }
        return z
    }

    /// ``z [B, T, latentDim]`` -> ``[B, T, nCodebooks]`` int32 via greedy
    /// residual quantization.
    func encode(_ z: MLXArray) -> MLXArray {
        var residual = z
        var tokens: [MLXArray] = []
        for vq in quantizers {
            let idx = vq.encode(residual) // [B, T]
            tokens.append(idx)
            residual = residual - vq.decodeCodes(idx)
        }
        return MLX.stacked(tokens, axis: -1).asType(.int32)
    }
}

// MARK: - Tokenizer (decode + encode path)

/// Higgs Audio tokenizer: decode (tokens -> waveform) and encode (waveform ->
/// tokens) paths. The encode path (acoustic encoder + HuBERT semantic model +
/// fusion) is needed only for voice cloning.
final class HiggsAudioCodec: Module, @unchecked Sendable {
    let configuration: HiggsAudioConfig

    // Decode path
    @ModuleInfo(key: "quantizer") var quantizer: HiggsResidualVectorQuantizer
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "acoustic_decoder") var acousticDecoder: HiggsAcousticDecoder

    // Encode path (voice cloning)
    @ModuleInfo(key: "acoustic_encoder") var acousticEncoder: HiggsAcousticEncoder
    @ModuleInfo(key: "semantic_model") var semanticModel: HiggsSemanticModel
    @ModuleInfo(key: "encoder_semantic") var encoderSemantic: HiggsSemanticConvNet
    @ModuleInfo(key: "fc") var fc: Linear

    init(_ config: HiggsAudioConfig) {
        self.configuration = config
        self._quantizer.wrappedValue = HiggsResidualVectorQuantizer()
        self._fc2.wrappedValue = Linear(1024, 256, bias: true)
        self._acousticDecoder.wrappedValue = HiggsAcousticDecoder()
        self._acousticEncoder.wrappedValue = HiggsAcousticEncoder()
        self._semanticModel.wrappedValue = HiggsSemanticModel()
        self._encoderSemantic.wrappedValue = HiggsSemanticConvNet()
        self._fc.wrappedValue = Linear(1024, 1024, bias: true)
    }

    /// ``codes [T, 8]`` -> waveform ``[T*960]`` (float32).
    func decode(_ codes: MLXArray) -> MLXArray {
        // [T, 8] -> [1, T, 8]
        let tokens = codes.expandedDimensions(axis: 0)
        var z = quantizer.decode(tokens) // [1, T, 1024]
        z = fc2(z) // [1, T, 256]
        var wav = acousticDecoder(z) // [1, T*960, 1]
        wav = wav[0, 0..., 0] // [T*960]
        return wav.asType(.float32)
    }

    /// ``waveform [B, T, 1]`` float32 at 24 kHz -> ``[B, T', 8]`` int32 codes,
    /// fusing the DAC acoustic features (24 kHz) with the HuBERT semantic
    /// features (16 kHz) and residual-quantizing the result.
    func encode(_ waveform: MLXArray) -> MLXArray {
        let b = waveform.dim(0)
        let t = waveform.dim(1)
        let wav2d = waveform[0..., 0..., 0].asType(.float32) // [B, T]
        let flat = wav2d.asArray(Float.self) // [B*T]

        // 1. Sinc resample 24 kHz -> 16 kHz, per sample.
        var rows: [[Float]] = []
        for bi in 0..<b {
            let row = Array(flat[(bi * t)..<((bi + 1) * t)])
            rows.append(higgsSincResample(row, origFreq: configuration.sampleRate, newFreq: configuration.semanticSampleRate))
        }
        let baseLen = rows.map { $0.count }.min() ?? 0
        let pad = configuration.downsampleFactor / 2
        let paddedLen = baseLen + 2 * pad
        var audio16k = [Float](repeating: 0, count: b * paddedLen)
        for bi in 0..<b {
            for i in 0..<baseLen { audio16k[bi * paddedLen + pad + i] = rows[bi][i] }
        }
        let audio16kArr = MLXArray(audio16k).reshaped([b, paddedLen]).asType(.float32)

        // 2. Semantic features: HuBERT mean-over-hidden-states -> [::dsf] -> CNN.
        var sem = semanticModel(audio16kArr) // [B, T16//320, 768]
        let dsf = configuration.semanticDownsampleFactor
        if dsf > 1 {
            let frames = sem.dim(1)
            let kept = Swift.stride(from: 0, to: frames, by: dsf).map { Int32($0) }
            sem = MLX.take(sem, MLXArray(kept), axis: 1)
        }
        sem = encoderSemantic(sem) // [B, T', 768]

        // 3. Acoustic features from the 24 kHz waveform.
        let ac = acousticEncoder(waveform.asType(.float32)) // [B, T'', 256]

        // 4. Truncate to a common frame count, fuse, project, quantize.
        let tmin = Swift.min(sem.dim(1), ac.dim(1))
        let semT = sem[0..., 0..<tmin, 0...]
        let acT = ac[0..., 0..<tmin, 0...]
        var emb = MLX.concatenated([acT, semT], axis: -1) // [B, tmin, 1024]
        emb = fc(emb)
        return quantizer.encode(emb) // [B, tmin, 8]
    }

    /// Replicates the Python codec ``sanitize`` for BOTH the decode and encode
    /// tensors. ``raw`` keys have the ``tied.embedding.modality_embeddings.0.model.``
    /// prefix already stripped. Keeps the acoustic encoder/decoder, quantizer,
    /// fc2, the HuBERT semantic model, the semantic CNN, and fc; drops
    /// decoder_semantic/fc1, the masked-spec embed, and VQ bookkeeping.
    static func sanitize(_ raw: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        var posG: MLXArray?
        var posV: MLXArray?

        let keepPrefixes = [
            "acoustic_encoder.", "acoustic_decoder.", "quantizer.", "fc2.",
            "semantic_model.", "encoder_semantic.",
        ]
        let dropPrefixes = ["decoder_semantic.", "fc1."]
        let dropExact = "semantic_model.masked_spec_embed"
        let dropSuffixes = [".embed_avg", ".cluster_size", ".inited"]

        let posGKey = "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0"
        let posVKey = "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1"

        for (k0, value) in raw {
            var k = k0
            if k0 == dropExact { continue }
            if dropPrefixes.contains(where: { k0.hasPrefix($0) }) { continue }
            if !(keepPrefixes.contains(where: { k0.hasPrefix($0) }) || k0 == "fc.weight" || k0 == "fc.bias") { continue }
            if dropSuffixes.contains(where: { k0.hasSuffix($0) }) { continue }

            // Pos-conv weight-norm: collect g/v and collapse after the loop.
            if k0 == posGKey { posG = value; continue }
            if k0 == posVKey { posV = value; continue }

            var v = value
            if k.hasSuffix(".codebook.embed") {
                k = String(k.dropLast("embed".count)) + "weight"
            }

            if v.ndim == 3 {
                if k.hasSuffix(".weight") {
                    if k.contains("conv_t") {
                        // Decoder transposed conv [Cin, Cout, k] -> [Cout, k, Cin].
                        v = v.transposed(1, 2, 0)
                    } else {
                        // Standard conv [Cout, Cin, k] -> [Cout, k, Cin].
                        v = v.transposed(0, 2, 1)
                    }
                } else if k.hasSuffix(".alpha") {
                    // Snake alpha [1, C, 1] -> [1, 1, C].
                    v = v.transposed(0, 2, 1)
                }
            }
            out[k] = v
        }

        // Collapse the pos-conv weight-norm into a single grouped-conv weight:
        //   weight = g * v / ||v||, transposed to MLX [out, kernel, in/groups].
        if let g = posG, let v = posV {
            let norm = MLX.sqrt(MLX.sum(v * v, axes: [0, 1], keepDims: true) + 1e-12)
            var w = g * v / norm
            w = w.transposed(0, 2, 1)
            out["semantic_model.encoder.pos_conv_embed.conv.weight"] = w
        }
        return out
    }
}

// MARK: - Sinc resample (24 kHz -> 16 kHz, torchaudio-compatible)

private func higgsGCD(_ a: Int, _ b: Int) -> Int {
    var (a, b) = (a, b)
    while b != 0 { (a, b) = (b, a % b) }
    return a
}

/// Hann-windowed sinc interpolation resample, matching torchaudio
/// ``functional.resample(method='sinc_interp_hann')`` so encoded reference
/// codes match the Python codec encode path.
func higgsSincResample(
    _ waveform: [Float],
    origFreq: Int,
    newFreq: Int,
    lowpassFilterWidth: Int = 6,
    rolloff: Float = 0.99
) -> [Float] {
    if origFreq == newFreq || waveform.isEmpty { return waveform }
    let g = higgsGCD(origFreq, newFreq)
    let origR = origFreq / g
    let newR = newFreq / g
    let baseFreq = Float(Swift.min(origR, newR)) * rolloff
    let width = Int((Float(lowpassFilterWidth) * Float(origR) / baseFreq).rounded(.up))

    // Build the per-phase resampling kernel [newR][idxCount], where
    // idxCount = 2*width + origR and phase ranges over [0, newR).
    let idxCount = 2 * width + origR
    var kernel = [[Float]](repeating: [Float](repeating: 0, count: idxCount), count: newR)
    let lpw = Float(lowpassFilterWidth)
    for phase in 0..<newR {
        for j in 0..<idxCount {
            let phaseTerm = Float(-phase) / Float(newR) // arange(0,-newR,-1)[phase]
            let idxTerm = Float(j - width) / Float(origR) // arange(-width, width+origR)[j]
            var tv = phaseTerm + idxTerm
            tv *= baseFreq
            tv = max(-lpw, min(lpw, tv))
            let window = pow(cos(tv * .pi / lpw / 2), 2)
            let tpi = tv * .pi
            let sinc = abs(tpi) < 1e-12 ? Float(1.0) : sin(tpi) / tpi
            kernel[phase][j] = sinc * window * (baseFreq / Float(origR))
        }
    }

    let length = waveform.count
    let padLeft = width
    let padRight = width + origR
    var padded = [Float](repeating: 0, count: length + padLeft + padRight)
    for i in 0..<length { padded[i + padLeft] = waveform[i] }
    let outLen = Int((Double(length) * Double(newR) / Double(origR)).rounded(.up))
    var result = [Float](repeating: 0, count: outLen)
    let convLen = padded.count - idxCount + 1

    for phase in 0..<newR {
        var o = 0
        while true {
            let convPos = o * origR
            if convPos >= convLen { break }
            var s = Float(0)
            for j in 0..<idxCount {
                s += padded[convPos + j] * kernel[phase][idxCount - 1 - j]
            }
            let pos = phase + o * newR
            if pos < outLen { result[pos] = s }
            o += 1
        }
    }
    return result
}
