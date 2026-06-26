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

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, dilation: Int = 1, bias: Bool = true) {
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.stride = stride
        self.padding = (kernelSize - stride) * dilation / 2

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
}

/// 8-codebook residual vector quantizer (decode-only).
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
}

// MARK: - Tokenizer (decode path)

/// Higgs Audio tokenizer decode path.
final class HiggsAudioCodec: Module, @unchecked Sendable {
    @ModuleInfo(key: "quantizer") var quantizer: HiggsResidualVectorQuantizer
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "acoustic_decoder") var acousticDecoder: HiggsAcousticDecoder

    override init() {
        self._quantizer.wrappedValue = HiggsResidualVectorQuantizer()
        self._fc2.wrappedValue = Linear(1024, 256, bias: true)
        self._acousticDecoder.wrappedValue = HiggsAcousticDecoder()
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

    /// Replicates the Python codec ``sanitize`` for the decode-path tensors.
    /// ``raw`` keys have the ``tied.embedding.modality_embeddings.0.model.``
    /// prefix already stripped.
    static func sanitize(_ raw: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in raw {
            // Decode-path prefixes only; drop encoder/semantic/fc (encode path)
            // and VQ bookkeeping.
            let isDecode =
                k.hasPrefix("acoustic_decoder.") ||
                k.hasPrefix("quantizer.") ||
                k.hasPrefix("fc2.")
            if !isDecode { continue }
            if k.hasSuffix(".cluster_size") || k.hasSuffix(".embed_avg") || k.hasSuffix(".inited") { continue }

            var key = k
            var value = v

            if key.hasSuffix(".codebook.embed") {
                key = String(key.dropLast("embed".count)) + "weight"
            }

            if value.ndim == 3 {
                if key.hasSuffix(".weight") {
                    if key.contains("conv_t") {
                        // [Cin, Cout, k] -> [Cout, k, Cin]
                        value = value.transposed(1, 2, 0)
                    } else {
                        // [Cout, Cin, k] -> [Cout, k, Cin]
                        value = value.transposed(0, 2, 1)
                    }
                } else if key.hasSuffix(".alpha") {
                    // [1, C, 1] -> [1, 1, C]
                    value = value.transposed(0, 2, 1)
                }
            }
            out[key] = value
        }
        return out
    }
}
