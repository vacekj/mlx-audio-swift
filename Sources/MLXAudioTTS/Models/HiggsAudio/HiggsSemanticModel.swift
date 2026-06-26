//
//  HiggsSemanticModel.swift
//  MLXAudio
//
//  The HuBERT/Wav2Vec2 "semantic model" bundled inside the Higgs codec, used
//  only by the voice-cloning encode path. Takes a 16 kHz mono waveform and
//  returns the mean of all encoder hidden states (feature-projection output +
//  every transformer layer output), which is fused with the DAC acoustic
//  features before RVQ. Mirrors mlx_audio.stt.models.wav2vec (12-layer, 768-d,
//  GQA-free, post-norm) and the Python mean-over-hidden-states recipe.
//
//  Architecture is fixed to the bosonai/higgs-tts-3-4b checkpoint:
//    feature_extractor: conv [1->512 k10 s5] (group-norm, gelu) + 6x [512->512]
//                       (gelu, no norm); strides [5,2,2,2,2,2,2] (product 320).
//    feature_projection: LayerNorm(512) -> Linear(512->768).
//    encoder: weight-normed grouped pos-conv (k128, groups16) + 12 layers,
//             hidden 768, 12 heads, intermediate 3072, eps 1e-5.
//

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Group norm

/// Group norm over ``[B, T, C]`` (channels-last). Used by the first feature
/// extractor conv with ``numGroups == channels`` (per-channel spatial norm).
final class HiggsGroupNorm: Module {
    let numGroups: Int
    let channels: Int
    let eps: Float

    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray

    init(numGroups: Int, channels: Int, eps: Float = 1e-5) {
        self.numGroups = numGroups
        self.channels = channels
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([channels])
        self._bias.wrappedValue = MLXArray.zeros([channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, T, C] -> [B, T, G, C/G]; normalize over (T, C/G).
        let b = x.dim(0), t = x.dim(1)
        let g = numGroups
        let cg = channels / g
        let xr = x.reshaped([b, t, g, cg])
        let mean = MLX.mean(xr, axes: [1, 3], keepDims: true) // [B,1,G,1]
        let centered = xr - mean
        let variance = MLX.mean(centered * centered, axes: [1, 3], keepDims: true)
        let normalized = centered / MLX.sqrt(variance + eps)
        let y = normalized.reshaped([b, t, channels])
        return y * weight + bias
    }
}

// MARK: - Feature extractor

/// One feature-extractor conv layer. Layer 0 is followed by a group norm;
/// layers 1..6 have no norm. All use GELU and zero padding.
final class HiggsFeatureConvLayer: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "layer_norm") var groupNorm: HiggsGroupNorm?

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, normalize: Bool) {
        self._conv.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: kernelSize, stride: stride, padding: 0, bias: false
        )
        if normalize {
            self._groupNorm.wrappedValue = HiggsGroupNorm(numGroups: outChannels, channels: outChannels)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv(x)
        if let groupNorm { y = groupNorm(y) }
        return MLXNN.gelu(y)
    }
}

/// 7-layer CNN feature extractor: raw audio ``[B, T, 1]`` -> ``[B, T', 512]``.
final class HiggsFeatureExtractor: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [HiggsFeatureConvLayer]

    override init() {
        let kernels = [10, 3, 3, 3, 3, 2, 2]
        let strides = [5, 2, 2, 2, 2, 2, 2]
        var layers: [HiggsFeatureConvLayer] = []
        var inCh = 1
        for i in 0..<7 {
            layers.append(HiggsFeatureConvLayer(
                inChannels: inCh, outChannels: 512,
                kernelSize: kernels[i], stride: strides[i], normalize: i == 0
            ))
            inCh = 512
        }
        self._convLayers.wrappedValue = layers
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        convLayers.reduce(x) { $1($0) }
    }
}

// MARK: - Feature projection

final class HiggsFeatureProjection: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "projection") var projection: Linear

    override init() {
        self._layerNorm.wrappedValue = LayerNorm(dimensions: 512, eps: 1e-5)
        self._projection.wrappedValue = Linear(512, 768, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projection(layerNorm(x))
    }
}

// MARK: - Positional convolutional embedding

/// Weight-normed grouped depthwise-style conv (k=128, groups=16) + "same" pad
/// trim + GELU. The weight-norm is collapsed into a plain ``.weight`` at load
/// time (see ``HiggsAudioCodec.sanitize``), so this is a stock grouped conv.
final class HiggsPositionalConvEmbedding: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d

    override init() {
        // numConvPosEmbeddings=128, numConvPosEmbeddingGroups=16, hidden=768.
        self._conv.wrappedValue = Conv1d(
            inputChannels: 768, outputChannels: 768,
            kernelSize: 128, stride: 1, padding: 64, groups: 16, bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv(x)
        // SamePad: kernel 128 is even -> drop one sample from the head.
        if y.dim(1) > 1 {
            y = y[0..., 1..., 0...]
        }
        return MLXNN.gelu(y)
    }
}

// MARK: - Transformer

/// Post-norm self-attention (no GQA): q scaled by head_dim^-0.5, bidirectional
/// (no mask). Keys: ``attention.{k,v,q,out}_proj``.
final class HiggsSemanticAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(hiddenSize: Int = 768, numHeads: Int = 12) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.scale = pow(Float(headDim), -0.5)
        self._qProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._kProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._vProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._outProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), t = x.dim(1)
        let q = (qProj(x) * scale).reshaped(b, t, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(b, t, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, t, numHeads, headDim).transposed(0, 2, 1, 3)
        let attn = MLX.softmax(matmul(q, k.transposed(0, 1, 3, 2)), axis: -1)
        let out = matmul(attn, v).transposed(0, 2, 1, 3).reshaped(b, t, -1)
        return outProj(out)
    }
}

final class HiggsSemanticFeedForward: Module {
    @ModuleInfo(key: "intermediate_dense") var intermediateDense: Linear
    @ModuleInfo(key: "output_dense") var outputDense: Linear

    init(hiddenSize: Int = 768, intermediateSize: Int = 3072) {
        self._intermediateDense.wrappedValue = Linear(hiddenSize, intermediateSize, bias: true)
        self._outputDense.wrappedValue = Linear(intermediateSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outputDense(MLXNN.gelu(intermediateDense(x)))
    }
}

/// Post-norm encoder layer: ``x + attn`` -> LayerNorm -> ``+ ffn`` -> FinalLayerNorm.
final class HiggsSemanticEncoderLayer: Module {
    @ModuleInfo(key: "attention") var attention: HiggsSemanticAttention
    @ModuleInfo(key: "feed_forward") var feedForward: HiggsSemanticFeedForward
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    override init() {
        self._attention.wrappedValue = HiggsSemanticAttention()
        self._feedForward.wrappedValue = HiggsSemanticFeedForward()
        self._layerNorm.wrappedValue = LayerNorm(dimensions: 768, eps: 1e-5)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: 768, eps: 1e-5)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attention(x)
        h = layerNorm(h)
        h = h + feedForward(h)
        return finalLayerNorm(h)
    }
}

/// 12-layer post-norm encoder. Returns ALL hidden states (embed+pos then each
/// layer's output) so the caller can mean them — 13 tensors of ``[B, T, 768]``.
final class HiggsSemanticEncoder: Module {
    @ModuleInfo(key: "pos_conv_embed") var posConvEmbed: HiggsPositionalConvEmbedding
    @ModuleInfo(key: "layers") var layers: [HiggsSemanticEncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(numLayers: Int = 12) {
        self._posConvEmbed.wrappedValue = HiggsPositionalConvEmbedding()
        self._layers.wrappedValue = (0..<numLayers).map { _ in HiggsSemanticEncoderLayer() }
        self._layerNorm.wrappedValue = LayerNorm(dimensions: 768, eps: 1e-5)
    }

    func hiddenStates(_ x: MLXArray) -> [MLXArray] {
        var h = x + posConvEmbed(x)
        var all: [MLXArray] = [h]
        for layer in layers {
            h = layer(h)
            all.append(h)
        }
        return all
    }
}

// MARK: - Semantic model

/// Bundled HuBERT/Wav2Vec2 semantic model: 16 kHz waveform -> mean of hidden
/// states ``[B, T', 768]``.
final class HiggsSemanticModel: Module, @unchecked Sendable {
    @ModuleInfo(key: "feature_extractor") var featureExtractor: HiggsFeatureExtractor
    @ModuleInfo(key: "feature_projection") var featureProjection: HiggsFeatureProjection
    @ModuleInfo(key: "encoder") var encoder: HiggsSemanticEncoder

    override init() {
        self._featureExtractor.wrappedValue = HiggsFeatureExtractor()
        self._featureProjection.wrappedValue = HiggsFeatureProjection()
        self._encoder.wrappedValue = HiggsSemanticEncoder()
    }

    /// ``audio16k``: ``[B, T]`` float at 16 kHz. Returns ``[B, T', 768]``.
    func callAsFunction(_ audio16k: MLXArray) -> MLXArray {
        let b = audio16k.dim(0), t = audio16k.dim(1)
        let x = audio16k.reshaped([b, t, 1]) // [B, T, 1]
        let feat = featureExtractor(x) // [B, T', 512]
        let proj = featureProjection(feat) // [B, T', 768]
        let hidden = encoder.hiddenStates(proj) // [13][B, T', 768]
        let stacked = MLX.stacked(hidden, axis: 0) // [13, B, T', 768]
        return MLX.mean(stacked, axis: 0) // [B, T', 768]
    }
}

// MARK: - Semantic encoder CNN (encoder_semantic)

/// Processes HuBERT features before fusion with acoustic features. All convs
/// use ELU. Matches ``mlx_audio.codec.models.higgs_audio.semantic`` with the
/// Higgs defaults: 2 conv blocks, strides/dilations/ratios all 1, 768-dim.
final class HiggsSemanticResidualUnit: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    init(dim: Int = 768, dilation: Int = 1, kernelSize: Int = 3) {
        let pad = (kernelSize - 1) * dilation / 2
        self._conv1.wrappedValue = Conv1d(
            inputChannels: dim, outputChannels: dim,
            kernelSize: kernelSize, stride: 1, padding: pad, dilation: dilation, bias: false
        )
        self._conv2.wrappedValue = Conv1d(
            inputChannels: dim, outputChannels: dim, kernelSize: 1, bias: false
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = MLXNN.elu(x)
        y = conv1(y)
        y = MLXNN.elu(y)
        y = conv2(y)
        return x + y
    }
}

final class HiggsSemanticConvBlock: Module {
    @ModuleInfo(key: "res_units") var resUnits: [HiggsSemanticResidualUnit]
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(dim: Int = 768, stride: Int = 1, dilation: Int = 1, kernelSize: Int = 3, unitKernelSize: Int = 3) {
        self._resUnits.wrappedValue = [
            HiggsSemanticResidualUnit(dim: dim, dilation: dilation, kernelSize: unitKernelSize),
            HiggsSemanticResidualUnit(dim: dim, dilation: dilation, kernelSize: unitKernelSize),
        ]
        let pad = (kernelSize - 1) // 2
        self._conv.wrappedValue = Conv1d(
            inputChannels: dim, outputChannels: dim,
            kernelSize: kernelSize, stride: stride, padding: pad, bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for ru in resUnits { y = ru(y) }
        return conv(y)
    }
}

/// ``encoder_semantic``: HuBERT features ``[B, T, 768]`` -> ``[B, T, 768]``.
final class HiggsSemanticConvNet: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "conv_blocks") var convBlocks: [HiggsSemanticConvBlock]

    init(hiddenSize: Int = 768, strides: [Int] = [1, 1], dilations: [Int] = [1, 1]) {
        self._conv.wrappedValue = Conv1d(
            inputChannels: hiddenSize, outputChannels: hiddenSize,
            kernelSize: 3, stride: 1, padding: 1, bias: false
        )
        var blocks: [HiggsSemanticConvBlock] = []
        for i in 0..<strides.count {
            blocks.append(HiggsSemanticConvBlock(
                dim: hiddenSize, stride: strides[i], dilation: dilations[i]
            ))
        }
        self._convBlocks.wrappedValue = blocks
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv(x)
        for b in convBlocks { y = b(y) }
        return y
    }
}
