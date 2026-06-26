//
//  HiggsAudioConfig.swift
//  MLXAudio
//
//  Higgs Audio v3 TTS (bosonai/higgs-tts-3-4b): a Qwen3-4B backed
//  conversational TTS model with fused multi-codebook audio token generation,
//  a delay pattern, inline control tokens, and a bundled DAC-style acoustic
//  decoder. Ported from the Python mlx-audio implementation
//  (mlx_audio/tts/models/higgs_audio_v3).
//

import Foundation

/// Text/backbone configuration for Higgs Audio v3 (Qwen3-4B).
public struct HiggsAudioTextConfig: Sendable {
    public var modelType: String = "qwen3"
    public var hiddenSize: Int = 2560
    public var hiddenLayers: Int = 36
    public var intermediateSize: Int = 9728
    public var attentionHeads: Int = 32
    public var kvHeads: Int = 8
    public var headDim: Int = 128
    public var maxPositionEmbeddings: Int = 32768
    public var ropeTheta: Float = 1_000_000
    public var rmsNormEps: Float = 1e-6
    public var vocabSize: Int = 151936
    public var tieWordEmbeddings: Bool = true

    public init() {}
}

/// Top-level configuration for Higgs Audio v3 TTS.
public struct HiggsAudioConfig: Sendable {
    public var modelType: String = "higgs_audio_v3"
    public var text: HiggsAudioTextConfig = .init()
    public var audioTokenId: Int = -100
    public var numCodebooks: Int = 8
    public var codebookSize: Int = 1026
    public var bocTokenId: Int = 1024
    public var eocTokenId: Int = 1025
    public var useDelayPattern: Bool = true
    public var sampleRate: Int = 24_000

    // MARK: Encode path (voice cloning)

    /// Sample rate the bundled HuBERT/Wav2Vec2 semantic model runs at.
    public var semanticSampleRate: Int = 16_000
    /// HuBERT feature-extractor stride product (16 kHz raw -> 50 Hz frames).
    public var downsampleFactor: Int = 320
    /// Stride-slice factor aligning HuBERT frames (50 Hz) to acoustic frames (25 Hz).
    public var semanticDownsampleFactor: Int = 2

    public init() {}

    /// Parse the Hugging Face ``config.json`` layout produced by Boson AI.
    public init(configJSON: [String: Any]) {
        if let textDict = configJSON["text_config"] as? [String: Any] {
            self.text = HiggsAudioTextConfig(textConfig: textDict)
        }
        if let mt = configJSON["model_type"] as? String { self.modelType = mt }
        if let atid = configJSON["audio_token_id"] as? Int { self.audioTokenId = atid }

        let enc = (configJSON["audio_encoder_config"] as? [String: Any]) ?? [:]
        let codebookSize = (configJSON["audio_codebook_size"] as? Int)
            ?? (enc["vocab_size"] as? Int)
            ?? 1026
        self.codebookSize = codebookSize
        self.numCodebooks = (configJSON["audio_num_codebooks"] as? Int)
            ?? (enc["num_codebooks"] as? Int)
            ?? 8
        self.bocTokenId = codebookSize - 2
        self.eocTokenId = codebookSize - 1
        self.useDelayPattern = (configJSON["use_delay_pattern"] as? Bool)
            ?? (enc["use_delay_pattern"] as? Bool)
            ?? true
        if let sr = configJSON["sample_rate"] as? Int { self.sampleRate = sr }
    }
}

extension HiggsAudioTextConfig {
    init(textConfig data: [String: Any]) {
        self.init()
        if let v = data["model_type"] as? String { self.modelType = v }
        if let v = data["hidden_size"] as? Int { self.hiddenSize = v }
        if let v = data["num_hidden_layers"] as? Int { self.hiddenLayers = v }
        if let v = data["intermediate_size"] as? Int { self.intermediateSize = v }
        if let v = data["num_attention_heads"] as? Int { self.attentionHeads = v }
        if let v = data["num_key_value_heads"] as? Int { self.kvHeads = v }
        if let v = data["head_dim"] as? Int { self.headDim = v }
        if let v = data["max_position_embeddings"] as? Int { self.maxPositionEmbeddings = v }
        if let rope = data["rope_parameters"] as? [String: Any], let theta = rope["rope_theta"] as? Double {
            self.ropeTheta = Float(theta)
        } else if let v = data["rope_theta"] as? Double {
            self.ropeTheta = Float(v)
        }
        if let v = data["rms_norm_eps"] as? Double { self.rmsNormEps = Float(v) }
        if let v = data["vocab_size"] as? Int { self.vocabSize = v }
        if let v = data["tie_word_embeddings"] as? Bool { self.tieWordEmbeddings = v }
    }
}
