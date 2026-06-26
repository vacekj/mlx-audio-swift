//
//  HiggsAudioSampler.swift
//  MLXAudio
//
//  Delay-pattern utilities and the SGLang-compatible delayed multi-codebook
//  sampler for Higgs Audio v3. Ported from
//  mlx_audio/tts/models/higgs_audio_v3/generation.py.
//

import Foundation
@preconcurrency import MLX

let higgsStopCode: Int32 = -1

/// Mutable state for the delayed multi-codebook sampler.
struct HiggsSamplerState {
    let numCodebooks: Int
    var delayCount: Int = 0
    var eocCountdown: Int? = nil
    var generationDone: Bool = false
    var lastCodes: MLXArray? = nil

    init(numCodebooks: Int) { self.numCodebooks = numCodebooks }
}

/// Convert delayed rows ``[L, N]`` back to raw codec codes ``[L - N + 1, N]``
/// by gathering the delay-pattern diagonal.
func higgsReverseDelayPattern(_ delayed: MLXArray) -> MLXArray {
    let length = delayed.dim(0)
    let numCodebooks = delayed.dim(1)
    let t = length - numCodebooks + 1
    precondition(t > 0, "delayed rows have L=\(length), N=\(numCodebooks); need L >= N")
    var cols: [MLXArray] = []
    for c in 0..<numCodebooks {
        let col = delayed[c..<(c + t), c..<(c + 1)]
        cols.append(col)
    }
    return MLX.concatenated(cols, axis: 1)
}

/// Top-k masking: keep the ``topK`` highest-probability tokens, mask the rest
/// with ``-inf``.
private func higgsApplyTopK(_ logits: MLXArray, topK: Int) -> MLXArray {
    let v = logits.dim(-1)
    guard topK > 0, topK < v else { return logits }
    let probs = MLX.softmax(logits, axis: -1)
    let sortedAsc = MLX.sorted(probs, axis: -1) // ascending
    // kth-largest probability threshold (broadcastable to [N, 1]).
    let thresh = sortedAsc[0..., (v - topK)..<(v - topK + 1)]
    let below = probs .< thresh
    return MLX.where(below, MLXArray(-Float.infinity), logits)
}

/// Sample ``[N, V]`` logits independently per codebook -> ``[N]`` int32.
func higgsSampleIndependent(_ logits: MLXArray, temperature: Float, topK: Int?) -> MLXArray {
    if temperature <= 1e-5 || topK == 1 {
        return logits.argMax(axis: -1).asType(.int32)
    }
    var l = logits / Float(temperature)
    if let topK {
        l = higgsApplyTopK(l, topK: topK)
    }
    return MLXRandom.categorical(l, axis: -1).asType(.int32)
}

/// Run one delayed multi-codebook sampler step. Returns ``[N]`` int32 delayed
/// codes for this timestep.
func higgsSamplerStep(
    logits: MLXArray, // [N, V]
    state: inout HiggsSamplerState,
    temperature: Float,
    topK: Int?,
    bocId: Int,
    eocId: Int
) -> MLXArray {
    let n = state.numCodebooks
    precondition(logits.dim(0) == n, "logits shape \(logits.shape) incompatible with num_codebooks=\(n)")

    if state.generationDone {
        return MLXArray(Array(repeating: higgsStopCode, count: n))
    }

    var codes = higgsSampleIndependent(logits, temperature: temperature, topK: topK)

    if state.delayCount < n {
        let nextCodebook = state.delayCount + 1
        if nextCodebook < n {
            let positions = MLXArray(Array(0..<n)) // int32 [n]
            let tailMask = positions .>= Int32(nextCodebook)
            codes = MLX.where(tailMask, MLXArray(Int32(bocId)), codes)
        }
        state.delayCount += 1
    } else if let cd = state.eocCountdown {
        let next = cd - 1
        state.eocCountdown = next
        if next <= 0 { state.generationDone = true }
    } else {
        let first = Int(codes[0].item(Int32.self))
        if first == eocId {
            if n <= 2 {
                state.generationDone = true
            } else {
                state.eocCountdown = n - 2
            }
        }
    }

    if !state.generationDone {
        state.lastCodes = codes
    }
    return codes
}
