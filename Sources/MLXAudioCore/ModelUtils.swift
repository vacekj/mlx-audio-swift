import Foundation
import HuggingFace

public enum ModelUtils {
    public static func resolveModelType(
        repoID: Repo.ID,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> String? {
        let modelNameComponents = repoID.name.split(separator: "/").last?.split(separator: "-")
        let modelURL = try await resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken,
            cache: cache
        )
        let configJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: modelURL.appendingPathComponent("config.json")))
        if let config = configJSON as? [String: Any] {
            return (config["model_type"] as? String)
                ?? (config["architecture"] as? String)
                ?? (config["model_version"] as? String)
                ?? modelNameComponents?.first?.lowercased()
        }
        return nil
    }

    /// Resolves a model from cache or downloads it if not cached.
    /// - Parameters:
    ///   - string: The repository name
    ///   - requiredExtension: File extension that must exist for cache to be considered complete (e.g., "safetensors")
    ///   - hfToken: The huggingface token for access to gated repositories, if needed.
    /// - Returns: The model directory URL
    public static func resolveOrDownloadModel(
        repoID: Repo.ID,
        requiredExtension: String,
        additionalMatchingPatterns: [String] = [],
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> URL {
        let client: HubClient
        if let token = hfToken, !token.isEmpty {
            print("Using HuggingFace token from configuration")
            client = HubClient(host: HubClient.defaultHost, bearerToken: token, cache: cache)
        } else {
            client = HubClient(cache: cache)
        }
        let resolvedCache = client.cache ?? cache
        return try await resolveOrDownloadModel(
            client: client,
            cache: resolvedCache,
            repoID: repoID,
            requiredExtension: requiredExtension,
            additionalMatchingPatterns: additionalMatchingPatterns
        )
    }

    /// Resolves a model from cache or downloads it if not cached.
    /// - Parameters:
    ///   - client: The HuggingFace Hub client
    ///   - cache: The HuggingFace cache
    ///   - repoID: The repository ID
    ///   - requiredExtension: File extension that must exist for cache to be considered complete (e.g., "safetensors")
    /// - Returns: The model directory URL
    public static func resolveOrDownloadModel(
        client: HubClient,
        cache: HubCache = .default,
        repoID: Repo.ID,
        requiredExtension: String,
        additionalMatchingPatterns: [String] = [],
        progressHandler: (@MainActor @Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        let normalizedRequiredExtension = requiredExtension.hasPrefix(".")
            ? String(requiredExtension.dropFirst())
            : requiredExtension

        var allowedExtensions: Set<String> = [
            "*.\(normalizedRequiredExtension)",
            "*.safetensors",
            "*.json",
            "*.txt",
            "*.wav",
        ]
        allowedExtensions.formUnion(additionalMatchingPatterns)

        // Fast path: resolve from the local HF hub cache with filesystem checks
        // only. downloadSnapshot otherwise resolves the "main" revision over the
        // network on every call, which stalls the load when offline or slow.
        if let local = locallyCachedSnapshot(
            repoID: repoID, cache: cache, requiredExtension: normalizedRequiredExtension
        ) {
            print("Model loaded from cache snapshot: \(local.path)")
            return local
        }

        // Download into the standard Hugging Face hub cache
        // (models--<org>--<model>/snapshots/<commit>/) and load directly from
        // the snapshot directory — no separate flat copy under mlx-audio/. This
        // keeps a single on-disk copy and matches the Python HF cache layout, so
        // the app's cache management (sizes/delete) sees the real model.
        print("Downloading model \(repoID)...")
        let snapshotDir = try await client.downloadSnapshot(
            of: repoID,
            kind: .model,
            revision: "main",
            matching: Array(allowedExtensions),
            progressHandler: progressHandler ?? { progress in
                print("\(progress.completedUnitCount)/\(progress.totalUnitCount) files")
            }
        )

        // Validate the required file is present and non-zero. Snapshot entries
        // are symlinks into blobs/; URL.resourceValues follows them for the size.
        let files = try? FileManager.default.contentsOfDirectory(
            at: snapshotDir, includingPropertiesForKeys: [.fileSizeKey]
        )
        let hasValidFile = files?.contains { file in
            guard file.pathExtension == normalizedRequiredExtension else { return false }
            let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            return size > 0
        } ?? false

        if !hasValidFile {
            clearHubCache(repoID: repoID, cache: cache)
            throw ModelUtilsError.incompleteDownload(repoID.description)
        }

        print("Model loaded from cache snapshot: \(snapshotDir.path)")
        return snapshotDir
    }

    /// Remove a repo's entry from the standard Hugging Face hub cache.
    private static func clearHubCache(repoID: Repo.ID, cache: HubCache) {
        let hubRepoDir = cache.repoDirectory(repo: repoID, kind: .model)
        if FileManager.default.fileExists(atPath: hubRepoDir.path) {
            print("Clearing Hub cache at: \(hubRepoDir.path)")
            try? FileManager.default.removeItem(at: hubRepoDir)
        }
    }

    /// Find a complete model snapshot in the standard HF hub cache
    /// (`models--<org>--<model>/snapshots/<commit>/`) using filesystem checks
    /// only — no network. Returns the snapshot dir if it has a non-zero file
    /// with ``requiredExtension`` and a config.json, else nil.
    private static func locallyCachedSnapshot(
        repoID: Repo.ID, cache: HubCache, requiredExtension: String
    ) -> URL? {
        let snapshotsDir = cache.repoDirectory(repo: repoID, kind: .model)
            .appendingPathComponent("snapshots")
        guard let snapshots = try? FileManager.default.contentsOfDirectory(
            at: snapshotsDir, includingPropertiesForKeys: [.fileSizeKey]
        ) else { return nil }
        for snapshot in snapshots {
            guard let files = try? FileManager.default.contentsOfDirectory(
                at: snapshot, includingPropertiesForKeys: [.fileSizeKey]
            ) else { continue }
            let hasRequired = files.contains { file in
                guard file.pathExtension == requiredExtension else { return false }
                let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
                return size > 0
            }
            let hasConfig = files.contains { $0.lastPathComponent == "config.json" }
            if hasRequired && hasConfig {
                return snapshot
            }
        }
        return nil
    }
}

public enum ModelUtilsError: LocalizedError {
    case incompleteDownload(String)

    public var errorDescription: String? {
        switch self {
        case .incompleteDownload(let repo):
            return "Downloaded model '\(repo)' has missing or zero-byte weight files. "
                + "The cache has been cleared — please try again."
        }
    }
}
