import Foundation
import HuggingFace

/// Byte-level progress for a model download, flattened out of Foundation's
/// `Progress` so callers (actors, `Sendable` closures) don't have to hop to the
/// main actor just to read it.
public struct ModelDownloadProgress: Sendable, Equatable {
    public let completedBytes: Int64
    public let totalBytes: Int64

    public init(completedBytes: Int64, totalBytes: Int64) {
        self.completedBytes = completedBytes
        self.totalBytes = totalBytes
    }

    /// 0...1, or 0 when the total isn't known yet.
    public var fraction: Double {
        guard totalBytes > 0 else { return 0 }
        return min(1, max(0, Double(completedBytes) / Double(totalBytes)))
    }
}

/// What the local Hugging Face cache holds for a repo.
public enum ModelSnapshotState: Sendable, Equatable {
    /// Nothing on disk.
    case missing
    /// A snapshot directory exists but is missing the weights or config —
    /// an interrupted download. Must be treated as "not downloaded": the
    /// loader will re-fetch it.
    case partial
    /// Complete enough to load without touching the network.
    case complete
}

public enum ModelUtils {
    /// File globs fetched for every model snapshot.
    ///
    /// Single source of truth on purpose: the prefetch path (files only, with
    /// progress) and the load path must agree on exactly which files make a
    /// snapshot complete. If they diverge, a prefetch reports 100% and then the
    /// loader silently downloads more with no progress shown.
    public static let defaultFilePatterns: [String] = [
        "*.safetensors",
        "*.json",
        "*.txt",
        "*.wav",
    ]

    public static func filePatterns(
        requiredExtension: String,
        additional: [String] = []
    ) -> [String] {
        var patterns = Set(defaultFilePatterns)
        patterns.insert("*.\(normalizedExtension(requiredExtension))")
        patterns.formUnion(additional)
        // Sorted so downloads are deterministic and logs are comparable.
        return patterns.sorted()
    }

    // MARK: - Cache locations

    /// Root of the Hugging Face hub cache actually in use. Resolved through
    /// `HubCache` so `HF_HUB_CACHE` / `HF_HOME` are honored — hardcoding
    /// `~/.cache/huggingface/hub` makes cache management look at a directory
    /// the downloader never writes to.
    public static var hubCacheDirectory: URL {
        HubCache.default.cacheDirectory
    }

    /// Directory holding a repo's `blobs/`, `refs/` and `snapshots/`.
    public static func repoDirectory(forRepoId repoId: String, cache: HubCache = .default) -> URL? {
        guard let repoID = Repo.ID(rawValue: repoId) else { return nil }
        return cache.repoDirectory(repo: repoID, kind: .model)
    }

    /// Whether a repo is cached, and whether that cache is usable.
    ///
    /// Deliberately stricter than "the snapshots directory is non-empty": an
    /// interrupted download leaves `config.json` behind with no weights, which
    /// that weaker test reports as downloaded while the loader re-fetches
    /// gigabytes.
    public static func snapshotState(
        forRepoId repoId: String,
        requiredExtension: String = "safetensors",
        cache: HubCache = .default
    ) -> ModelSnapshotState {
        guard let repoID = Repo.ID(rawValue: repoId) else { return .missing }
        return snapshotState(
            repoID: repoID,
            cache: cache,
            requiredExtension: normalizedExtension(requiredExtension)
        )
    }

    /// ``snapshotState(forRepoId:requiredExtension:cache:)`` against a specific
    /// cache root.
    public static func snapshotState(
        forRepoId repoId: String,
        requiredExtension: String = "safetensors",
        cacheDirectory: URL
    ) -> ModelSnapshotState {
        snapshotState(
            forRepoId: repoId,
            requiredExtension: requiredExtension,
            cache: HubCache(cacheDirectory: cacheDirectory)
        )
    }

    /// Bytes of a repo already fetched into partially-downloaded blobs.
    ///
    /// The Hub client resumes an interrupted transfer with a `Range` request
    /// into `blobs/<etag>.incomplete`, and its own progress only counts bytes
    /// pulled in the current session. Adding these makes a resumed download's
    /// progress continue where it stopped instead of restarting near zero.
    public static func incompleteBytes(forRepoId repoId: String, cache: HubCache = .default) -> Int64 {
        guard let repoDir = repoDirectory(forRepoId: repoId, cache: cache) else { return 0 }
        let blobs = repoDir.appendingPathComponent("blobs")
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: blobs, includingPropertiesForKeys: [.fileSizeKey]
        ) else { return 0 }
        return files
            .filter { $0.pathExtension == "incomplete" }
            .reduce(0) { $0 + Int64((try? $1.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0) }
    }

    /// On-disk byte size of a repo's cache entry, measured from `blobs/` where
    /// the real content lives (`snapshots/` is only symlinks into it).
    public static func cachedSizeBytes(forRepoId repoId: String, cache: HubCache = .default) -> Int64 {
        guard let repoDir = repoDirectory(forRepoId: repoId, cache: cache) else { return 0 }
        return directorySizeBytes(repoDir)
    }

    /// Sums regular-file sizes under `url`, following symlinks so a snapshot
    /// entry is measured by its blob target rather than its path length.
    public static func directorySizeBytes(_ url: URL) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else {
            return 0
        }
        var size: Int64 = 0
        var countedBlobs = Set<String>()
        for case let fileURL as URL in enumerator {
            let resolved = fileURL.resolvingSymlinksInPath()
            // Snapshot symlinks point back into blobs/; without deduping, a repo
            // with N revisions is counted N+1 times.
            guard countedBlobs.insert(resolved.path).inserted else { continue }
            guard let attrs = try? fm.attributesOfItem(atPath: resolved.path),
                  (attrs[.type] as? FileAttributeType) == .typeRegular,
                  let fileSize = attrs[.size] as? Int64
            else { continue }
            size += fileSize
        }
        return size
    }

    // MARK: - Download

    /// Downloads a model's files into the Hugging Face cache and returns the
    /// snapshot directory — without constructing the model.
    ///
    /// This is the path to use for "download now" UI and for pre-fetching
    /// before a load: it reports real byte progress, honors cancellation, and
    /// never allocates GPU memory. Loading a multi-gigabyte model just to
    /// populate the cache spikes memory and races other MLX work, since Metal
    /// evaluation is not safe to run concurrently.
    ///
    /// Uses the same globs and the same completeness check as
    /// ``resolveOrDownloadModel(client:cache:repoID:requiredExtension:additionalMatchingPatterns:progressHandler:)``,
    /// so a successful prefetch guarantees the subsequent load hits the local
    /// fast path.
    @discardableResult
    public static func prefetchModel(
        repoId: String,
        requiredExtension: String = "safetensors",
        additionalMatchingPatterns: [String] = [],
        hfToken: String? = nil,
        cache: HubCache = .default,
        progressHandler: (@Sendable (ModelDownloadProgress) -> Void)? = nil
    ) async throws -> URL {
        guard let repoID = Repo.ID(rawValue: repoId) else {
            throw ModelUtilsError.invalidRepository(repoId)
        }
        // Bytes already on disk from an interrupted attempt. The Hub client's
        // progress starts from zero for those files even though it resumes
        // them, so carry the offset to keep a resumed bar where it left off.
        let carried = incompleteBytes(forRepoId: repoId, cache: cache)
        return try await resolveOrDownloadModel(
            client: makeClient(hfToken: hfToken, cache: cache),
            cache: cache,
            repoID: repoID,
            requiredExtension: requiredExtension,
            additionalMatchingPatterns: additionalMatchingPatterns,
            progressHandler: progressHandler.map { adaptProgress($0, carriedBytes: carried) }
        )
    }

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
    ///   - repoID: The repository name
    ///   - requiredExtension: File extension that must exist for cache to be considered complete (e.g., "safetensors")
    ///   - hfToken: The huggingface token for access to gated repositories, if needed.
    /// - Returns: The model directory URL
    public static func resolveOrDownloadModel(
        repoID: Repo.ID,
        requiredExtension: String,
        additionalMatchingPatterns: [String] = [],
        hfToken: String? = nil,
        cache: HubCache = .default,
        progressHandler: (@MainActor @Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        let client = makeClient(hfToken: hfToken, cache: cache)
        let resolvedCache = client.cache ?? cache
        return try await resolveOrDownloadModel(
            client: client,
            cache: resolvedCache,
            repoID: repoID,
            requiredExtension: requiredExtension,
            additionalMatchingPatterns: additionalMatchingPatterns,
            progressHandler: progressHandler
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
        let normalizedRequiredExtension = normalizedExtension(requiredExtension)
        let patterns = filePatterns(
            requiredExtension: normalizedRequiredExtension,
            additional: additionalMatchingPatterns
        )

        // Fast path: resolve from the local HF hub cache with filesystem checks
        // only. downloadSnapshot otherwise resolves the "main" revision over the
        // network on every call, which stalls the load when offline or slow.
        if let local = locallyCachedSnapshot(
            repoID: repoID, cache: cache, requiredExtension: normalizedRequiredExtension
        ) {
            return local
        }

        // Download into the standard Hugging Face hub cache
        // (models--<org>--<model>/snapshots/<commit>/) and load directly from
        // the snapshot directory — no separate flat copy under mlx-audio/. This
        // keeps a single on-disk copy and matches the Python HF cache layout, so
        // the app's cache management (sizes/delete) sees the real model.
        let snapshotDir: URL
        do {
            snapshotDir = try await client.downloadSnapshot(
                of: repoID,
                kind: .model,
                revision: "main",
                matching: patterns,
                progressHandler: progressHandler
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw ModelUtilsError.downloadFailed(repoID.description, error)
        }

        // A cancelled download leaves a half-populated snapshot behind. Surface
        // it as cancellation rather than as a corrupt-model error.
        try Task.checkCancellation()

        // Validate the required file is present and non-zero. Snapshot entries
        // are symlinks into blobs/; URL.resourceValues follows them for the size.
        guard snapshotIsComplete(snapshotDir, requiredExtension: normalizedRequiredExtension) else {
            clearHubCache(repoID: repoID, cache: cache)
            throw ModelUtilsError.incompleteDownload(repoID.description)
        }

        return snapshotDir
    }

    // MARK: - Internals

    private static func makeClient(hfToken: String?, cache: HubCache) -> HubClient {
        // Default URLSession request timeout is 60s of idle. A multi-gigabyte
        // shard on a slow or bursty link can sit quiet longer than that between
        // chunks, and the transfer dies mid-file with "The request timed out."
        // Resource timeout stays long so a legitimate multi-hour download isn't
        // killed for taking time.
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 300
        configuration.timeoutIntervalForResource = 86_400
        configuration.waitsForConnectivity = true
        let session = URLSession(configuration: configuration)

        guard let token = hfToken, !token.isEmpty else {
            return HubClient(session: session, cache: cache)
        }
        return HubClient(
            session: session,
            host: HubClient.defaultHost,
            bearerToken: token,
            cache: cache
        )
    }

    /// Bridges the Hub client's main-actor `Progress` callbacks onto the
    /// `Sendable` value type callers actually want.
    private static func adaptProgress(
        _ handler: @escaping @Sendable (ModelDownloadProgress) -> Void,
        carriedBytes: Int64
    ) -> @MainActor @Sendable (Progress) -> Void {
        { progress in
            let total = progress.totalUnitCount
            handler(
                ModelDownloadProgress(
                    completedBytes: min(total, progress.completedUnitCount + carriedBytes),
                    totalBytes: total
                )
            )
        }
    }

    private static func normalizedExtension(_ requiredExtension: String) -> String {
        requiredExtension.hasPrefix(".") ? String(requiredExtension.dropFirst()) : requiredExtension
    }

    /// Remove a repo's entry from the standard Hugging Face hub cache.
    private static func clearHubCache(repoID: Repo.ID, cache: HubCache) {
        let hubRepoDir = cache.repoDirectory(repo: repoID, kind: .model)
        if FileManager.default.fileExists(atPath: hubRepoDir.path) {
            try? FileManager.default.removeItem(at: hubRepoDir)
        }
    }

    private static func snapshotState(
        repoID: Repo.ID,
        cache: HubCache,
        requiredExtension: String
    ) -> ModelSnapshotState {
        if locallyCachedSnapshot(repoID: repoID, cache: cache, requiredExtension: requiredExtension) != nil {
            return .complete
        }
        let repoDir = cache.repoDirectory(repo: repoID, kind: .model)
        return FileManager.default.fileExists(atPath: repoDir.path) ? .partial : .missing
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
        return snapshots.first { snapshotIsComplete($0, requiredExtension: requiredExtension) }
    }

    /// A snapshot counts as usable when it holds at least one non-empty file
    /// with the required extension. An interrupted download leaves either no
    /// weights file at all or a zero-byte one, so this is the test that
    /// actually distinguishes the two.
    ///
    /// Deliberately does not also demand `config.json`: not every repo ships
    /// one (`beshkenadze/kitten-tts-g2p` names its config `us_bart_config.json`),
    /// and requiring it made those repos fail the local check forever — every
    /// load re-resolved the revision over the network, and a strict
    /// post-download check would delete the files it had just fetched.
    private static func snapshotIsComplete(_ snapshot: URL, requiredExtension: String) -> Bool {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: snapshot, includingPropertiesForKeys: [.fileSizeKey]
        ) else { return false }
        return files.contains { file in
            guard file.pathExtension == requiredExtension else { return false }
            let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            return size > 0
        }
    }
}

public enum ModelUtilsError: LocalizedError {
    case incompleteDownload(String)
    case invalidRepository(String)
    case downloadFailed(String, Error)

    public var errorDescription: String? {
        switch self {
        case .incompleteDownload(let repo):
            return "Downloaded model '\(repo)' has missing or zero-byte weight files. "
                + "The cache has been cleared — please try again."
        case .invalidRepository(let repo):
            return "'\(repo)' is not a valid Hugging Face repository id."
        case .downloadFailed(let repo, let underlying):
            return "Couldn't download '\(repo)': \(underlying.localizedDescription)"
        }
    }
}
