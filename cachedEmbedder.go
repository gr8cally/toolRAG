package main

import (
    "context"
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "fmt"
    "os"
    "path/filepath"
)

type EmbeddingMap map[string][]float32

type embedCacheFile struct {
	Version    int                  `json:"version"`
	Key        string               `json:"key"`   // identifies doc+chunking
	Model      string               `json:"model"` // embed model name (optional)
	Embeddings map[string][]float32 `json:"embeddings"`
}

// stable cache key: fileName + content hash + chunking params (and optionally model)
func makeEmbedCacheKey(fileName string, content string, chunkSize int, model string) string {
	sum := sha256.Sum256([]byte(content))
	return fmt.Sprintf("%s|sha256:%s|chunk:%d|model:%s", fileName, hex.EncodeToString(sum[:]), chunkSize, model)
}

func loadEmbeddingsFromFile(path string) (EmbeddingMap, *embedCacheFile, bool, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil, false, nil
		}
		return nil, nil, false, err
	}

	var cf embedCacheFile
	if err := json.Unmarshal(b, &cf); err != nil {
		return nil, nil, false, err
	}
	if cf.Embeddings == nil {
		return nil, &cf, false, nil
	}
	return EmbeddingMap(cf.Embeddings), &cf, true, nil
}

func saveEmbeddingsToFileAtomic(path string, cf embedCacheFile) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"

	b, err := json.MarshalIndent(cf, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Main helper: load if possible, else compute via embedFn and save.
func getEmbeddingsCached(
    ctx context.Context,
    cachePath string,
    cacheKey string,
    model string,
    embedFn func(context.Context) (map[string][]float32, error),
) (map[string][]float32, error) {

	emb, cf, ok, err := loadEmbeddingsFromFile(cachePath)
	if err != nil {
		return nil, err
	}
	if ok && cf != nil && cf.Key == cacheKey {
		// Cache hit
		return map[string][]float32(emb), nil
	}

	// Cache miss → run real embedding
	out, err := embedFn(ctx)
	if err != nil {
		return nil, err
	}

	// Save
	_ = saveEmbeddingsToFileAtomic(cachePath, embedCacheFile{
		Version:    1,
		Key:        cacheKey,
		Model:      model,
		Embeddings: out,
	})
    return out, nil
}

// embedWithCache wraps Embedder.Embed with a tiny on-disk JSON cache.
// Behavior is controlled by ENV var EMBED_CACHE_MODE:
//   - "off":  always call API (never load/save cache)
//   - "load": only load from cache; error if not found or key mismatch
//   - "auto" (default): load if key matches, else call API and save
// The cache key is derived from fileName, content hash, chunking, and model name.
func embedWithCache(
    ctx context.Context,
    embedder Embedder,
    chunks []Chunk,
    fileName string,
    contentStr string,
    chunkSize int,
    modelName string,
) (map[string][]float32, error) {
    mode := os.Getenv("EMBED_CACHE_MODE") // "auto" | "load" | "off"
    if mode == "" {
        mode = "auto"
    }
    cachePath := "tmp/embeddings_cache.json"

    cacheKey := makeEmbedCacheKey(fileName, contentStr, chunkSize, modelName)

    switch mode {
    case "off":
        // Always call API
        return embedder.Embed(ctx, chunks)

    case "load":
        // Never call API, only load
        loaded, cf, ok, err := loadEmbeddingsFromFile(cachePath)
        if err != nil {
            return nil, fmt.Errorf("failed to load embeddings cache: %w", err)
        }
        if !ok || cf == nil || cf.Key != cacheKey {
            return nil, fmt.Errorf("no matching cached embeddings found (set EMBED_CACHE_MODE=auto to generate once)")
        }
        return map[string][]float32(loaded), nil

    default: // "auto"
        return getEmbeddingsCached(ctx, cachePath, cacheKey, modelName, func(ctx context.Context) (map[string][]float32, error) {
            return embedder.Embed(ctx, chunks)
        })
    }
}
