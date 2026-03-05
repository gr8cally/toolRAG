package main

import (
	"context"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"

	chroma "github.com/amikos-tech/chroma-go/pkg/api/v2"
)

func loadDocumentsFromDataDir(ctx context.Context) error {
	dataDir := currentConfig.RAGDataDir
	if dataDir == "" {
		dataDir = "./data"
	}

	// Ensure data directory exists
	if _, err := os.Stat(dataDir); os.IsNotExist(err) {
		log.Printf("Data directory does not exist, creating: %s", dataDir)
		if err := os.MkdirAll(dataDir, 0755); err != nil {
			return fmt.Errorf("failed to create data directory: %w", err)
		}
		return nil
	}

	if ragDocsCollection == nil {
		return fmt.Errorf("ragDocsCollection not initialized")
	}

	if hfEmbedderConcrete == nil {
		return fmt.Errorf("HF embedder not initialized")
	}

	chunkSize := currentConfig.ChunkLength
	if chunkSize <= 0 {
		chunkSize = 800
	}

	var corpus []BM25Doc
	documentChunks := 0

	err := filepath.WalkDir(dataDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}

		raw, err := os.ReadFile(path)
		if err != nil {
			log.Printf("Failed to read file %s: %v", path, err)
			return nil
		}

		text := strings.TrimSpace(string(raw))
		if text == "" {
			return nil
		}

		// Strategy 1: Check if document already exists in ChromaDB
		// Check if any chunks from this file are already indexed
		firstChunkID := stableID("rag", path, "0")
		existingDocs, err := ragDocsCollection.Get(ctx,
			chroma.WithIDs(chroma.DocumentID(firstChunkID)))

		if err == nil && existingDocs.Count() > 0 {
			// Document already indexed, load existing chunks for BM25 index
			log.Printf("Document %s already indexed, loading from ChromaDB", filepath.Base(path))

			// Get all chunks for this file to build BM25 index
			allChunks, err := ragDocsCollection.Get(ctx,
				chroma.WithWhere(chroma.EqString("source", path)))

			if err == nil {
				resultIDs := allChunks.GetIDs()
				resultDocs := allChunks.GetDocuments()
				for i := range resultIDs {
					if i < len(resultDocs) {
						corpus = append(corpus, BM25Doc{
							ID:   string(resultIDs[i]),
							Text: resultDocs[i].ContentString(),
						})
						documentChunks++
					}
				}
			}
			return nil
		}

		chunks := chunkText(text, chunkSize)
		if len(chunks) == 0 {
			return nil
		}

		embedInputs := make([]Chunk, 0, len(chunks))
		for i, c := range chunks {
			id := stableID("rag", path, fmt.Sprintf("%d", i))
			embedInputs = append(embedInputs, Chunk{ID: id, Text: c})
		}

		// Strategy 2: Use cached embeddings to avoid redundant API calls
		modelName := currentConfig.EmbedModelName
		if modelName == "" {
			modelName = "sentence-transformers/all-MiniLM-L6-v2"
		}

		vecs, err := embedWithCache(
			ctx,
			hfEmbedderConcrete,
			embedInputs,
			filepath.Base(path),
			text,
			chunkSize,
			modelName,
		)
		if err != nil {
			return fmt.Errorf("embedding %s: %w", path, err)
		}

		for i, c := range chunks {
			id := embedInputs[i].ID
			vec := vecs[id]
			meta := map[string]interface{}{
				"source": path,
				"type":   "document",
				"chunk":  i,
			}
			if err := chromaUpsert(ctx, ragDocsCollection, id, c, vec, meta); err != nil {
				log.Printf("Warning: upsert failed for %s chunk %d: %v", path, i, err)
				continue
			}
			documentChunks++

			corpus = append(corpus, BM25Doc{ID: id, Text: c})
		}

		return nil
	})
	if err != nil {
		return err
	}

	if documentChunks > 0 {
		log.Printf("Indexed %d chunks from %s", documentChunks, dataDir)
	}

	// Build BM25 corpus for hybrid retrieval
	bm25Index = NewBM25Index(corpus)

	return nil
}
