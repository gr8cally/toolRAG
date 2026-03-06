package main

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strings"
	"time"

	chroma "github.com/amikos-tech/chroma-go/pkg/api/v2"
	"github.com/amikos-tech/chroma-go/pkg/embeddings"
)

func chromaUpsert(ctx context.Context, c chroma.Collection, id string, doc string, embeddingVec []float32, metadata map[string]interface{}) error {
	if c == nil {
		return fmt.Errorf("collection is nil")
	}
	if id == "" {
		return fmt.Errorf("empty id")
	}
	if embeddingVec == nil {
		return fmt.Errorf("nil embedding")
	}

	// Build a single DocumentMetadata with all attributes
	// Each key-value pair becomes an attribute within the same metadata object
	attrs := make([]*chroma.MetaAttribute, 0, len(metadata))
	for k, v := range metadata {
		// Convert all values to strings for simplicity
		// ChromaDB also supports int, float, and bool, but string works universally
		valString := fmt.Sprintf("%v", v)
		attrs = append(attrs, chroma.NewStringAttribute(k, valString))
	}

	// Create a single metadata object with all attributes
	meta := chroma.NewDocumentMetadata(attrs...)
	emb := embeddings.NewEmbeddingFromFloat32(embeddingVec)

	// Upsert with 1 ID, 1 embedding, 1 metadata, 1 text
	// All arrays must have the same length (1 in this case)
	err := c.Upsert(
		ctx,
		chroma.WithIDs(chroma.DocumentID(id)),
		chroma.WithEmbeddings(emb),
		chroma.WithMetadatas(meta),
		chroma.WithTexts(doc),
	)
	return err
}

func chromaQuery(ctx context.Context, c chroma.Collection, queryEmbedding []float32, k int) (ids []string, docs []string, metas []map[string]interface{}, err error) {
	if c == nil {
		return nil, nil, nil, fmt.Errorf("collection is nil")
	}
	if k <= 0 {
		k = 3
	}

	q := embeddings.NewEmbeddingFromFloat32(queryEmbedding)
	res, err := c.Query(
		ctx,
		chroma.WithQueryEmbeddings([]embeddings.Embedding{q}...),
		chroma.WithNResults(k),
		chroma.WithIncludeQuery(chroma.IncludeDocuments, chroma.IncludeMetadatas))
	if err != nil {
		return nil, nil, nil, err
	}

	// chroma-go v2 returns results with Get methods
	idGroups := res.GetIDGroups()
	if len(idGroups) == 0 {
		return []string{}, []string{}, []map[string]interface{}{}, nil
	}

	// Convert DocumentIDs to []string
	idGroup := idGroups[0]
	ids = make([]string, len(idGroup))

	for i, id := range idGroup {
		ids[i] = string(id)
	}

	// Convert Documents to []string
	docGroups := res.GetDocumentsGroups()
	if len(docGroups) > 0 {
		docGroup := docGroups[0]
		docs = make([]string, len(docGroup))
		for i, doc := range docGroup {
			docs[i] = doc.ContentString()
		}
	}

	// Convert DocumentMetadatas to []map[string]interface{}
	metaGroups := res.GetMetadatasGroups()
	if len(metaGroups) > 0 {
		metaGroup := metaGroups[0]
		metas = make([]map[string]interface{}, len(metaGroup))
		for i, meta := range metaGroup {
			// Convert DocumentMetadata interface to map
			m := make(map[string]interface{})
			// Try common keys that might be present
			if val, ok := meta.GetRaw("source"); ok {
				m["source"] = val
			}
			if val, ok := meta.GetRaw("timestamp"); ok {
				m["timestamp"] = val
			}
			if val, ok := meta.GetRaw("type"); ok {
				m["type"] = val
			}
			metas[i] = m
		}
	}
	return ids, docs, metas, nil
}

func chromaGetByIDs(ctx context.Context, c chroma.Collection, ids []string) ([]Retrieved, error) {
	if c == nil {
		return nil, fmt.Errorf("collection is nil")
	}
	if len(ids) == 0 {
		return []Retrieved{}, nil
	}

	// Convert []string to variadic DocumentID arguments
	docIDs := make([]chroma.DocumentID, len(ids))
	for i, id := range ids {
		docIDs[i] = chroma.DocumentID(id)
	}

	getRes, err := c.Get(ctx, chroma.WithIDs(docIDs...))
	if err != nil {
		return nil, err
	}

	// `Get` returns flat arrays aligned by ID order provided.
	resultIDs := getRes.GetIDs()
	resultDocs := getRes.GetDocuments()
	resultMetas := getRes.GetMetadatas()

	out := make([]Retrieved, 0, len(resultIDs))
	for i := range resultIDs {
		r := Retrieved{
			ID: string(resultIDs[i]), // Convert DocumentID to string
		}
		if i < len(resultDocs) {
			r.Text = resultDocs[i].ContentString() // Use ContentString() method
		}
		if i < len(resultMetas) {
			if s, ok := resultMetas[i].GetRaw("source"); ok {
				r.Source = fmt.Sprintf("%v", s)
			}
		}
		out = append(out, r)
	}

	return out, nil
}

// chromaGetByFilter fetches all documents from c that match the given metadata key=value filter.
// Timestamps in metadata are parsed and set on Retrieved.Timestamp.
func chromaGetByFilter(ctx context.Context, c chroma.Collection, key, value string) ([]Retrieved, error) {
	if c == nil {
		return nil, fmt.Errorf("collection is nil")
	}

	res, err := c.Get(ctx,
		chroma.WithWhere(chroma.EqString(key, value)),
		chroma.WithInclude(chroma.IncludeDocuments, chroma.IncludeMetadatas),
	)
	if err != nil {
		return nil, err
	}

	resultIDs := res.GetIDs()
	resultDocs := res.GetDocuments()
	resultMetas := res.GetMetadatas()

	out := make([]Retrieved, 0, len(resultIDs))
	for i := range resultIDs {
		r := Retrieved{ID: string(resultIDs[i])}
		if i < len(resultDocs) {
			r.Text = resultDocs[i].ContentString()
		}
		if i < len(resultMetas) {
			if v, ok := resultMetas[i].GetRaw("timestamp"); ok {
				if ts, err := time.Parse(time.RFC3339, fmt.Sprintf("%v", v)); err == nil {
					r.Timestamp = ts
				}
			}
		}
		out = append(out, r)
	}
	return out, nil
}

func loadRecentConversationHistory(ctx context.Context, k int) ([]string, error) {
	if conversationCollection == nil {
		return []string{}, nil
	}

	results, err := chromaGetByFilter(ctx, conversationCollection, "type", "conversation")
	if err != nil {
		return nil, err
	}

	// Sort chronologically by stored timestamp.
	sort.Slice(results, func(i, j int) bool {
		return results[i].Timestamp.Before(results[j].Timestamp)
	})

	// Apply limit after sorting so we keep the most recent k turns.
	if k > 0 && len(results) > k {
		results = results[len(results)-k:]
	}

	out := make([]string, 0, len(results))
	for _, r := range results {
		t := strings.TrimSpace(r.Text)
		if t == "" {
			continue
		}
		out = append(out, t)
	}
	return out, nil
}

func storeConversationHistory(ctx context.Context, userMsg, assistantMsg string) {
	if conversationCollection == nil || hfEmbedderConcrete == nil {
		return
	}

	conversation := fmt.Sprintf("User: %s\nAssistant: %s", userMsg, assistantMsg)
	id := stableID("conv", time.Now().Format(time.RFC3339Nano), userMsg, assistantMsg)

	vecs, err := hfEmbedderConcrete.Embed(ctx, []Chunk{{ID: id, Text: conversation}})
	if err != nil {
		log.Printf("Warning: Failed to embed conversation: %v", err)
		return
	}

	meta := map[string]interface{}{
		"type":      "conversation",
		"timestamp": time.Now().Format(time.RFC3339),
	}
	if err := chromaUpsert(ctx, conversationCollection, id, conversation, vecs[id], meta); err != nil {
		log.Printf("Warning: Failed to store conversation: %v", err)
	}
}
