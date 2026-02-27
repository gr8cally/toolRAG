package main

import (
	"context"
	"fmt"
	"sort"
	"strings"

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

	emb := embeddings.Embedding(embeddingVec)

	_, err := c.Upsert(
		ctx,
		[]string{id},
		[]embeddings.Embedding{emb},
		[]map[string]interface{}{metadata},
		[]string{doc},
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

	q := embeddings.Embedding(queryEmbedding)
	res, err := c.Query(ctx, []embeddings.Embedding{q}, k, nil, nil, nil)
	if err != nil {
		return nil, nil, nil, err
	}

	// chroma-go returns nested results (per query)
	if len(res.IDs) == 0 {
		return []string{}, []string{}, []map[string]interface{}{}, nil
	}

	ids = res.IDs[0]
	if len(res.Documents) > 0 {
		docs = res.Documents[0]
	}
	if len(res.Metadatas) > 0 {
		metas = res.Metadatas[0]
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

	getRes, err := c.Get(ctx, ids, nil, nil, nil)
	if err != nil {
		return nil, err
	}

	// `Get` returns flat arrays aligned by ID order provided.
	out := make([]Retrieved, 0, len(getRes.IDs))
	for i := range getRes.IDs {
		r := Retrieved{
			ID: getRes.IDs[i],
		}
		if i < len(getRes.Documents) {
			r.Text = getRes.Documents[i]
		}
		if i < len(getRes.Metadatas) {
			if s, ok := getRes.Metadatas[i]["source"]; ok {
				r.Source = fmt.Sprintf("%v", s)
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
	if k <= 0 {
		k = 20
	}

	// Pragmatic memory fetch: semantic query against a generic phrase.
	// This keeps the system simple while ensuring stored conversation turns are retrievable later.
	results, err := vectorRetrieve(ctx, conversationCollection, "conversation history", k)
	if err != nil {
		return nil, err
	}

	out := make([]string, 0, len(results))
	for _, r := range results {
		t := strings.TrimSpace(r.Text)
		if t == "" {
			continue
		}
		out = append(out, t)
	}

	// Provide stable output ordering across runs.
	sort.Strings(out)
	return out, nil
}