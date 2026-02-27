package main

import (
	"context"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"

	chroma "github.com/amikos-tech/chroma-go/pkg/api/v2"
)

type BM25Doc struct {
	ID   string
	Text string
}

type BM25Index struct {
	docs      []BM25Doc
	tf        []map[string]int
	docLen    []int
	df        map[string]int
	avgDocLen float64
}

func NewBM25Index(docs []BM25Doc) *BM25Index {
	idx := &BM25Index{
		docs:   docs,
		tf:     make([]map[string]int, len(docs)),
		docLen: make([]int, len(docs)),
		df:     map[string]int{},
	}
	var totalLen int
	for i, d := range docs {
		tokens := tokenize(d.Text)
		idx.docLen[i] = len(tokens)
		totalLen += len(tokens)

		tf := map[string]int{}
		seen := map[string]bool{}
		for _, tok := range tokens {
			tf[tok]++
			if !seen[tok] {
				seen[tok] = true
				idx.df[tok]++
			}
		}
		idx.tf[i] = tf
	}
	if len(docs) > 0 {
		idx.avgDocLen = float64(totalLen) / float64(len(docs))
	}
	return idx
}

func (idx *BM25Index) Search(query string, k int) []string {
	if idx == nil || len(idx.docs) == 0 {
		return nil
	}
	qTokens := tokenize(query)
	if len(qTokens) == 0 {
		return nil
	}

	type scored struct {
		id    string
		score float64
	}

	N := float64(len(idx.docs))
	k1 := 1.5
	b := 0.75

	scores := make([]scored, 0, len(idx.docs))
	for i, d := range idx.docs {
		var score float64
		dl := float64(idx.docLen[i])
		tf := idx.tf[i]

		for _, t := range qTokens {
			f := float64(tf[t])
			if f == 0 {
				continue
			}
			df := float64(idx.df[t])
			idf := math.Log(1.0 + (N-df+0.5)/(df+0.5))

			den := f + k1*(1.0-b+b*(dl/idx.avgDocLen))
			score += idf * (f * (k1 + 1.0) / den)
		}
		if score > 0 {
			scores = append(scores, scored{id: d.ID, score: score})
		}
	}

	sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
	if k > len(scores) {
		k = len(scores)
	}
	out := make([]string, 0, k)
	for i := 0; i < k; i++ {
		out = append(out, scores[i].id)
	}
	return out
}

var nonWord = regexp.MustCompile(`[^\p{L}\p{N}]+`)

func tokenize(s string) []string {
	s = strings.ToLower(s)
	s = nonWord.ReplaceAllString(s, " ")
	parts := strings.Fields(s)
	return parts
}

type Retrieved struct {
	ID     string
	Text   string
	Source string
}

func vectorRetrieve(ctx context.Context, c chroma.Collection, query string, k int) ([]Retrieved, error) {
	if c == nil {
		return nil, fmt.Errorf("collection is nil")
	}
	if hfEmbedder == nil {
		return nil, fmt.Errorf("HF embedder not initialized")
	}

	qID := stableID("q", query)
	vecs, err := hfEmbedder.Embed(ctx, []Chunk{{ID: qID, Text: query}})
	if err != nil {
		return nil, err
	}
	qVec := vecs[qID]

	ids, docs, metas, err := chromaQuery(ctx, c, qVec, k)
	if err != nil {
		return nil, err
	}

	out := make([]Retrieved, 0, len(ids))
	for i := range ids {
		r := Retrieved{ID: ids[i]}
		if i < len(docs) {
			r.Text = docs[i]
		}
		if i < len(metas) {
			if s, ok := metas[i]["source"]; ok {
				r.Source = fmt.Sprintf("%v", s)
			}
		}
		out = append(out, r)
	}
	return out, nil
}

func hybridRetrieve(ctx context.Context, c chroma.Collection, query string, k int) ([]Retrieved, error) {
	// Vector top-k
	vecTop, err := vectorRetrieve(ctx, c, query, k)
	if err != nil {
		return nil, err
	}

	// Lexical top-k (BM25) by IDs, then fetch those docs from Chroma by ID
	var lexIDs []string
	if bm25Index != nil {
		lexIDs = bm25Index.Search(query, k)
	}

	lexTop, err := chromaGetByIDs(ctx, c, lexIDs)
	if err != nil {
		return nil, err
	}

	// RRF fuse
	fusedIDs := rrfFuseIDs(lexIDs, idsFromRetrieved(vecTop), 60)
	if k > len(fusedIDs) {
		k = len(fusedIDs)
	}
	finalIDs := fusedIDs[:k]

	// Build map of id->Retrieved from both sources
	m := map[string]Retrieved{}
	for _, r := range lexTop {
		m[r.ID] = r
	}
	for _, r := range vecTop {
		if _, ok := m[r.ID]; !ok {
			m[r.ID] = r
		}
	}

	out := make([]Retrieved, 0, len(finalIDs))
	for _, id := range finalIDs {
		if r, ok := m[id]; ok {
			out = append(out, r)
		}
	}
	return out, nil
}

func idsFromRetrieved(rs []Retrieved) []string {
	out := make([]string, 0, len(rs))
	for _, r := range rs {
		out = append(out, r.ID)
	}
	return out
}

func rrfFuseIDs(a, b []string, k int) []string {
	score := map[string]float64{}
	add := func(list []string) {
		for i, id := range list {
			score[id] += 1.0 / float64(k+i+1)
		}
	}
	add(a)
	add(b)

	uniq := make([]string, 0, len(score))
	for id := range score {
		uniq = append(uniq, id)
	}
	sort.Slice(uniq, func(i, j int) bool { return score[uniq[i]] > score[uniq[j]] })
	return uniq
}