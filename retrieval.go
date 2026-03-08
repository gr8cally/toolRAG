package main

import (
	"context"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
	"time"

	chroma "github.com/amikos-tech/chroma-go/pkg/api/v2"
)

// ------------------
// BM25 Lexical Search
// ------------------

// BM25Doc represents a document in the BM25 index.
// Used for lexical (keyword-based) search to complement vector search.
type BM25Doc struct {
	ID   string // Unique document/chunk identifier
	Text string // Document content for lexical matching
}

// BM25Index implements the BM25 ranking algorithm for lexical search.
// BM25 is a probabilistic retrieval model that ranks documents based on
// term frequency (TF) and inverse document frequency (IDF).
type BM25Index struct {
	docs      []BM25Doc        // All documents in the corpus
	tf        []map[string]int // Term frequency per document (tf[docIdx][term] = count)
	docLen    []int            // Length of each document in tokens
	df        map[string]int   // Document frequency (how many docs contain each term)
	avgDocLen float64          // Average document length across corpus
}

// NewBM25Index constructs a BM25 index from a corpus of documents.
// This pre-computes all the statistics needed for BM25 scoring:
// - Term frequencies (TF) for each document
// - Document frequencies (DF) across the corpus
// - Average document length
// These statistics enable fast BM25 ranking at query time.
func NewBM25Index(docs []BM25Doc) *BM25Index {
	idx := &BM25Index{
		docs:   docs,
		tf:     make([]map[string]int, len(docs)),
		docLen: make([]int, len(docs)),
		df:     map[string]int{},
	}

	var totalLen int

	// Build term frequency and document frequency statistics
	for i, d := range docs {
		tokens := tokenize(d.Text)
		idx.docLen[i] = len(tokens)
		totalLen += len(tokens)

		// Count term frequency (TF) for this document
		tf := map[string]int{}
		seen := map[string]bool{}
		for _, tok := range tokens {
			tf[tok]++
			// For document frequency (DF), count each term only once per document
			if !seen[tok] {
				seen[tok] = true
				idx.df[tok]++ // Increment DF for this term
			}
		}
		idx.tf[i] = tf
	}

	// Calculate average document length (used in BM25 normalization)
	if len(docs) > 0 {
		idx.avgDocLen = float64(totalLen) / float64(len(docs))
	}
	return idx
}

// Search performs BM25 ranking on the query and returns top k document IDs.
// BM25 formula: score = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
// where:
//   - qi = query term i
//   - f(qi, D) = frequency of qi in document D
//   - |D| = length of document D
//   - avgdl = average document length
//   - k1 = term frequency saturation parameter (typically 1.2-2.0)
//   - b = length normalization parameter (typically 0.75)
//
// Returns document IDs ranked by relevance score (highest first).
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

	// BM25 parameters
	N := float64(len(idx.docs)) // Total number of documents
	k1 := 1.5                   // Term frequency saturation (controls TF scaling)
	b := 0.75                   // Length normalization (0 = no norm, 1 = full norm)

	scores := make([]scored, 0, len(idx.docs))

	// Score each document against the query
	for i, d := range idx.docs {
		var score float64
		dl := float64(idx.docLen[i]) // Document length
		tf := idx.tf[i]              // Term frequencies for this doc

		// Calculate BM25 score by summing contributions from each query term
		for _, t := range qTokens {
			f := float64(tf[t]) // Term frequency in this document
			if f == 0 {
				continue // Term not in document, skip
			}

			// Calculate IDF (Inverse Document Frequency)
			// Penalizes common terms, rewards rare terms
			df := float64(idx.df[t])
			idf := math.Log(1.0 + (N-df+0.5)/(df+0.5))

			// Calculate BM25 term score with length normalization
			den := f + k1*(1.0-b+b*(dl/idx.avgDocLen))
			score += idf * (f * (k1 + 1.0) / den)
		}

		// Only include documents with non-zero scores
		if score > 0 {
			scores = append(scores, scored{id: d.ID, score: score})
		}
	}

	// Sort by score descending (best matches first)
	sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })

	// Return top k document IDs
	if k > len(scores) {
		k = len(scores)
	}
	out := make([]string, 0, k)
	for i := 0; i < k; i++ {
		out = append(out, scores[i].id)
	}
	return out
}

// ------------------
// Tokenization
// ------------------

// nonWord matches sequences of characters that are NOT letters or numbers.
// Used to split text into word tokens for BM25 indexing.
var nonWord = regexp.MustCompile(`[^\p{L}\p{N}]+`)
var edgeNonWord = regexp.MustCompile(`^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$`)

// tokenize converts text into lowercase word tokens.
// Steps:
//  1. Convert to lowercase for case-insensitive matching
//  2. Replace all non-word characters (punctuation, etc.) with spaces
//  3. Split on whitespace to get individual tokens
//
// Example: "Hello, World! 123" → ["hello", "world", "123"]
func tokenize(s string) []string {
	s = strings.ToLower(s)
	rawFields := strings.Fields(s)
	if len(rawFields) == 0 {
		return nil
	}

	out := make([]string, 0, len(rawFields)*2)
	for _, field := range rawFields {
		normalized := edgeNonWord.ReplaceAllString(field, "")
		if normalized == "" {
			continue
		}

		// Preserve full identifier-like tokens so values like
		// "do-not-index-unique-12345" can be exact-matched.
		if strings.ContainsAny(normalized, "-_") {
			out = append(out, normalized)
		}

		parts := strings.Fields(nonWord.ReplaceAllString(normalized, " "))
		out = append(out, parts...)
	}
	return out
}

// ------------------
// Vector & Hybrid Retrieval
// ------------------

// Retrieved represents a document/chunk retrieved from the vector store.
// Contains both the content and metadata needed for display and ranking.
type Retrieved struct {
	ID        string    // Unique chunk/document identifier
	Text      string    // Document content
	Source    string    // Source file path (from metadata)
	Timestamp time.Time // Stored timestamp (from metadata), zero if absent
}

// vectorRetrieve performs semantic search using vector embeddings.
// This is the "vector search" component of hybrid retrieval.
//
// Process:
//  1. Embed the query text using HuggingFace embedder
//  2. Search ChromaDB for the k nearest neighbors by cosine similarity
//  3. Return retrieved documents with metadata
//
// Unlike BM25 (lexical search), vector search captures semantic meaning,
// so it can match documents even when they don't share exact keywords.
//
// Example: Query "ML models" can match documents about "machine learning algorithms"
func vectorRetrieve(ctx context.Context, c chroma.Collection, query string, k int) ([]Retrieved, error) {
	if c == nil {
		return nil, fmt.Errorf("collection is nil")
	}

	if hfEmbedderConcrete == nil {
		return nil, fmt.Errorf("HF embedder not initialized")
	}

	// Generate embedding for the query
	qID := stableID("q", query)
	vecs, err := hfEmbedderConcrete.Embed(ctx, []Chunk{{ID: qID, Text: query}})
	if err != nil {
		return nil, err
	}
	qVec := vecs[qID]

	// Search ChromaDB for top k similar documents by vector similarity
	ids, docs, metas, err := chromaQuery(ctx, c, qVec, k)
	if err != nil {
		return nil, err
	}

	// Package results into Retrieved structs
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

// hybridRetrieve combines vector search and BM25 lexical search using Reciprocal Rank Fusion (RRF).
// This "best of both worlds" approach leverages:
//   - Vector search: Captures semantic meaning and conceptual similarity
//   - BM25 search: Captures exact keyword matches and lexical patterns
//
// Process:
//  1. Perform vector search to get top k semantically similar documents
//  2. Perform BM25 lexical search to get top k keyword-matching documents
//  3. Fuse both result lists using RRF (Reciprocal Rank Fusion)
//  4. Return top k documents from the fused ranking
//
// RRF combines rankings by summing reciprocal ranks, which is more robust
// than score-based fusion since it doesn't require score normalization.
//
// Example: A query about "neural networks" might:
//   - Vector search finds: docs about "deep learning", "AI models"
//   - BM25 finds: docs containing exact phrase "neural networks"
//   - RRF combines both to return the most relevant overall results
func hybridRetrieve(ctx context.Context, c chroma.Collection, query string, k int) ([]Retrieved, error) {
	// Step 1: Vector search for semantic similarity
	vecTop, err := vectorRetrieve(ctx, c, query, k)
	if err != nil {
		return nil, err
	}

	// Step 2: BM25 lexical search for keyword matching
	var lexIDs []string
	if bm25Index != nil {
		lexIDs = bm25Index.Search(query, k)
	}

	// Fetch full document content for BM25 results from ChromaDB
	lexTop, err := chromaGetByIDs(ctx, c, lexIDs)
	if err != nil {
		return nil, err
	}

	// Step 3: Fuse rankings using Reciprocal Rank Fusion (RRF)
	// k=60 is a common RRF constant that balances ranking contributions
	fusedIDs := rrfFuseIDs(lexIDs, idsFromRetrieved(vecTop), 60)
	if k > len(fusedIDs) {
		k = len(fusedIDs)
	}
	finalIDs := fusedIDs[:k]

	// Step 4: Build a map of document ID -> content from both sources
	m := map[string]Retrieved{}
	for _, r := range lexTop {
		m[r.ID] = r
	}
	for _, r := range vecTop {
		// Don't overwrite if lexTop already has this doc
		if _, ok := m[r.ID]; !ok {
			m[r.ID] = r
		}
	}

	// Step 5: Return documents in fused ranking order
	out := make([]Retrieved, 0, len(finalIDs))
	for _, id := range finalIDs {
		if r, ok := m[id]; ok {
			out = append(out, r)
		}
	}
	return out, nil
}

// ------------------
// Helper Functions
// ------------------

// idsFromRetrieved extracts document IDs from a list of Retrieved documents.
// Useful for converting Retrieved results back to ID lists for fusion.
func idsFromRetrieved(rs []Retrieved) []string {
	out := make([]string, 0, len(rs))
	for _, r := range rs {
		out = append(out, r.ID)
	}
	return out
}

// rrfFuseIDs combines two ranked lists of document IDs using Reciprocal Rank Fusion (RRF).
//
// RRF formula: RRF_score(d) = Σ 1 / (k + rank_i(d))
// where:
//   - d = document
//   - rank_i(d) = position of document d in ranking i (0-indexed)
//   - k = constant to prevent division by small numbers (typically 60)
//
// Process:
//  1. For each document in each ranking, add its reciprocal rank to its score
//  2. Sort all documents by their combined RRF scores (descending)
//  3. Return the fused ranking
//
// Benefits of RRF:
//   - Doesn't require score normalization (only uses ranks)
//   - Robust to differences in score scales between retrievers
//   - Simple and effective in practice
//
// Example:
//
//	List A: [doc1, doc2, doc3]  →  doc1 gets 1/(60+0), doc2 gets 1/(60+1), doc3 gets 1/(60+2)
//	List B: [doc3, doc1, doc4]  →  doc3 gets 1/(60+0), doc1 gets 1/(60+1), doc4 gets 1/(60+2)
//	Fused:  doc1 and doc3 appear in both, so they get higher combined scores
func rrfFuseIDs(a, b []string, k int) []string {
	score := map[string]float64{}

	// Helper function to add reciprocal ranks from a single ranking list
	add := func(list []string) {
		for i, id := range list {
			// i is 0-indexed position, so rank is i
			score[id] += 1.0 / float64(k+i+1)
		}
	}

	// Add scores from both rankings
	add(a)
	add(b)

	// Collect all unique document IDs
	uniq := make([]string, 0, len(score))
	for id := range score {
		uniq = append(uniq, id)
	}

	// Sort by RRF score (highest first)
	sort.Slice(uniq, func(i, j int) bool { return score[uniq[i]] > score[uniq[j]] })

	return uniq
}
