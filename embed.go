package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// embeddings.go
// Go-only embeddings client for sentence-transformers/all-MiniLM-L6-v2 hosted online.
// Supports Hugging Face Inference API or a generic TEI (/embed) endpoint, chosen via env.
//
// Env:
//   # Option 1: Hugging Face Inference API (default if TEI not set)
//   HUGGINGFACE_API_TOKEN=hf_xxx
//   HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2
//   HUGGINGFACE_POOLING=mean           # mean|max (default: mean)
//   HUGGINGFACE_NORMALIZE=true         # true|false (default: true)
//   HUGGINGFACE_WAIT_FOR_MODEL=true    # default: true
//
//   # Option 2: TEI (Text Embeddings Inference) or compatible /embed service
//   TEI_URL=https://your-tei-hostname  # if set, TEI is used (POST {TEI_URL}/embed)
//   TEI_API_TOKEN=optional_bearer_token
//
//   # Batching
//   EMBEDDING_BATCH_SIZE=64            # default: 64

type Chunk struct {
	ID, Text string
}

// Embedder is a minimal interface you can call from your upload flow.
type Embedder interface {
	Embed(ctx context.Context, chunks []Chunk) (map[string][]float32, error)
}

func NewEmbedderFromEnv() (Embedder, error) {
	if currentConfig.HFAPIKey == "" {
		return nil, fmt.Errorf("missing HF_API_KEY in config")
	}
	model := currentConfig.EmbedModelName
	if model == "" {
		model = "sentence-transformers/all-MiniLM-L6-v2"
	}

	return &hfEmbedder{
		client:    &http.Client{Timeout: 60 * time.Second},
		token:     currentConfig.HFAPIKey,
		model:     model,
		pooling:   "mean",
		normalize: true,
		wait:      true,
		batch:     64,
	}, nil
}

// -------------------- Hugging Face Inference API --------------------

type hfEmbedder struct {
	client          *http.Client
	token, model    string
	pooling         string // mean|max
	normalize, wait bool
	batch           int
}

func (h *hfEmbedder) Embed(ctx context.Context, chunks []Chunk) (map[string][]float32, error) {
	if len(chunks) == 0 {
		return map[string][]float32{}, nil
	}
	out := make(map[string][]float32, len(chunks))

	// Endpoint (feature-extraction pipeline with pooling & normalization)
	// https://api-inference.huggingface.co/pipeline/feature-extraction/{model}?pooling=mean&normalize=true
	base := "https://router.huggingface.co/hf-inference/models/"
	url := fmt.Sprintf("%s%s/pipeline/feature-extraction", base, h.model)

	type reqBody struct {
		Inputs     []string               `json:"inputs"`
		Parameters map[string]interface{} `json:"parameters,omitempty"`
	}
	type resBody [][]float32

	params := map[string]interface{}{
		"pooling":   h.pooling,   // "mean" | "max"
		"normalize": h.normalize, // true | false
	}

	for i := 0; i < len(chunks); i += h.batch {
		j := i + h.batch
		if j > len(chunks) {
			j = len(chunks)
		}
		batch := chunks[i:j]

		inputs := make([]string, len(batch))
		for k, c := range batch {
			inputs[k] = c.Text
		}

		payload, _ := json.Marshal(reqBody{Inputs: inputs, Parameters: params})
		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+h.token)

		fmt.Println("embed request", time.Now())

		resp, err := h.client.Do(req)
		if err != nil {
			return nil, err
		}
		var rb resBody
		func() {
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				var dbg bytes.Buffer
				_, _ = dbg.ReadFrom(resp.Body)
				err = fmt.Errorf("HF API non-200: %d: %s", resp.StatusCode, dbg.String())
				return
			}
			err = json.NewDecoder(resp.Body).Decode(&rb)
		}()
		if err != nil {
			return nil, err
		}
		if len(rb) != len(batch) {
			return nil, fmt.Errorf("embeddings count mismatch")
		}

		for k, c := range batch {
			vec := make([]float32, len(rb[k]))
			copy(vec, rb[k])
			out[c.ID] = vec
		}
	}
	return out, nil
}

// -------------------- TEI (/embed) compatible client --------------------

type teiEmbedder struct {
	baseURL string
	token   string
	client  *http.Client
	batch   int
}

func (t *teiEmbedder) Embed(ctx context.Context, chunks []Chunk) (map[string][]float32, error) {
	if len(chunks) == 0 {
		return map[string][]float32{}, nil
	}
	out := make(map[string][]float32, len(chunks))

	type reqBody struct {
		Inputs []string `json:"inputs"`
	}
	type respBody struct {
		Embeddings [][]float32 `json:"embeddings"`
	}

	url := t.baseURL + "/embed"

	for i := 0; i < len(chunks); i += t.batch {
		if i > 1 {
			break
		}
		j := i + t.batch
		if j > len(chunks) {
			j = len(chunks)
		}
		batch := chunks[i:j]

		inputs := make([]string, len(batch))
		for k, c := range batch {
			inputs[k] = c.Text
		}
		payload, _ := json.Marshal(reqBody{Inputs: inputs})

		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		req.Header.Set("Content-Type", "application/json")
		if t.token != "" {
			req.Header.Set("Authorization", "Bearer "+t.token)
		}

		resp, err := t.client.Do(req)
		if err != nil {
			return nil, err
		}
		var rb respBody
		func() {
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				var dbg bytes.Buffer
				_, _ = dbg.ReadFrom(resp.Body)
				err = fmt.Errorf("TEI non-200: %d: %s", resp.StatusCode, dbg.String())
				return
			}
			err = json.NewDecoder(resp.Body).Decode(&rb)
		}()
		if err != nil {
			return nil, err
		}
		if len(rb.Embeddings) != len(batch) {
			return nil, fmt.Errorf("TEI embeddings count mismatch: have %d want %d", len(rb.Embeddings), len(batch))
		}
		for k, c := range batch {
			vec := make([]float32, len(rb.Embeddings[k]))
			copy(vec, rb.Embeddings[k])
			out[c.ID] = vec
		}
	}
	return out, nil
}
