# Handoff Packet — toolRAG (Go)
*(Share this file with the next LLM / developer.)*

## Goal
Build a Go LangChain agent that supports tool-enhanced RAG with:

- HuggingFace embeddings: `sentence-transformers/all-MiniLM-L6-v2` using `HF_API_KEY`
- OpenRouter LLM: `nvidia/nemotron-3-nano-30b-a3b:free` using `OPENROUTER_API_KEY`
- ChromaDB vector store (external service)
- Hybrid retrieval
- Long-term memory retention (conversation history stored in vector store)
- **No HTTP endpoints**; run as `go run main.go "prompt"` and print conversation history + final response to stdout
- Load + index all files in `./data` at startup
- Conversation history saved (**not tool-call history**)

---

## What is implemented already

### 1) CLI agent execution + stdout printing
- CLI arg prompt read and agent run is in [`main()`](main.go:275).
- Prints “=== Conversation History ===” then prints retrieved prior history + current run messages and then prints “=== Final Response ===”.

### 2) LLM setup (OpenRouter)
- OpenRouter OpenAI-compatible client in [`main()`](main.go:315) using base URL `https://openrouter.ai/api/v1`.
- Default model is `nvidia/nemotron-3-nano-30b-a3b:free` via [`loadConfigFromEnv()`](main.go:92).

### 3) Tools
- LangChain tool implementations (Name/Description/Call) are in:
  - [`FlightScheduleTool`](langchainTools.go:82)
  - [`HotelScheduleTool`](langchainTools.go:14)
  - [`CurrencyConverterTool`](langchainTools.go:39)
  - 4th tool: [`InternalKnowledgeTool`](langchainTools.go:65) → [`queryInternalKnowledge()`](main.go:224)

### 4) ChromaDB vector store
- Chroma HTTP client and collections:
  - [`initChroma()`](chroma.go:12)
  - [`initChromaCollection()`](chroma.go:25)
- Helpers for upsert/query/get:
  - [`chromaUpsert()`](chroma_helpers.go:11)
  - [`chromaQuery()`](chroma_helpers.go:34)
  - [`chromaGetByIDs()`](chroma_helpers.go:63)
- Two collections used:
  - `rag_docs`
  - `conversation_memory`

### 5) HF embeddings (MiniLM)
- HF Inference API embedder:
  - [`NewEmbedderFromEnv()`](embed.go:40)
- Uses `HF_API_KEY` and `EMBEDDING_MODEL` defaulting to `sentence-transformers/all-MiniLM-L6-v2`.

### 6) Data/ ingestion + indexing into Chroma
- Walks `RAG_DATA_DIR` (default `./data`), chunks text, embeds chunks, and upserts into `rag_docs` in [`loadDocumentsFromDataDir()`](main.go:103).
- Simple chunking is [`chunkText()`](main.go:38).

### 7) Hybrid retrieval
- BM25 implemented in [`BM25Index`](retrieval.go:18).
- Vector retrieval via Chroma + embeddings in [`vectorRetrieve()`](retrieval.go:117).
- Hybrid retrieval with RRF fusion in [`hybridRetrieve()`](retrieval.go:153).
- Internal RAG tool uses hybrid for documents and vector-only for conversation:
  - [`queryInternalKnowledge()`](main.go:224).

### 8) Conversation memory persistence
- Stores conversation turns into `conversation_memory` with embeddings in [`storeConversationHistory()`](main.go:201).
- Loads remembered history via [`loadRecentConversationHistory()`](chroma_helpers.go:85) and prints it in [`main()`](main.go:330).

### 9) Env/example + docs updated
- Updated env sample in [`.env-example`](.env-example:7).
- README reflects external Chroma + HF required in [`README.md`](README.md:18).
- Chroma-go dependency promoted to direct in [`go.mod`](go.mod:5).

---

## What remains / known gaps

### A) “Full conversation history” (strict interpretation gap)
- Current history loading is **semantic query-based**, not “all turns ever”.
- [`loadRecentConversationHistory()`](chroma_helpers.go:85) calls [`vectorRetrieve()`](retrieval.go:117) with query `"conversation history"`.
- This returns **some** remembered turns, not guaranteed complete chronological history.

### B) Conversation ordering
- To stabilize output, retrieval currently sorts alphabetically (stable but **not chronological**) via `sort.Strings(out)` in [`loadRecentConversationHistory()`](chroma_helpers.go:85).
- Stored metadata includes `"timestamp"` in [`storeConversationHistory()`](main.go:201), but retrieval does not sort by it.

### C) Build/verify status
- Go toolchain wasn’t available in the environment used during implementation, so compilation was not verified there.
- Run locally:
  - `go test ./...`
  - `go vet ./...`
  - `go mod tidy`
and fix compile/runtime issues (if any).

### D) Hybrid retrieval scope (optional)
- Hybrid retrieval is implemented for `rag_docs` (BM25 + vector + fusion).
- Conversation retrieval is vector-only in [`queryInternalKnowledge()`](main.go:224).
- Requirement only explicitly demands hybrid retrieval overall; if you want hybrid for conversation memory too, implement BM25 index for `conversation_memory` as well.

---

## Recommended next steps (implementation plan)

### 1) Deterministic conversation history listing (all turns, chronological)
Implement “full conversation history” as *every stored turn, ordered by timestamp*:

- Store each turn with sortable ID:
  - e.g. `conv|{RFC3339Nano}|{sha256}` using [`stableID()`](main.go:33) or similar.
- Fetch all turns from Chroma:
  - Option A (recommended): maintain an index doc listing IDs in order.
  - Option B: paginate `Get` or query repeatedly (depends on Chroma API capabilities).
- Sort by the `"timestamp"` metadata (RFC3339) rather than alphabetically.
- Print in chronological order before appending the current run.

### 2) Replace alphabetical sort
Remove `sort.Strings(out)` in [`loadRecentConversationHistory()`](chroma_helpers.go:85) and replace with timestamp sort using metadata.

### 3) (Optional) hybrid memory retrieval
If desired:
- Build BM25 corpus for conversations too (similar to [`bm25Index`](main.go:66)) and fuse with vector results.

---

## Acceptance criteria
- `go run main.go "prompt"` prints:
  1) conversation history (clarify whether “last N” vs “all turns ever” is required)
  2) final agent response
- Uses HF embeddings + OpenRouter model + Chroma collections.
- Tools available: 3 domain tools + internal RAG query tool.
- Loads and indexes `./data` files on startup.

---

## Files to read first
- [`main.go`](main.go:1)
- [`langchainTools.go`](langchainTools.go:1)
- [`retrieval.go`](retrieval.go:1)
- [`chroma_helpers.go`](chroma_helpers.go:1)
- [`embed.go`](embed.go:1)
- [`chroma.go`](chroma.go:1)

---

## Suggested models for the next LLM
Pick a model strong at Go + multi-file refactors:
- Claude 3.5 Sonnet / Claude 3.5 Haiku
- GPT-4.1 / GPT-4o
- DeepSeek R1 / V3

When handing off, specify whether “full conversation history” means “all turns ever, chronological” or “last N remembered turns is acceptable”.