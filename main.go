package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	chroma "github.com/amikos-tech/chroma-go/pkg/api/v2"
	"github.com/joho/godotenv"
	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/tools"
)

type RagChunk struct {
	ID       string
	Text     string
	Source   string
	Kind     string // "document" or "conversation"
	TimeRFC9 string
}

func stableID(parts ...string) string {
	h := sha256.Sum256([]byte(strings.Join(parts, "|")))
	return hex.EncodeToString(h[:])
}

func chunkText(text string, chunkSize int) []string {
	if chunkSize <= 0 {
		chunkSize = 800
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	// Very simple chunker: split on paragraphs, then hard-wrap.
	var chunks []string
	paras := strings.Split(text, "\n\n")
	for _, p := range paras {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		for len(p) > chunkSize {
			chunks = append(chunks, strings.TrimSpace(p[:chunkSize]))
			p = strings.TrimSpace(p[chunkSize:])
		}
		if p != "" {
			chunks = append(chunks, p)
		}
	}
	return chunks
}

var (
	conversationLog []string
	llmClient       llms.Model

	hfEmbedderConcrete Embedder

	// BM25 corpus cache for rag_docs (hybrid retrieval)
	bm25Index *BM25Index
)

type Config struct {
	OpenRouterAPIKey string // OPENROUTER_API_KEY (required)
	HFAPIKey         string // HF_API_KEY (required)
	OpenRouterModel  string // OPENROUTER_MODEL (default: required model)
	EmbedModelName   string // EMBEDDING_MODEL (default: sentence-transformers/all-MiniLM-L6-v2)
	ChromaDBHost     string // CHROMA_DB_HOST (default: http://localhost:8000)
	RAGDataDir       string // RAG_DATA_DIR (default: ./data)
	ChunkLength      int    // CHUNK_LENGTH (default: 800)
}

var currentConfig Config

// ------------------
// RAG Functions
// ------------------

func loadConfigFromEnv() Config {
	chunkLen := 800
	if v := os.Getenv("CHUNK_LENGTH"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			chunkLen = n
		}
	}

	return Config{
		OpenRouterAPIKey: os.Getenv("OPENROUTER_API_KEY"),
		HFAPIKey:         os.Getenv("HF_API_KEY"),
		OpenRouterModel:  getEnvWithDefault("OPENROUTER_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free"),
		EmbedModelName:   getEnvWithDefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
		ChromaDBHost:     getEnvWithDefault("CHROMA_DB_HOST", "http://localhost:8000"),
		RAGDataDir:       getEnvWithDefault("RAG_DATA_DIR", "./data"),
		ChunkLength:      chunkLen,
	}
}

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

// ------------------
// Utility Functions
// ------------------

func getEnvWithDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// ------------------
// Main Application
// ------------------

func main() {
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found, using environment variables")
	}
	currentConfig = loadConfigFromEnv()

	if currentConfig.OpenRouterAPIKey == "" {
		log.Fatal("OPENROUTER_API_KEY not set in environment")
	}
	if currentConfig.HFAPIKey == "" {
		log.Fatal("HF_API_KEY not set in environment (required for embeddings)")
	}

	if len(os.Args) < 2 {
		log.Fatal("Usage: go run main.go \"<your prompt here>\"")
	}
	userPrompt := os.Args[1]

	ctx := context.Background()

	// Init Chroma (external service)
	if err := initChroma(currentConfig.ChromaDBHost); err != nil {
		log.Fatalf("failed to init chroma: %v", err)
	}
	defer func() {
		if err := chromaClient.Close(); err != nil {
			log.Printf("Error closing Chroma client: %v", err)
		}
	}()
	if err := initChromaCollection(ctx); err != nil {
		log.Fatalf("failed to init chroma collections: %v", err)
	}

	// Init HF embedder
	var err error
	hfEmbedderConcrete, err = NewEmbedderFromEnv()
	if err != nil {
		log.Fatalf("failed to init HF embedder: %v", err)
	}

	// Initialize LLM (OpenRouter with OpenAI-compatible API)
	llmClient, err = openai.New(
		openai.WithToken(currentConfig.OpenRouterAPIKey),
		openai.WithModel(currentConfig.OpenRouterModel),
		openai.WithBaseURL("https://openrouter.ai/api/v1"),
	)
	if err != nil {
		log.Fatalf("Failed to initialize LLM: %v", err)
	}

	// Index data/ documents (chunks) into rag_docs
	if err := loadDocumentsFromDataDir(ctx); err != nil {
		log.Printf("Warning: Failed to load documents: %v", err)
	}

	// Load prior conversation history
	prior, err := loadRecentConversationHistory(ctx, 20)
	if err != nil {
		log.Printf("Warning: Failed to load conversation history: %v", err)
	}

	// Also store prior in in-process log so the full output includes previous runs + this run.
	conversationLog = append(conversationLog, prior...)

	// Initialize tools
	agentTools := []tools.Tool{
		FlightScheduleTool{},
		HotelScheduleTool{},
		CurrencyConverterTool{},
		InternalKnowledgeTool{ctx: ctx},
	}

	executor, err := agents.Initialize(
		llmClient,
		agentTools,
		agents.ZeroShotReactDescription,
		agents.WithMaxIterations(5),
	)
	if err != nil {
		log.Fatalf("Failed to initialize agent: %v", err)
	}

	// Append current turn to the printed conversation log (conversation-only, not tool traces)
	conversationLog = append(conversationLog, fmt.Sprintf("User: %s", userPrompt))

	response, err := chains.Run(ctx, executor, userPrompt)
	if err != nil {
		log.Fatalf("Agent execution failed: %v", err)
	}

	conversationLog = append(conversationLog, fmt.Sprintf("Assistant: %s", response))
	fmt.Println("=== Conversation Log (including prior) ===")
	for _, entry := range conversationLog {
		fmt.Println(entry)
	}

	// Persist conversation turn to Chroma
	storeConversationHistory(ctx, userPrompt, response)

	fmt.Println("\n=== Final Response ===")
	fmt.Println(response)
}
