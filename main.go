package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/joho/godotenv"
	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/tools"
)

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
