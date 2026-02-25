package main

import (
	"context"
	"fmt"
	"io/fs"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/tools"
)

// Simple in-memory vector store
type Document struct {
	Content   string
	Embedding []float32
	Metadata  map[string]interface{}
}

type InMemoryVectorStore struct {
	documents []Document
	embedder  embeddings.Embedder
}

func NewInMemoryVectorStore(embedder embeddings.Embedder) *InMemoryVectorStore {
	return &InMemoryVectorStore{
		documents: make([]Document, 0),
		embedder:  embedder,
	}
}

func (vs *InMemoryVectorStore) AddDocument(ctx context.Context, content string, metadata map[string]interface{}) error {
	// Generate embedding
	embeddings, err := vs.embedder.EmbedQuery(ctx, content)
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	vs.documents = append(vs.documents, Document{
		Content:   content,
		Embedding: embeddings,
		Metadata:  metadata,
	})

	return nil
}

func (vs *InMemoryVectorStore) SimilaritySearch(ctx context.Context, query string, k int) ([]Document, error) {
	if len(vs.documents) == 0 {
		return []Document{}, nil
	}

	// Generate query embedding
	queryEmbedding, err := vs.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}

	// Calculate cosine similarity for each document
	type scoredDoc struct {
		doc   Document
		score float32
	}
	scores := make([]scoredDoc, 0, len(vs.documents))

	for _, doc := range vs.documents {
		similarity := cosineSimilarity(queryEmbedding, doc.Embedding)
		scores = append(scores, scoredDoc{doc: doc, score: similarity})
	}

	// Sort by score (descending)
	for i := 0; i < len(scores)-1; i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score > scores[i].score {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}

	// Return top k results
	if k > len(scores) {
		k = len(scores)
	}

	results := make([]Document, k)
	for i := 0; i < k; i++ {
		results[i] = scores[i].doc
	}

	return results, nil
}

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// Global variables
var (
	vectorStore     *InMemoryVectorStore
	conversationLog []string
	llmClient       llms.Model
)

type Config struct {
	HFAPIKey       string // HF_API_KEY (required)
	EmbedModelName string // EMBED_MODEL_NAME
	GeminiAPIKey   string // GEMINI_API_KEY
	LLMModelName   string // LLM_MODEL_NAME
	ChromaDBHost   string // CHROMA_DB_HOST
	RAGDataDir     string // RAG_DATA_DIR
	ChunkLength    int    // CHUNK_LENGTH
	Port           int    // PORT
}

var currentConfig Config

// ------------------
// RAG Functions
// ------------------

// Initialize in-memory vector store
func initVectorStore(ctx context.Context, embedder embeddings.Embedder) error {
	vectorStore = NewInMemoryVectorStore(embedder)
	log.Println("Initialized in-memory vector store")
	return nil
}

// Load documents from data/ directory
func loadDocumentsFromDataDir(ctx context.Context) error {
	dataDir := "./data"

	// Check if data directory exists
	if _, err := os.Stat(dataDir); os.IsNotExist(err) {
		log.Printf("Data directory does not exist, creating: %s", dataDir)
		if err := os.MkdirAll(dataDir, 0755); err != nil {
			return fmt.Errorf("failed to create data directory: %w", err)
		}
		return nil
	}

	// If vector store is not available, skip loading
	if vectorStore == nil {
		log.Println("Vector store not available, skipping document loading")
		return nil
	}

	// Walk through data directory
	documentCount := 0
	err := filepath.WalkDir(dataDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Skip directories
		if d.IsDir() {
			return nil
		}

		// Read file content
		content, err := os.ReadFile(path)
		if err != nil {
			log.Printf("Failed to read file %s: %v", path, err)
			return nil
		}

		// Add document to vector store
		err = vectorStore.AddDocument(ctx, string(content), map[string]interface{}{
			"source": path,
			"type":   "document",
		})
		if err != nil {
			log.Printf("Warning: Failed to add document %s: %v", path, err)
		} else {
			documentCount++
		}

		return nil
	})

	if err != nil {
		return fmt.Errorf("failed to walk data directory: %w", err)
	}

	if documentCount > 0 {
		log.Printf("Loaded %d documents from data directory", documentCount)
	}

	return nil
}

// Store conversation in vector store
func storeConversationHistory(ctx context.Context, userMsg, assistantMsg string) {
	if vectorStore == nil {
		return
	}

	conversation := fmt.Sprintf("User: %s\nAssistant: %s", userMsg, assistantMsg)
	err := vectorStore.AddDocument(ctx, conversation, map[string]interface{}{
		"type":      "conversation",
		"timestamp": time.Now().Format(time.RFC3339),
	})
	if err != nil {
		log.Printf("Warning: Failed to store conversation: %v", err)
	}
}

// Query RAG for internal knowledge
func queryInternalKnowledge(ctx context.Context, query string) (string, error) {
	if vectorStore == nil {
		return "Vector store not available", nil
	}

	// Search for relevant documents
	docs, err := vectorStore.SimilaritySearch(ctx, query, 3)
	if err != nil {
		return "", fmt.Errorf("failed to search vector store: %w", err)
	}

	if len(docs) == 0 {
		return "No relevant information found in internal knowledge base.", nil
	}

	// Format results
	var results []string
	for i, doc := range docs {
		source := "unknown"
		if s, ok := doc.Metadata["source"]; ok {
			source = fmt.Sprintf("%v", s)
		}
		results = append(results, fmt.Sprintf("Result %d (source: %s):\n%s", i+1, source, doc.Content))
	}

	return strings.Join(results, "\n\n"), nil
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
	// Load environment variables
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found, using environment variables")
	}

	// Get command line argument
	if len(os.Args) < 2 {
		log.Fatal("Usage: go run main.go \"<your prompt here>\"")
	}
	userPrompt := os.Args[1]

	// init chroma client
	err := initChroma("currentConfig.ChromaDBHost")
	if err != nil {
		log.Fatalf("failed to init chroma: %v", err)
		return
	}
	defer func() {
		if err := chromaClient.Close(); err != nil {
			log.Printf("Error closing Chroma client: %v", err)
		}
	}()

	// Get API keys
	openRouterKey := os.Getenv("OPENROUTER_API_KEY")
	if openRouterKey == "" {
		log.Fatal("OPENROUTER_API_KEY not set in environment")
	}

	hfAPIKey := os.Getenv("HF_API_KEY")
	if hfAPIKey == "" {
		log.Println("Warning: HF_API_KEY not set, using OpenAI embeddings")
	}

	ctx := context.Background()

	// Initialize LLM (OpenRouter with OpenAI-compatible API)
	modelName := getEnvWithDefault("OPENROUTER_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")
	llmClient, err = openai.New(
		openai.WithToken(openRouterKey),
		openai.WithModel(modelName),
		openai.WithBaseURL("https://openrouter.ai/api/v1"),
	)
	if err != nil {
		log.Fatalf("Failed to initialize LLM: %v", err)
	}

	// Initialize embeddings with OpenAI directly (for embeddings only)
	// Note: OpenRouter may not support embeddings, so we use OpenAI directly with the HF key as fallback
	// sentence-transformers/all-MiniLM-L6-v2
	var embedder embeddings.Embedder
	if hfAPIKey != "" {
		// Try to use OpenAI embeddings with a separate key
		embeddingLLM, err := openai.New(
			openai.WithToken(openRouterKey),
			openai.WithModel("text-embedding-ada-002"),
		)
		if err == nil {
			embedder, err = embeddings.NewEmbedder(embeddingLLM)
			if err != nil {
				log.Printf("Warning: Failed to initialize embedder: %v. Vector store will have limited functionality.", err)
			}
		}
	}

	// Initialize vector store
	if embedder != nil {
		if err := initVectorStore(ctx, embedder); err != nil {
			log.Printf("Warning: Vector store initialization failed: %v", err)
		}

		// Load documents from data directory
		if err := loadDocumentsFromDataDir(ctx); err != nil {
			log.Printf("Warning: Failed to load documents: %v", err)
		}
	} else {
		log.Println("Warning: No embedder available. RAG functionality will be limited.")
	}

	// Initialize tools
	agentTools := []tools.Tool{
		FlightScheduleTool{},
		HotelScheduleTool{},
		CurrencyConverterTool{},
		InternalKnowledgeTool{ctx: ctx},
	}

	// Create agent executor
	executor, err := agents.Initialize(
		llmClient,
		agentTools,
		agents.ZeroShotReactDescription,
		agents.WithMaxIterations(5),
	)
	if err != nil {
		log.Fatalf("Failed to initialize agent: %v", err)
	}

	// Log user prompt
	conversationLog = append(conversationLog, fmt.Sprintf("User: %s", userPrompt))

	// Execute agent
	fmt.Println("=== Conversation History ===")
	response, err := chains.Run(ctx, executor, userPrompt)
	if err != nil {
		log.Fatalf("Agent execution failed: %v", err)
	}

	// Log assistant response
	conversationLog = append(conversationLog, fmt.Sprintf("Assistant: %s", response))

	// Store conversation in vector store
	storeConversationHistory(ctx, userPrompt, response)

	// Print conversation history
	for _, entry := range conversationLog {
		fmt.Println(entry)
	}

	fmt.Println("\n=== Final Response ===")
	fmt.Println(response)
}
