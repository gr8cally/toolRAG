package main

import (
	"context"
	"encoding/json"
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

// ------------------
// Tool Logic (Original 3 tools)
// ------------------

func getFlightSchedule(origin, destination string) map[string]interface{} {
	return map[string]interface{}{
		"origin":            origin,
		"destination":       destination,
		"flight_time_hours": 5.5,
		"price_usd":         920,
	}
}

func getHotelSchedule(city string) map[string]interface{} {
	return map[string]interface{}{
		"city": city,
		"hotels": []map[string]interface{}{
			{
				"name":      "Nairobi Serena",
				"price_usd": 250,
			},
			{
				"name":      "Radisson Blu",
				"price_usd": 200,
			},
		},
	}
}

func convertCurrency(amount float64, from, to string) map[string]interface{} {
	rate := 925.0
	return map[string]interface{}{
		"amount_converted": amount * rate,
		"currency":         to,
	}
}

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
// LangChain Tools
// ------------------

// Flight Schedule Tool
type FlightScheduleTool struct{}

func (t FlightScheduleTool) Name() string {
	return "get_flight_schedule"
}

func (t FlightScheduleTool) Description() string {
	return "Return a flight schedule option from origin to destination with duration and USD price. Input should be a JSON object with 'origin' and 'destination' fields."
}

func (t FlightScheduleTool) Call(ctx context.Context, input string) (string, error) {
	var params struct {
		Origin      string `json:"origin"`
		Destination string `json:"destination"`
	}

	if err := json.Unmarshal([]byte(input), &params); err != nil {
		return "", fmt.Errorf("invalid input format: %w", err)
	}

	result := getFlightSchedule(params.Origin, params.Destination)
	output, _ := json.Marshal(result)
	return string(output), nil
}

// Hotel Schedule Tool
type HotelScheduleTool struct{}

func (t HotelScheduleTool) Name() string {
	return "get_hotel_schedule"
}

func (t HotelScheduleTool) Description() string {
	return "Return hotel options in a city with nightly USD prices. Input should be a JSON object with 'city' field."
}

func (t HotelScheduleTool) Call(ctx context.Context, input string) (string, error) {
	var params struct {
		City string `json:"city"`
	}

	if err := json.Unmarshal([]byte(input), &params); err != nil {
		return "", fmt.Errorf("invalid input format: %w", err)
	}

	result := getHotelSchedule(params.City)
	output, _ := json.Marshal(result)
	return string(output), nil
}

// Currency Converter Tool
type CurrencyConverterTool struct{}

func (t CurrencyConverterTool) Name() string {
	return "convert_currency"
}

func (t CurrencyConverterTool) Description() string {
	return "Convert currency amount from one currency to another. Input should be a JSON object with 'amount', 'from', and 'to' fields."
}

func (t CurrencyConverterTool) Call(ctx context.Context, input string) (string, error) {
	var params struct {
		Amount float64 `json:"amount"`
		From   string  `json:"from"`
		To     string  `json:"to"`
	}

	if err := json.Unmarshal([]byte(input), &params); err != nil {
		return "", fmt.Errorf("invalid input format: %w", err)
	}

	result := convertCurrency(params.Amount, params.From, params.To)
	output, _ := json.Marshal(result)
	return string(output), nil
}

// Internal Knowledge Tool (RAG)
type InternalKnowledgeTool struct {
	ctx context.Context
}

func (t InternalKnowledgeTool) Name() string {
	return "query_internal_knowledge"
}

func (t InternalKnowledgeTool) Description() string {
	return "Query the internal knowledge base for information from documents and previous conversations. Input should be a search query string."
}

func (t InternalKnowledgeTool) Call(ctx context.Context, input string) (string, error) {
	return queryInternalKnowledge(ctx, input)
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
	modelName := getEnvWithDefault("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
	var err error
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
