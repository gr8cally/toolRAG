package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// ------------------
// LangChain Tools
// ------------------

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

func queryInternalKnowledge(ctx context.Context, query string) (string, error) {
	if hfEmbedderConcrete == nil || ragDocsCollection == nil || conversationCollection == nil {
		return "Internal knowledge base not initialized.", nil
	}

	// Hybrid retrieve from rag_docs and also vector-retrieve from conversation memory.
	docResults, err := hybridRetrieve(ctx, ragDocsCollection, query, 4)
	if err != nil {
		return "", err
	}
	memResults, err := vectorRetrieve(ctx, conversationCollection, query, 3)
	if err != nil {
		return "", err
	}

	if len(docResults) == 0 && len(memResults) == 0 {
		return "No relevant information found in internal knowledge base.", nil
	}

	var out []string
	if len(docResults) > 0 {
		out = append(out, "=== Relevant Documents (hybrid) ===")
		for i, r := range docResults {
			out = append(out, fmt.Sprintf("Doc %d (source: %s):\n%s", i+1, r.Source, r.Text))
		}
	}
	if len(memResults) > 0 {
		out = append(out, "=== Relevant Past Conversations (vector) ===")
		for i, r := range memResults {
			out = append(out, fmt.Sprintf("Memory %d:\n%s", i+1, r.Text))
		}
	}

	return strings.Join(out, "\n\n"), nil
}

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
