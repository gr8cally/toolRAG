package main

import (
	"context"
	"encoding/json"
	"fmt"
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
	return `IMPORTANT: Use this tool FIRST for ANY question that could be answered by documents, forms, policies, procedures, or previous conversations.

This tool searches the internal knowledge base using hybrid search (semantic + keyword matching) to find the most relevant information.

When to use:
- Questions about forms, applications, or documents (e.g., "what boxes...", "what fields...", "what information...")
- Questions about policies, procedures, or guidelines
- Questions about specific requirements or instructions
- Any factual question that might be in stored documents
- When you're unsure - always check the knowledge base first!

Do NOT answer from general knowledge if this tool might have the answer. Always search first.

Input: A natural language search query (e.g., "boxes required on application form" or "DBS disclosure requirements")`
}

func (t InternalKnowledgeTool) Call(ctx context.Context, input string) (string, error) {
	return queryInternalKnowledge(ctx, input)
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
