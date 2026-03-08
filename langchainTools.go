package main

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
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
	return "Return hotel options in a city with nightly USD prices. Input can be plain text like 'Nairobi' or JSON like {\"city\":\"Nairobi\"}."
}

func (t HotelScheduleTool) Call(ctx context.Context, input string) (string, error) {
	city, err := parseHotelToolInput(input)
	if err != nil {
		return "", err
	}
	result := getHotelSchedule(city)
	output, _ := json.Marshal(result)
	return string(output), nil
}

// Currency Converter Tool
type CurrencyConverterTool struct{}

func (t CurrencyConverterTool) Name() string {
	return "convert_currency"
}

func (t CurrencyConverterTool) Description() string {
	return "Convert currency amount from one currency to another. Input can be plain text like '100 USD to NGN' or JSON like {\"amount\":100,\"from\":\"USD\",\"to\":\"NGN\"}."
}

func (t CurrencyConverterTool) Call(ctx context.Context, input string) (string, error) {
	amount, from, to, err := parseCurrencyToolInput(input)
	if err != nil {
		return "", err
	}
	result := convertCurrency(amount, from, to)
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
	return "Return a flight schedule option from origin to destination with duration and USD price. Input can be plain text like 'from Lagos to Nairobi' or JSON like {\"origin\":\"Lagos\",\"destination\":\"Nairobi\"}."
}

func (t FlightScheduleTool) Call(ctx context.Context, input string) (string, error) {
	origin, destination, err := parseFlightToolInput(input)
	if err != nil {
		return "", err
	}
	result := getFlightSchedule(origin, destination)
	output, _ := json.Marshal(result)
	return string(output), nil
}

func parseHotelToolInput(input string) (string, error) {
	var params struct {
		City string `json:"city"`
	}
	if err := json.Unmarshal([]byte(input), &params); err == nil && strings.TrimSpace(params.City) != "" {
		return strings.TrimSpace(params.City), nil
	}

	city := strings.TrimSpace(strings.Trim(input, `"'`))
	if city == "" {
		return "", fmt.Errorf("invalid hotel input: %q", input)
	}
	return city, nil
}

func parseCurrencyToolInput(input string) (float64, string, string, error) {
	var params struct {
		Amount float64 `json:"amount"`
		From   string  `json:"from"`
		To     string  `json:"to"`
	}
	if err := json.Unmarshal([]byte(input), &params); err == nil && params.Amount > 0 && params.From != "" && params.To != "" {
		return params.Amount, strings.ToUpper(params.From), strings.ToUpper(params.To), nil
	}

	re := regexp.MustCompile(`(?i)(\d+(?:\.\d+)?)\s*([A-Z]{3})\s+(?:to|in)\s+([A-Z]{3})`)
	matches := re.FindStringSubmatch(input)
	if len(matches) != 4 {
		return 0, "", "", fmt.Errorf("invalid currency input: %q", input)
	}

	amount, err := strconv.ParseFloat(matches[1], 64)
	if err != nil {
		return 0, "", "", fmt.Errorf("invalid currency amount: %w", err)
	}

	return amount, strings.ToUpper(matches[2]), strings.ToUpper(matches[3]), nil
}

func parseFlightToolInput(input string) (string, string, error) {
	var params struct {
		Origin      string `json:"origin"`
		Destination string `json:"destination"`
	}
	if err := json.Unmarshal([]byte(input), &params); err == nil && params.Origin != "" && params.Destination != "" {
		return strings.TrimSpace(params.Origin), strings.TrimSpace(params.Destination), nil
	}

	re := regexp.MustCompile(`(?i)(?:from\s+)?([A-Za-z .'-]+?)\s+(?:to|->)\s+([A-Za-z .'-]+)\s*$`)
	matches := re.FindStringSubmatch(strings.TrimSpace(input))
	if len(matches) != 3 {
		return "", "", fmt.Errorf("invalid flight input: %q", input)
	}

	return strings.TrimSpace(matches[1]), strings.TrimSpace(matches[2]), nil
}
