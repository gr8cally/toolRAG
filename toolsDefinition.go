package main

import (
	"context"
	"fmt"
	"strings"
)

// ------------------
// Tool Logic
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

func queryInternalKnowledge(ctx context.Context, query string) (string, error) {
	if hfEmbedderConcrete == nil || ragDocsCollection == nil || conversationCollection == nil {
		return "Internal knowledge base not initialized.", nil
	}

	// Hybrid retrieve from rag_docs and also vector-retrieve from conversation memory.
	docResults, err := hybridRetrieve(ctx, ragDocsCollection, query, 4)
	if err != nil {
		return "", err
	}

	if len(docResults) == 0 {
		return "No relevant information found in internal knowledge base.", nil
	}

	var out []string
	if len(docResults) > 0 {
		out = append(out, "=== Relevant Documents (hybrid) ===")
		for i, r := range docResults {
			out = append(out, fmt.Sprintf("Doc %d (source: %s):\n%s", i+1, r.Source, r.Text))
		}
	}

	return strings.Join(out, "\n\n"), nil
}
