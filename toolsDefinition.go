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
	if hfEmbedderConcrete == nil || ragDocsCollection == nil {
		return "NOT FOUND: internal knowledge base not initialized.", nil
	}

	// Hybrid retrieve from the indexed internal document collection.
	docResults, err := hybridRetrieve(ctx, ragDocsCollection, query, 4)
	if err != nil {
		return "", err
	}

	if len(docResults) == 0 {
		return "NOT FOUND: no relevant information found in internal knowledge base.", nil
	}

	var out []string
	out = append(out, "FOUND: relevant internal knowledge")
	for i, r := range docResults {
		out = append(out, fmt.Sprintf("Source %d: %s", i+1, r.Source))
		out = append(out, fmt.Sprintf("Snippet %d: %s", i+1, compactSnippet(r.Text)))
	}

	return strings.Join(out, "\n\n"), nil
}
