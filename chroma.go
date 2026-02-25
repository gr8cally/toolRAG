package main

import (
	"context"
	"fmt"

	chroma "github.com/amikos-tech/chroma-go/pkg/api/v2"
)

var chromaClient chroma.Client

func initChroma(baseURL string) error {
	c, err := chroma.NewHTTPClient(
		chroma.WithBaseURL(baseURL),
	)
	if err != nil {
		return fmt.Errorf("creating Chroma client: %w", err)
	}
	chromaClient = c
	return nil
}

var ragDocsCollection, conversationCollection chroma.Collection

func initChromaCollection(ctx context.Context) error {
	c, err := chromaClient.GetOrCreateCollection(ctx, "rag_docs")
	if err != nil {
		return fmt.Errorf("GetOrCreateCollection failed: %w", err)
	}
	ragDocsCollection = c

	c, err = chromaClient.GetOrCreateCollection(ctx, "conversation_memory")
	if err != nil {
		return fmt.Errorf("GetOrCreateCollection failed: %w", err)
	}
	conversationCollection = c
	return nil
}
