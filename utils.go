package main

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

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
