package main

import "strings"

var retrievalStopWords = map[string]struct{}{
	"a": {}, "an": {}, "and": {}, "any": {}, "are": {}, "as": {}, "at": {},
	"be": {}, "by": {}, "do": {}, "does": {}, "for": {}, "from": {}, "how": {},
	"i": {}, "if": {}, "in": {}, "is": {}, "it": {}, "must": {}, "of": {},
	"on": {}, "or": {}, "should": {}, "that": {}, "the": {}, "their": {},
	"there": {}, "they": {}, "this": {}, "to": {}, "was": {}, "what": {},
	"when": {}, "where": {}, "which": {}, "who": {}, "why": {}, "with": {},
}

func retrievalLooksRelevant(query string, results []Retrieved) bool {
	queryTerms := informativeTokens(query)
	if len(queryTerms) == 0 {
		return false
	}

	for _, r := range results {
		docTerms := map[string]struct{}{}
		for _, tok := range tokenize(r.Text) {
			docTerms[tok] = struct{}{}
		}

		matches := 0
		for _, tok := range queryTerms {
			if _, ok := docTerms[tok]; ok {
				matches++
			}
		}

		if matches >= 2 {
			return true
		}
	}

	return false
}

func informativeTokens(text string) []string {
	raw := tokenize(text)
	out := make([]string, 0, len(raw))
	seen := make(map[string]struct{}, len(raw))

	for _, tok := range raw {
		if len(tok) < 3 {
			continue
		}
		if _, stop := retrievalStopWords[tok]; stop {
			continue
		}
		if _, ok := seen[tok]; ok {
			continue
		}
		seen[tok] = struct{}{}
		out = append(out, tok)
	}

	return out
}

func compactSnippet(text string) string {
	text = strings.Join(strings.Fields(text), " ")
	if len(text) <= 500 {
		return text
	}
	return text[:500] + "..."
}
