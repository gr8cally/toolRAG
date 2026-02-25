# toolRAG

A LangChain-based AI agent with RAG (Retrieval Augmented Generation) capabilities, built in Go.

## Features

- **4 Built-in Tools**:
  1. `get_flight_schedule` - Get flight schedules between cities
  2. `get_hotel_schedule` - Get hotel options in a city
  3. `convert_currency` - Convert currency between different types
  4. `query_internal_knowledge` - Query internal knowledge base with RAG

- **RAG System**: Automatically loads and indexes documents from the `data/` directory
- **Conversation History**: Stores conversations in the vector store for future retrieval
- **LangChain Integration**: Uses LangChain Go for agent orchestration
- **OpenRouter Support**: Works with OpenRouter API for LLM access

## Prerequisites

- Go 1.25.1 or higher
- OpenRouter API key ([Get one here](https://openrouter.ai/))
- (Optional) HuggingFace API key for embeddings

## Installation

1. Clone the repository
2. Install dependencies:
```bash
go mod tidy
```

3. Create a `.env` file based on `.env-example`:
```bash
cp .env-example .env
```

4. Edit `.env` and add your API keys:
```
OPENROUTER_API_KEY=your_openrouter_key_here
HF_API_KEY=your_huggingface_key_here
```

## Usage

Run the agent with a prompt:

```bash
go run main.go "What is the weather in Jos?"
```

Or build and run:

```bash
go build -o toolrag main.go
./toolrag "Find me a flight from Lagos to Nairobi and a hotel there"
```

## Adding Documents to RAG

Place any text files in the `data/` directory and they will be automatically loaded and indexed when the application starts.

```bash
mkdir -p data
echo "Our company policy: All meetings start at 9 AM" > data/company_policy.txt
```

Then query it:

```bash
./toolrag "What time do meetings start?"
```

## Environment Variables

See `.env-example` for all available environment variables:

- `OPENROUTER_API_KEY` (required) - Your OpenRouter API key
- `HF_API_KEY` (optional) - Your HuggingFace API key for embeddings
- `OPENROUTER_MODEL` (optional) - Model to use (default: anthropic/claude-3.5-sonnet)
- `EMBEDDING_MODEL` (optional) - Embedding model (default: sentence-transformers/all-MiniLM-L6-v2)
- `VECTOR_STORE_PATH` (optional) - Path to store vectors (default: ./vectorstore)

## Output

The application prints:
1. Conversation history
2. Final response from the agent

Example:
```
=== Conversation History ===
User: Find me a flight from Lagos to Nairobi
Assistant: I found a flight from Lagos to Nairobi...

=== Final Response ===
I found a flight from Lagos to Nairobi...
```
