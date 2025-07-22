# RAG LLM BASED FOR INSTAGRAM CHAT (🛜)

This project is a Retrieval-Augmented Generation (RAG) system that combines Instagram personal data with live web information using LLMs (Large Language Models). It supports local vector search (FAISS), FastAPI for serving, and secure API key management.

## Features

- Ingest and process Instagram export data (JSON, images, SRT, etc.)
- Build and query a FAISS vector database locally
- Rerank results using embeddings or GPT
- Combine Instagram context and live web search
- Interactive CLI and FastAPI server
- Secure environment variable management

## Folder Structure

```
├── Dockerfile
├── requirements.txt
├── .env.example
├── rag.py
├── ingest.py
├── reels.py
├── ... (other scripts)
├── data/
│   └── instagram-days... (Instagram export folders)
├── storage/
│   └── faiss/ (vector DB and docs)
```

## Getting Started

### 1. Clone the repository

```sh
git clone <repo-url>
cd "RAG LLM BASED FOR INSTAGRAM CHAT (🛜)"
```

### 2. Create and edit your `.env` file

Copy `.env.example` to `.env` and fill in your API keys:

```sh
cp .env.example .env
```

### 3. Run locally (optional)

```sh
pip install -r requirements.txt
python rag.py
```

## .env Example

See `.env.example` for required environment variables.

## Security Notes

- **Do not commit your real API keys to source control.**
- Use `.env` for secrets and pass them securely to Docker.

## License

MIT

---

# .env.example

# Copy this file to .env and fill in your secrets

OPENAI_API_KEY=your_openai_api_key_here

# Add other secrets as needed
