# Flipkart Multimodal RAG System

A production-quality **Multimodal Retrieval-Augmented Generation (RAG)** system for retail, built on the Flipkart e-commerce dataset.

---

## LLM Provider Architecture

| Provider | Model | Task |
|---|---|---|
| **Ollama (local)** | `llama3` | Main RAG Q&A — product recommendations, price queries, conversational answers |
| **HuggingFace (local)** | `google/flan-t5-base` | Structured feature extraction, product summaries, spec explanations |
| **Groq (cloud)** | `llama3-8b-8192` | Multi-product comparison tables, complex analytical queries (ultra-fast) |

---

## Features

- **Text RAG** — "Best phones under 20000", "Cheap running shoes", "Laptops for coding"
- **Image RAG** — Upload a product photo → find visually similar products via CLIP
- **Hybrid RAG** — "Find a cheaper version of this product" + image upload
- **Product Comparison** — Select multiple products → Groq generates a comparison table
- **Feature Extraction** — HuggingFace flan-t5 explains product specs in plain English
- **Top Rated** — Browse highest-rated products by category
- **Filters** — Price range, brand, category, minimum rating

---

## Project Structure

```
retail-multimodal-rag/
├── data/                          # Place flipkart CSV here
├── images/                        # Downloaded product images (auto-populated)
├── indexes/                       # FAISS indexes (auto-generated)
│   ├── text_index.faiss
│   ├── metadata.pkl
│   ├── image_index.faiss
│   └── image_metadata.pkl
├── logs/
├── backend/
│   ├── data_loader.py             # CSV preprocessing pipeline
│   ├── text_embeddings.py         # SentenceTransformers + FAISS text index
│   ├── image_embeddings.py        # CLIP + FAISS image index
│   ├── vector_store.py            # FAISS index manager
│   ├── retriever.py               # Unified search interface
│   ├── llm_providers.py           # Ollama / HuggingFace / Groq providers
│   ├── rag_pipeline.py            # End-to-end RAG orchestration
│   ├── build_indexes.py           # One-time index builder script
│   └── api.py                     # FastAPI server
├── frontend/
│   └── streamlit_app.py           # Streamlit UI
├── .env.example                   # Environment variable template
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Start Ollama with llama3

```bash
ollama pull llama3
ollama serve
```

### 4. Place the dataset

```bash
# Copy the Flipkart CSV into the data folder
cp /path/to/flipkart_com-ecommerce_sample.csv data/
```

### 5. Build indexes

```bash
# Full build (text + image, ~2000 images downloaded)
python backend/build_indexes.py

# Text-only (faster, no image download needed)
python backend/build_indexes.py --no-images

# Development mode (limit rows for speed)
python backend/build_indexes.py --max-rows 5000 --max-imgs 500
```

### 6. Start the API server

```bash
cd retail-multimodal-rag
python backend/api.py
# API docs: http://localhost:8000/docs
```

### 7. Start the Streamlit frontend

```bash
streamlit run frontend/streamlit_app.py
# Opens at: http://localhost:8501
```

---

## API Endpoints

| Method | Endpoint | Provider | Description |
|---|---|---|---|
| GET | `/health` | — | Health check + provider status |
| POST | `/search_text` | — | Fast text search (no LLM) |
| POST | `/search_image` | — | Fast image search (no LLM) |
| POST | `/search_hybrid` | — | Fast hybrid search (no LLM) |
| POST | `/rag_query` | Ollama | Full RAG text answer |
| POST | `/rag_image_query` | Ollama | Full RAG image answer |
| POST | `/rag_hybrid_query` | Ollama | Full RAG hybrid answer |
| POST | `/compare_products` | **Groq** | Multi-product comparison table |
| POST | `/explain_product` | **HuggingFace** | Feature extraction |
| GET | `/top_rated` | Ollama | Top-rated products |
| GET | `/brands` | — | All unique brands |
| GET | `/categories` | — | All unique categories |

---

## Tech Stack

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (text), `openai/clip-vit-base-patch32` (image)
- **Vector DB**: FAISS (IndexFlatIP, cosine similarity)
- **LLMs**: Ollama llama3 · HuggingFace flan-t5-base · Groq llama3-8b-8192
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Streamlit
- **Data**: Pandas, NumPy

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required** for comparison features |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `HF_MODEL` | `google/flan-t5-base` | HuggingFace model |
| `API_PORT` | `8000` | FastAPI port |
| `API_BASE_URL` | `http://localhost:8000` | URL for Streamlit to call |
