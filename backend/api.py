"""
FastAPI backend for the Flipkart Multimodal RAG system.

Endpoints:
  GET  /health              — health check + provider status
  GET  /stats               — index statistics
  POST /search_text         — text-only product search
  POST /search_image        — image-only product search (CLIP)
  POST /search_hybrid       — text + image hybrid search
  POST /rag_query           — full RAG answer (text query → Ollama)
  POST /rag_image_query     — full RAG answer (image query → Ollama)
  POST /rag_hybrid_query    — full RAG answer (hybrid → Ollama)
  POST /compare_products    — product comparison (→ Groq)
  POST /explain_product     — product feature extraction (→ HuggingFace)
  GET  /top_rated           — top-rated products (→ Ollama)
  GET  /brands              — list unique brands
  GET  /categories          — list unique categories
"""

import io
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

# Ensure backend directory is on path
sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline import RAGPipeline, create_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global pipeline instance (loaded at startup)
# ---------------------------------------------------------------------------

pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load indexes and models at startup; clean up at shutdown."""
    global pipeline
    logger.info("Starting up — loading RAG pipeline...")
    try:
        pipeline = create_pipeline(
            text_index_path=os.getenv("TEXT_INDEX_PATH", "indexes/text_index.faiss"),
            text_meta_path=os.getenv("TEXT_META_PATH", "indexes/metadata.pkl"),
            image_index_path=os.getenv("IMAGE_INDEX_PATH", "indexes/image_index.faiss"),
            image_meta_path=os.getenv("IMAGE_META_PATH", "indexes/image_metadata.pkl"),
        )
        logger.info("RAG pipeline ready.")
    except Exception as e:
        logger.error("Failed to load pipeline: %s", e)
        logger.warning("API will start but search endpoints will fail until indexes are built.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Flipkart Multimodal RAG API",
    description=(
        "Production-quality Retail RAG system with text, image, and hybrid search. "
        "Powered by Ollama (llama3), HuggingFace (flan-t5), and Groq (llama3-8b-8192)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_pipeline() -> RAGPipeline:
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not loaded. Run build_indexes.py first.",
        )
    return pipeline


def _parse_image(file: UploadFile) -> Image.Image:
    """Read an uploaded file and return a PIL Image."""
    try:
        contents = file.file.read()
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


def _sanitize(obj):
    """
    Recursively replace float NaN/Inf values with None so the response
    is always valid JSON. Python's json module rejects NaN and Infinity.
    """
    import math
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _parse_filters(
    min_price: Optional[float],
    max_price: Optional[float],
    brand: Optional[str],
    category: Optional[str],
    min_rating: Optional[float],
) -> Optional[dict]:
    filters = {}
    if min_price is not None:
        filters["min_price"] = min_price
    if max_price is not None:
        filters["max_price"] = max_price
    if brand:
        filters["brand"] = brand
    if category:
        filters["category"] = category
    if min_rating is not None:
        filters["min_rating"] = min_rating
    return filters or None


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TextSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language product query")
    k: int = Field(5, ge=1, le=20, description="Number of results")
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    min_rating: Optional[float] = None


class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language product query")
    k: int = Field(5, ge=1, le=20)
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    min_rating: Optional[float] = None


class CompareRequest(BaseModel):
    product_ids: list[str] = Field(..., description="List of uniq_id values to compare")
    query: str = Field("Compare these products", description="Comparison intent")


class ExplainRequest(BaseModel):
    product_id: str = Field(..., description="uniq_id of the product to explain")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    """Health check and LLM provider availability."""
    p = _require_pipeline()
    return {
        "status": "ok",
        "providers": p.provider_status(),
        "index_stats": p.retriever.store.stats(),
    }


@app.get("/stats", tags=["System"])
async def get_stats():
    """Return index statistics."""
    p = _require_pipeline()
    return p.retriever.store.stats()


@app.post("/search_text", tags=["Search"])
async def search_text(req: TextSearchRequest):
    """
    Search products by text query. Returns ranked product list without LLM generation.
    Fast — suitable for autocomplete or quick product listing.
    """
    p = _require_pipeline()
    filters = _parse_filters(req.min_price, req.max_price, req.brand, req.category, req.min_rating)
    try:
        results = _sanitize(p.retriever.search_text(req.query, k=req.k, filters=filters))
        return {"query": req.query, "results": results, "count": len(results)}
    except Exception as e:
        logger.error("search_text failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search_image", tags=["Search"])
async def search_image(
    file: UploadFile = File(..., description="Product image to search by"),
    k: int = Form(5),
    min_price: Optional[float] = Form(None),
    max_price: Optional[float] = Form(None),
    brand: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
):
    """
    Search products by uploading an image. Uses CLIP embeddings.
    Returns visually similar products.
    """
    p = _require_pipeline()
    image = _parse_image(file)
    filters = _parse_filters(min_price, max_price, brand, category, None)
    try:
        results = _sanitize(p.retriever.search_image(image, k=k, filters=filters))
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error("search_image failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search_hybrid", tags=["Search"])
async def search_hybrid(
    query: str = Form(...),
    file: UploadFile = File(...),
    k: int = Form(5),
    text_weight: float = Form(0.5),
    image_weight: float = Form(0.5),
    min_price: Optional[float] = Form(None),
    max_price: Optional[float] = Form(None),
):
    """
    Hybrid search combining a text query and an uploaded image.
    Example: 'Find a cheaper version of this product' + image upload.
    """
    p = _require_pipeline()
    image = _parse_image(file)
    filters = _parse_filters(min_price, max_price, None, None, None)
    try:
        results = _sanitize(p.retriever.search_hybrid(
            query, image, k=k,
            text_weight=text_weight, image_weight=image_weight,
            filters=filters,
        ))
        return {"query": query, "results": results, "count": len(results)}
    except Exception as e:
        logger.error("search_hybrid failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag_query", tags=["RAG"])
async def rag_query(req: RAGQueryRequest):
    """
    Full RAG pipeline for text queries.
    Retrieves products → builds context → generates answer via Ollama (llama3).
    """
    p = _require_pipeline()
    filters = _parse_filters(req.min_price, req.max_price, req.brand, req.category, req.min_rating)
    try:
        result = _sanitize(p.answer_text_query(req.query, k=req.k, filters=filters))
        return result
    except Exception as e:
        logger.error("rag_query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag_image_query", tags=["RAG"])
async def rag_image_query(
    file: UploadFile = File(...),
    text_query: str = Form(""),
    k: int = Form(5),
):
    """
    Full RAG pipeline for image queries.
    Retrieves similar products via CLIP → generates answer via Ollama (llama3).
    """
    p = _require_pipeline()
    image = _parse_image(file)
    try:
        result = _sanitize(p.answer_image_query(image, text_query=text_query, k=k))
        return result
    except Exception as e:
        logger.error("rag_image_query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag_hybrid_query", tags=["RAG"])
async def rag_hybrid_query(
    query: str = Form(...),
    file: UploadFile = File(...),
    k: int = Form(5),
    text_weight: float = Form(0.5),
    image_weight: float = Form(0.5),
):
    """
    Full RAG pipeline for hybrid text+image queries.
    Fuses text and image search → generates answer via Ollama (llama3).
    """
    p = _require_pipeline()
    image = _parse_image(file)
    try:
        result = _sanitize(p.answer_hybrid_query(
            query, image, k=k,
            text_weight=text_weight, image_weight=image_weight,
        ))
        return result
    except Exception as e:
        logger.error("rag_hybrid_query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare_products", tags=["RAG"])
async def compare_products(req: CompareRequest):
    """
    Compare multiple products using Groq (llama3-8b-8192).
    Groq's fast inference handles large multi-product context efficiently.
    Returns a structured comparison table and recommendation.
    """
    p = _require_pipeline()
    if len(req.product_ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 product IDs to compare.")
    try:
        result = _sanitize(p.compare_products(req.product_ids, user_query=req.query))
        return result
    except Exception as e:
        logger.error("compare_products failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain_product", tags=["RAG"])
async def explain_product(req: ExplainRequest):
    """
    Explain a product's features using HuggingFace flan-t5-base.
    Returns structured key features and target audience.
    """
    p = _require_pipeline()
    try:
        result = _sanitize(p.explain_product(req.product_id))
        return result
    except Exception as e:
        logger.error("explain_product failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/top_rated", tags=["RAG"])
async def top_rated(
    k: int = Query(10, ge=1, le=50),
    category: Optional[str] = Query(None),
):
    """
    Get top-rated products with an Ollama-generated summary.
    Optionally filter by category.
    """
    p = _require_pipeline()
    try:
        result = _sanitize(p.get_top_rated(k=k, category=category))
        return result
    except Exception as e:
        logger.error("top_rated failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/brands", tags=["Metadata"])
async def get_brands():
    """Return a sorted list of all unique brand names in the catalog."""
    p = _require_pipeline()
    brands = sorted(set(
        str(m.get("brand", "")) for m in p.retriever.store.text_meta
        if m.get("brand") and str(m.get("brand")) not in ("nan", "Unknown", "")
    ))
    return {"brands": brands, "count": len(brands)}


@app.get("/categories", tags=["Metadata"])
async def get_categories():
    """Return a sorted list of all unique top-level categories."""
    p = _require_pipeline()
    cats = sorted(set(
        str(m.get("category", "")) for m in p.retriever.store.text_meta
        if m.get("category") and str(m.get("category")) not in ("nan", "Unknown", "")
    ))
    return {"categories": cats, "count": len(cats)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
        log_level="info",
    )
