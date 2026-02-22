"""
Retriever layer — high-level search interface that sits between the
vector store and the RAG pipeline.

Supports:
  - search_text(query, k, filters)   → text embedding search
  - search_image(image, k)           → CLIP image embedding search
  - search_hybrid(text, image, k)    → fused text + image search
  - search_by_clip_text(query, k)    → CLIP text→image cross-modal search
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from image_embeddings import CLIPImageEmbedder
from text_embeddings import TextEmbedder
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetailRetriever:
    """
    Unified retriever for the Flipkart multimodal RAG system.

    Wraps the VectorStore and embedding models to provide a clean
    search API used by both the RAG pipeline and the FastAPI layer.

    Parameters
    ----------
    vector_store : VectorStore
        Pre-loaded vector store instance.
    text_embedder : TextEmbedder, optional
        Reuse an existing embedder to avoid reloading the model.
    clip_embedder : CLIPImageEmbedder, optional
        Reuse an existing CLIP embedder.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        text_embedder: Optional[TextEmbedder] = None,
        clip_embedder: Optional[CLIPImageEmbedder] = None,
    ):
        self.store = vector_store
        self._text_embedder = text_embedder or TextEmbedder()
        self._clip_embedder = clip_embedder or CLIPImageEmbedder()

    # ------------------------------------------------------------------
    # Public search methods
    # ------------------------------------------------------------------

    def search_text(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search products by natural-language text query.

        Parameters
        ----------
        query : str
            User query, e.g. "best phones under 20000".
        k : int
            Number of results to return (after optional filtering).
        filters : dict, optional
            Post-retrieval filters. Supported keys:
              - min_price (float)
              - max_price (float)
              - brand (str or list[str])
              - category (str or list[str])
              - min_rating (float)

        Returns
        -------
        list[dict]
            Ranked product metadata dicts with `score` field.
        """
        logger.info("Text search: '%s' (k=%d)", query, k)
        fetch_k = k * 3 if filters else k  # over-fetch to allow for filtering
        vec = self._text_embedder.embed_single(query)
        results = self.store.search_text_index(vec, k=fetch_k)
        if filters:
            results = _apply_filters(results, filters)
        return results[:k]

    def search_image(
        self,
        image: Image.Image,
        k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search products by uploading a product image.

        Parameters
        ----------
        image : PIL.Image.Image
            Query image (resized internally).
        k : int
        filters : dict, optional

        Returns
        -------
        list[dict]
        """
        logger.info("Image search (k=%d)", k)
        fetch_k = k * 3 if filters else k
        vec = self._clip_embedder.embed_single_image(image)
        results = self.store.search_image_index(vec, k=fetch_k)
        if filters:
            results = _apply_filters(results, filters)
        return results[:k]

    def search_hybrid(
        self,
        text: str,
        image: Image.Image,
        k: int = 5,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Hybrid search combining a text query and an uploaded image.

        Example use case: "Find a cheaper version of this product"
        where the user also uploads an image.

        Parameters
        ----------
        text : str
            Text component of the query.
        image : PIL.Image.Image
            Image component of the query.
        k : int
        text_weight, image_weight : float
            Score fusion weights.
        filters : dict, optional

        Returns
        -------
        list[dict]
        """
        logger.info("Hybrid search: '%s' (k=%d, tw=%.2f, iw=%.2f)", text, k, text_weight, image_weight)
        fetch_k = k * 3 if filters else k

        text_vec = self._text_embedder.embed_single(text)
        image_vec = self._clip_embedder.embed_single_image(image)

        results = self.store.search_hybrid(
            text_vec, image_vec, k=fetch_k,
            text_weight=text_weight, image_weight=image_weight,
        )
        if filters:
            results = _apply_filters(results, filters)
        return results[:k]

    def search_by_clip_text(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Cross-modal search: encode a text query with CLIP and search the image index.
        Useful for queries like "red running shoes" against the image index.

        Parameters
        ----------
        query : str
        k : int
        filters : dict, optional

        Returns
        -------
        list[dict]
        """
        logger.info("CLIP text→image search: '%s' (k=%d)", query, k)
        fetch_k = k * 3 if filters else k
        vec = self._clip_embedder.embed_single_text(query)
        results = self.store.search_image_index(vec, k=fetch_k)
        if filters:
            results = _apply_filters(results, filters)
        return results[:k]

    def get_top_rated(self, k: int = 10, category: Optional[str] = None) -> list[dict]:
        """
        Return top-rated products, optionally filtered by category.
        Uses the metadata directly (no vector search needed).
        """
        meta = self.store.text_meta
        if category:
            meta = [m for m in meta if str(m.get("category", "")).lower() == category.lower()]
        rated = [m for m in meta if m.get("product_rating") is not None]
        rated.sort(key=lambda x: float(x["product_rating"]), reverse=True)
        return rated[:k]

    def compare_products(self, product_ids: list[str]) -> list[dict]:
        """
        Retrieve full metadata for a list of product uniq_ids.
        Used by the comparison feature.
        """
        id_set = set(product_ids)
        return [m for m in self.store.text_meta if m.get("uniq_id") in id_set]


# ------------------------------------------------------------------
# Filter helpers
# ------------------------------------------------------------------

def _apply_filters(results: list[dict], filters: dict) -> list[dict]:
    """
    Apply post-retrieval metadata filters to a list of result dicts.

    Supported filter keys:
      min_price, max_price, brand, category, min_rating
    """
    out = []
    min_price = filters.get("min_price")
    max_price = filters.get("max_price")
    brands = _normalize_list(filters.get("brand"))
    categories = _normalize_list(filters.get("category"))
    min_rating = filters.get("min_rating")

    for item in results:
        price = _safe_float(item.get("discounted_price"))
        rating = _safe_float(item.get("product_rating"))

        if min_price is not None and (price is None or price < min_price):
            continue
        if max_price is not None and (price is None or price > max_price):
            continue
        if brands and str(item.get("brand", "")).lower() not in [b.lower() for b in brands]:
            continue
        if categories and str(item.get("category", "")).lower() not in [c.lower() for c in categories]:
            continue
        if min_rating is not None and (rating is None or rating < min_rating):
            continue
        out.append(item)

    return out


def _normalize_list(value) -> list[str]:
    """Normalize a filter value to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
