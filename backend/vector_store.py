"""
Vector store manager — centralizes loading, caching, and access to both
the text and image FAISS indexes. Acts as the single source of truth for
all index state in the application.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Default index paths (relative to project root; override via env vars)
TEXT_INDEX_PATH = Path("indexes/text_index.faiss")
TEXT_META_PATH = Path("indexes/metadata.pkl")
IMAGE_INDEX_PATH = Path("indexes/image_index.faiss")
IMAGE_META_PATH = Path("indexes/image_metadata.pkl")


class VectorStore:
    """
    Manages FAISS indexes for text and image embeddings.

    Provides a unified interface for:
    - Loading indexes from disk (with in-memory caching)
    - Performing top-k similarity searches
    - Returning results enriched with product metadata

    The store is designed to be instantiated once and reused across
    the lifetime of the API server (singleton pattern recommended).
    """

    def __init__(
        self,
        text_index_path: Path = TEXT_INDEX_PATH,
        text_meta_path: Path = TEXT_META_PATH,
        image_index_path: Path = IMAGE_INDEX_PATH,
        image_meta_path: Path = IMAGE_META_PATH,
    ):
        self.text_index_path = Path(text_index_path)
        self.text_meta_path = Path(text_meta_path)
        self.image_index_path = Path(image_index_path)
        self.image_meta_path = Path(image_meta_path)

        self._text_index: Optional[faiss.Index] = None
        self._text_meta: Optional[list[dict]] = None
        self._image_index: Optional[faiss.Index] = None
        self._image_meta: Optional[list[dict]] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_text_index(self) -> None:
        """Load text FAISS index and metadata into memory."""
        if not self.text_index_path.exists():
            raise FileNotFoundError(
                f"Text index not found at {self.text_index_path}. "
                "Run the build_indexes script first."
            )
        logger.info("Loading text index from %s", self.text_index_path)
        self._text_index = faiss.read_index(str(self.text_index_path))
        with open(self.text_meta_path, "rb") as f:
            self._text_meta = pickle.load(f)
        logger.info("Text index loaded: %d vectors", self._text_index.ntotal)

    def load_image_index(self) -> None:
        """Load image FAISS index and metadata into memory."""
        if not self.image_index_path.exists():
            logger.warning(
                "Image index not found at %s. Image search will be unavailable.",
                self.image_index_path,
            )
            return
        logger.info("Loading image index from %s", self.image_index_path)
        self._image_index = faiss.read_index(str(self.image_index_path))
        with open(self.image_meta_path, "rb") as f:
            self._image_meta = pickle.load(f)
        logger.info("Image index loaded: %d vectors", self._image_index.ntotal)

    def load_all(self) -> None:
        """Load both text and image indexes."""
        self.load_text_index()
        self.load_image_index()

    # ------------------------------------------------------------------
    # Index accessors (lazy-load on first access)
    # ------------------------------------------------------------------

    @property
    def text_index(self) -> faiss.Index:
        if self._text_index is None:
            self.load_text_index()
        return self._text_index

    @property
    def text_meta(self) -> list[dict]:
        if self._text_meta is None:
            self.load_text_index()
        return self._text_meta

    @property
    def image_index(self) -> Optional[faiss.Index]:
        if self._image_index is None and self.image_index_path.exists():
            self.load_image_index()
        return self._image_index

    @property
    def image_meta(self) -> Optional[list[dict]]:
        if self._image_meta is None and self.image_index_path.exists():
            self.load_image_index()
        return self._image_meta

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

    def search_text_index(
        self, query_vector: np.ndarray, k: int = 10
    ) -> list[dict]:
        """
        Search the text FAISS index with a pre-computed query vector.

        Parameters
        ----------
        query_vector : np.ndarray
            Shape (1, dim) float32 L2-normalized embedding.
        k : int
            Number of nearest neighbours to return.

        Returns
        -------
        list[dict]
            Product metadata dicts, each augmented with a `score` field.
        """
        query_vector = _ensure_2d(query_vector)
        scores, indices = self.text_index.search(query_vector, k)
        return _build_results(scores[0], indices[0], self.text_meta)

    def search_image_index(
        self, query_vector: np.ndarray, k: int = 10
    ) -> list[dict]:
        """
        Search the image FAISS index with a pre-computed query vector.

        Parameters
        ----------
        query_vector : np.ndarray
            Shape (1, dim) float32 L2-normalized CLIP embedding.
        k : int
            Number of nearest neighbours to return.

        Returns
        -------
        list[dict]
            Product metadata dicts with `score` field.
        """
        if self.image_index is None:
            raise RuntimeError("Image index is not loaded or does not exist.")
        query_vector = _ensure_2d(query_vector)
        scores, indices = self.image_index.search(query_vector, k)
        return _build_results(scores[0], indices[0], self.image_meta)

    def search_hybrid(
        self,
        text_vector: np.ndarray,
        image_vector: np.ndarray,
        k: int = 10,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
    ) -> list[dict]:
        """
        Fuse text and image similarity scores via weighted reciprocal rank fusion.

        Retrieves 2*k candidates from each index, then re-ranks by combining
        normalized scores. Returns the top-k unique products.

        Parameters
        ----------
        text_vector : np.ndarray
            Shape (1, text_dim) text embedding.
        image_vector : np.ndarray
            Shape (1, image_dim) CLIP image embedding.
        k : int
            Final number of results to return.
        text_weight, image_weight : float
            Relative weights for score fusion (need not sum to 1).

        Returns
        -------
        list[dict]
        """
        fetch_k = min(k * 3, 50)

        text_results = self.search_text_index(text_vector, k=fetch_k)
        image_results = self.search_image_index(image_vector, k=fetch_k)

        # Score map keyed by uniq_id
        score_map: dict[str, dict] = {}

        for rank, item in enumerate(text_results):
            uid = item.get("uniq_id", str(rank))
            score_map.setdefault(uid, {"meta": item, "text_score": 0.0, "image_score": 0.0})
            score_map[uid]["text_score"] = float(item.get("score", 0.0))

        for rank, item in enumerate(image_results):
            uid = item.get("uniq_id", str(rank))
            score_map.setdefault(uid, {"meta": item, "text_score": 0.0, "image_score": 0.0})
            score_map[uid]["image_score"] = float(item.get("score", 0.0))

        # Fuse scores
        fused = []
        for uid, entry in score_map.items():
            combined = (
                text_weight * entry["text_score"]
                + image_weight * entry["image_score"]
            )
            result = dict(entry["meta"])
            result["score"] = combined
            result["text_score"] = entry["text_score"]
            result["image_score"] = entry["image_score"]
            fused.append(result)

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:k]

    # ------------------------------------------------------------------
    # Index stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a summary of loaded index sizes."""
        return {
            "text_index_vectors": self._text_index.ntotal if self._text_index else 0,
            "image_index_vectors": self._image_index.ntotal if self._image_index else 0,
            "text_index_loaded": self._text_index is not None,
            "image_index_loaded": self._image_index is not None,
        }


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _ensure_2d(vec: np.ndarray) -> np.ndarray:
    """Ensure the query vector is shape (1, dim) float32."""
    vec = np.array(vec, dtype=np.float32)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    return vec


def _build_results(
    scores: np.ndarray,
    indices: np.ndarray,
    metadata: list[dict],
) -> list[dict]:
    """
    Zip FAISS search results with metadata, filtering out invalid indices.
    """
    results = []
    for score, idx in zip(scores, indices):
        if idx < 0 or idx >= len(metadata):
            continue
        item = dict(metadata[idx])
        item["score"] = float(score)
        results.append(item)
    return results
