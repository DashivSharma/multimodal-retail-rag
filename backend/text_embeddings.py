"""
Text embedding pipeline using SentenceTransformers.
Generates dense vector embeddings for product text fields and persists
them to a FAISS index alongside a metadata pickle file.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_PATH = Path("indexes/text_index.faiss")
DEFAULT_META_PATH = Path("indexes/metadata.pkl")
BATCH_SIZE = 256


class TextEmbedder:
    """
    Wraps a SentenceTransformer model to produce and manage text embeddings.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier.
    model : SentenceTransformer
        Loaded model instance (lazy-loaded on first use).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading SentenceTransformer model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of strings into L2-normalized float32 embeddings.

        Parameters
        ----------
        texts : list[str]
        show_progress : bool
            Show tqdm progress bar during batched encoding.

        Returns
        -------
        np.ndarray of shape (N, embedding_dim)
        """
        logger.info("Encoding %d texts in batches of %d", len(texts), BATCH_SIZE)
        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # cosine similarity via inner product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Encode a single query string. Returns shape (1, dim)."""
        return self.embed([text], show_progress=False)


def build_text_index(
    df: pd.DataFrame,
    text_column: str = "text_field",
    index_path: Path = DEFAULT_INDEX_PATH,
    meta_path: Path = DEFAULT_META_PATH,
    model_name: str = DEFAULT_MODEL,
) -> tuple[faiss.Index, list[dict]]:
    """
    Build a FAISS flat inner-product index from the product text fields.

    Since embeddings are L2-normalized, inner product == cosine similarity.
    Uses IndexFlatIP for exact search; swap to IndexIVFFlat for large datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed product DataFrame (must contain `text_column` and `idx`).
    text_column : str
        Column name holding the rich text strings to embed.
    index_path : Path
        Where to save the FAISS index file.
    meta_path : Path
        Where to save the metadata pickle.
    model_name : str
        SentenceTransformer model identifier.

    Returns
    -------
    (faiss.Index, list[dict])
        The built index and the metadata list.
    """
    index_path = Path(index_path)
    meta_path = Path(meta_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    embedder = TextEmbedder(model_name)
    texts = df[text_column].fillna("").tolist()
    embeddings = embedder.embed(texts)

    dim = embeddings.shape[1]
    logger.info("Building FAISS IndexFlatIP with dim=%d, n=%d", dim, len(embeddings))
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Build metadata list aligned with FAISS row indices
    meta_cols = [
        "idx", "uniq_id", "product_name", "brand", "category",
        "retail_price", "discounted_price", "product_rating",
        "overall_rating", "image_url", "all_image_urls",
        "description", "text_field", "product_url",
    ]
    available = [c for c in meta_cols if c in df.columns]
    metadata = df[available].to_dict(orient="records")

    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info("Saved text index to %s (%d vectors)", index_path, index.ntotal)
    logger.info("Saved metadata to %s", meta_path)
    return index, metadata


def load_text_index(
    index_path: Path = DEFAULT_INDEX_PATH,
    meta_path: Path = DEFAULT_META_PATH,
) -> tuple[faiss.Index, list[dict]]:
    """
    Load a previously built FAISS text index and its metadata from disk.

    Returns
    -------
    (faiss.Index, list[dict])
    """
    index_path = Path(index_path)
    meta_path = Path(meta_path)

    if not index_path.exists():
        raise FileNotFoundError(f"Text index not found: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    logger.info("Loading text index from %s", index_path)
    index = faiss.read_index(str(index_path))

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    logger.info("Loaded text index: %d vectors", index.ntotal)
    return index, metadata


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_and_preprocess

    csv = sys.argv[1] if len(sys.argv) > 1 else "../data/flipkart_com-ecommerce_sample.csv"
    df = load_and_preprocess(csv, max_rows=2000)
    build_text_index(
        df,
        index_path=Path("../indexes/text_index.faiss"),
        meta_path=Path("../indexes/metadata.pkl"),
    )
