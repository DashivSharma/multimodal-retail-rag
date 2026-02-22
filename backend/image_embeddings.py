"""
Image embedding pipeline using OpenAI CLIP (via HuggingFace Transformers).
Downloads product images from URLs, generates CLIP embeddings, and stores
them in a FAISS index for similarity search.
"""

import io
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_IMAGE_INDEX_PATH = Path("indexes/image_index.faiss")
DEFAULT_IMAGE_META_PATH = Path("indexes/image_metadata.pkl")

IMAGE_SIZE = (224, 224)
DOWNLOAD_TIMEOUT = 10       # seconds per image request
MAX_DOWNLOAD_WORKERS = 8    # concurrent download threads
BATCH_SIZE = 64             # images per CLIP forward pass
REQUEST_DELAY = 0.05        # polite delay between requests


class CLIPImageEmbedder:
    """
    Wraps CLIP to produce L2-normalized image and text embeddings.
    Supports both PIL Image objects and raw URL strings.
    """

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL):
        self.model_name = model_name
        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self):
        if self._model is None:
            logger.info("Loading CLIP model: %s on %s", self.model_name, self.device)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()

    @property
    def model(self) -> CLIPModel:
        self._load()
        return self._model

    @property
    def processor(self) -> CLIPProcessor:
        self._load()
        return self._processor

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """
        Encode a list of PIL Images into L2-normalized float32 embeddings.

        Returns
        -------
        np.ndarray of shape (N, 512)
        """
        self._load()
        all_embeddings = []
        for i in range(0, len(images), BATCH_SIZE):
            batch = images[i : i + BATCH_SIZE]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
            all_embeddings.append(feats.cpu().numpy().astype(np.float32))
        return np.vstack(all_embeddings)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """
        Encode text strings via CLIP's text encoder.
        Useful for cross-modal text→image search.

        Returns
        -------
        np.ndarray of shape (N, 512)
        """
        self._load()
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    def embed_single_image(self, image: Image.Image) -> np.ndarray:
        """Embed a single PIL Image. Returns shape (1, 512)."""
        return self.embed_images([image])

    def embed_single_text(self, text: str) -> np.ndarray:
        """Embed a single text query via CLIP. Returns shape (1, 512)."""
        return self.embed_text([text])


def _download_image(url: str, save_dir: Optional[Path] = None) -> Optional[Image.Image]:
    """
    Download an image from a URL and return a PIL Image.
    Optionally cache to disk.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, headers=headers)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = url.split("/")[-1].split("?")[0]
            img.save(save_dir / fname)

        return img
    except Exception as e:
        logger.debug("Failed to download %s: %s", url, e)
        return None


def download_images_parallel(
    urls: list[str],
    save_dir: Optional[Path] = None,
    max_workers: int = MAX_DOWNLOAD_WORKERS,
) -> dict[str, Optional[Image.Image]]:
    """
    Download images in parallel using a thread pool.

    Returns
    -------
    dict mapping url → PIL Image (or None on failure)
    """
    results: dict[str, Optional[Image.Image]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(_download_image, url, save_dir): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            results[url] = future.result()
            time.sleep(REQUEST_DELAY)
    return results


def build_image_index(
    df: pd.DataFrame,
    image_dir: Path = Path("images"),
    index_path: Path = DEFAULT_IMAGE_INDEX_PATH,
    meta_path: Path = DEFAULT_IMAGE_META_PATH,
    model_name: str = DEFAULT_CLIP_MODEL,
    max_images: Optional[int] = None,
) -> tuple[faiss.Index, list[dict]]:
    """
    Download product images, generate CLIP embeddings, and build a FAISS index.

    Only rows with a valid `image_url` are indexed. Rows without images are
    skipped — the image index will be a subset of the full product catalog.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed product DataFrame with `image_url` column.
    image_dir : Path
        Directory to cache downloaded images.
    index_path : Path
        Output path for the FAISS index.
    meta_path : Path
        Output path for the metadata pickle.
    model_name : str
        CLIP model identifier.
    max_images : int, optional
        Cap the number of images to process (for development).

    Returns
    -------
    (faiss.Index, list[dict])
    """
    index_path = Path(index_path)
    meta_path = Path(meta_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter to rows with valid image URLs
    img_df = df[df["image_url"].notna()].copy()
    if max_images:
        img_df = img_df.head(max_images)

    logger.info("Downloading %d product images...", len(img_df))
    urls = img_df["image_url"].tolist()
    url_to_image = download_images_parallel(urls, save_dir=image_dir)

    # Align images with metadata, skipping failed downloads
    valid_rows = []
    valid_images = []
    for _, row in img_df.iterrows():
        img = url_to_image.get(row["image_url"])
        if img is not None:
            valid_rows.append(row)
            valid_images.append(img)

    logger.info("Successfully downloaded %d / %d images", len(valid_images), len(img_df))

    if not valid_images:
        raise RuntimeError("No images could be downloaded. Check network access.")

    embedder = CLIPImageEmbedder(model_name)
    logger.info("Generating CLIP embeddings for %d images...", len(valid_images))
    embeddings = embedder.embed_images(valid_images)

    dim = embeddings.shape[1]
    logger.info("Building FAISS IndexFlatIP with dim=%d, n=%d", dim, len(embeddings))
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    meta_cols = [
        "idx", "uniq_id", "product_name", "brand", "category",
        "retail_price", "discounted_price", "product_rating",
        "overall_rating", "image_url", "all_image_urls",
        "description", "product_url",
    ]
    available = [c for c in meta_cols if c in img_df.columns]
    metadata = [dict(row[available]) for row in valid_rows]

    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info("Saved image index to %s (%d vectors)", index_path, index.ntotal)
    return index, metadata


def load_image_index(
    index_path: Path = DEFAULT_IMAGE_INDEX_PATH,
    meta_path: Path = DEFAULT_IMAGE_META_PATH,
) -> tuple[faiss.Index, list[dict]]:
    """Load a previously built FAISS image index and metadata from disk."""
    index_path = Path(index_path)
    meta_path = Path(meta_path)

    if not index_path.exists():
        raise FileNotFoundError(f"Image index not found: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Image metadata not found: {meta_path}")

    logger.info("Loading image index from %s", index_path)
    index = faiss.read_index(str(index_path))

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    logger.info("Loaded image index: %d vectors", index.ntotal)
    return index, metadata
