"""
One-time index building script.

Run this before starting the API server:
    python backend/build_indexes.py

Options (set via environment variables or command-line args):
    --csv       Path to Flipkart CSV  (default: data/flipkart_com-ecommerce_sample.csv)
    --max-rows  Limit rows for dev    (default: all rows)
    --no-images Skip image indexing   (faster, text-only mode)
    --max-imgs  Max images to download (default: 2000 for speed)
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure backend is on path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Build FAISS indexes for the RAG system")
    parser.add_argument(
        "--csv",
        default="data/flipkart_com-ecommerce_sample.csv",
        help="Path to Flipkart CSV file",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit number of rows (for development)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image index building (text-only mode)",
    )
    parser.add_argument(
        "--max-imgs",
        type=int,
        default=2000,
        help="Maximum number of images to download and index",
    )
    parser.add_argument(
        "--text-index",
        default="indexes/text_index.faiss",
    )
    parser.add_argument(
        "--text-meta",
        default="indexes/metadata.pkl",
    )
    parser.add_argument(
        "--image-index",
        default="indexes/image_index.faiss",
    )
    parser.add_argument(
        "--image-meta",
        default="indexes/image_metadata.pkl",
    )
    parser.add_argument(
        "--image-dir",
        default="images",
        help="Directory to cache downloaded images",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Load and preprocess data
    # ------------------------------------------------------------------
    from data_loader import load_and_preprocess

    logger.info("=" * 60)
    logger.info("STEP 1: Loading and preprocessing data")
    logger.info("=" * 60)
    df = load_and_preprocess(args.csv, max_rows=args.max_rows)
    logger.info("Dataset ready: %d products", len(df))

    # ------------------------------------------------------------------
    # Step 2: Build text index
    # ------------------------------------------------------------------
    from text_embeddings import build_text_index

    logger.info("=" * 60)
    logger.info("STEP 2: Building text embeddings + FAISS index")
    logger.info("=" * 60)
    build_text_index(
        df,
        index_path=Path(args.text_index),
        meta_path=Path(args.text_meta),
    )
    logger.info("Text index built successfully.")

    # ------------------------------------------------------------------
    # Step 3: Build image index (optional)
    # ------------------------------------------------------------------
    if not args.no_images:
        from image_embeddings import build_image_index

        logger.info("=" * 60)
        logger.info("STEP 3: Downloading images + building CLIP image index")
        logger.info("  (Use --no-images to skip this step)")
        logger.info("=" * 60)
        try:
            build_image_index(
                df,
                image_dir=Path(args.image_dir),
                index_path=Path(args.image_index),
                meta_path=Path(args.image_meta),
                max_images=args.max_imgs,
            )
            logger.info("Image index built successfully.")
        except Exception as e:
            logger.error("Image index build failed: %s", e)
            logger.warning("Continuing without image index. Image search will be unavailable.")
    else:
        logger.info("STEP 3: Skipped (--no-images flag set)")

    logger.info("=" * 60)
    logger.info("Index building complete!")
    logger.info("  Text index:  %s", args.text_index)
    logger.info("  Image index: %s", args.image_index if not args.no_images else "skipped")
    logger.info("Start the API with: python backend/api.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
