"""
Data loading and preprocessing pipeline for the Flipkart retail dataset.
Handles cleaning, normalization, and feature extraction from raw CSV data.
"""

import ast
import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Columns we actually need downstream
REQUIRED_COLUMNS = [
    "uniq_id",
    "product_name",
    "product_category_tree",
    "retail_price",
    "discounted_price",
    "image",
    "description",
    "product_rating",
    "overall_rating",
    "brand",
    "product_specifications",
    "product_url",
]


def _parse_price(value) -> Optional[float]:
    """Convert a price-like value to float, returning None on failure."""
    if pd.isna(value):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _parse_category(raw: str) -> str:
    """Extract a clean top-level category string from the nested JSON-like tree."""
    if pd.isna(raw) or not raw:
        return "Unknown"
    try:
        cats = ast.literal_eval(raw)
        if isinstance(cats, list) and cats:
            parts = [p.strip() for p in cats[0].split(">>")]
            return parts[0] if parts else "Unknown"
    except Exception:
        pass
    # Fallback: grab first segment before >>
    match = re.match(r'"?([^">]+)', str(raw))
    return match.group(1).strip() if match else "Unknown"


def _parse_first_image_url(raw: str) -> Optional[str]:
    """Return the first image URL from the JSON-like image list column."""
    if pd.isna(raw) or not raw:
        return None
    try:
        urls = ast.literal_eval(raw)
        if isinstance(urls, list) and urls:
            return urls[0].strip()
    except Exception:
        pass
    # Fallback: extract first URL with regex
    match = re.search(r'https?://[^\s"\']+', str(raw))
    return match.group(0) if match else None


def _parse_all_image_urls(raw: str) -> list[str]:
    """Return all image URLs from the JSON-like image list column."""
    if pd.isna(raw) or not raw:
        return []
    try:
        urls = ast.literal_eval(raw)
        if isinstance(urls, list):
            return [u.strip() for u in urls if u.strip()]
    except Exception:
        pass
    return re.findall(r'https?://[^\s"\']+', str(raw))


def _parse_rating(value) -> Optional[float]:
    """Normalize rating strings like '4.2' or 'No rating available' to float."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s.lower().startswith("no"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _build_text_field(row: pd.Series) -> str:
    """
    Concatenate key fields into a single rich text string for embedding.
    This is the primary field used by the text retriever.
    For egs Nike Running Shoes | Brand: Nike | Category: Footwear | Price: Rs.2999 | Rating: 4.5 | Comfortable running shoes for daily use
    """
    parts = []

    if pd.notna(row.get("product_name")):
        parts.append(str(row["product_name"]))

    if pd.notna(row.get("brand")) and str(row["brand"]).lower() not in ("nan", ""):
        parts.append(f"Brand: {row['brand']}")

    if pd.notna(row.get("category")):
        parts.append(f"Category: {row['category']}")

    if pd.notna(row.get("discounted_price")):
        parts.append(f"Price: Rs.{row['discounted_price']:.0f}")

    if pd.notna(row.get("product_rating")):
        parts.append(f"Rating: {row['product_rating']}")

    if pd.notna(row.get("description")) and str(row["description"]).strip():
        desc = str(row["description"])[:500]  # cap to avoid token overflow
        parts.append(desc)

    return " | ".join(parts)


def load_and_preprocess(
    csv_path: str,
    max_rows: Optional[int] = None,
    min_price: float = 0.0,
) -> pd.DataFrame:
    """
    Load the Flipkart CSV, clean and enrich it, and return a processed DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the raw Flipkart CSV file.
    max_rows : int, optional
        Limit the number of rows loaded (useful for development/testing).
    min_price : float
        Drop products with discounted_price below this threshold.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with additional derived columns:
        - category        : top-level category string
        - image_url       : first product image URL
        - all_image_urls  : list of all image URLs
        - retail_price    : float
        - discounted_price: float
        - product_rating  : float or NaN
        - overall_rating  : float or NaN
        - text_field      : rich text string for embedding
    """
    logger.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(
        csv_path,
        usecols=lambda c: c in REQUIRED_COLUMNS,
        nrows=max_rows,
        low_memory=False,
    )
    logger.info("Loaded %d rows", len(df))

    # --- Price normalization ---
    df["retail_price"] = df["retail_price"].apply(_parse_price)
    df["discounted_price"] = df["discounted_price"].apply(_parse_price)

    # Drop rows with no usable price
    before = len(df)
    df = df[df["discounted_price"].notna() & (df["discounted_price"] >= min_price)]
    logger.info("Dropped %d rows with missing/invalid price", before - len(df))

    # --- Rating normalization ---
    df["product_rating"] = df["product_rating"].apply(_parse_rating)
    df["overall_rating"] = df["overall_rating"].apply(_parse_rating)

    # --- Category parsing ---
    df["category"] = df["product_category_tree"].apply(_parse_category)

    # --- Image URL extraction ---
    df["image_url"] = df["image"].apply(_parse_first_image_url)
    df["all_image_urls"] = df["image"].apply(_parse_all_image_urls)

    # --- Text field for embedding ---
    df["text_field"] = df.apply(_build_text_field, axis=1)

    # --- Clean brand / product_name ---
    df["brand"] = df["brand"].fillna("Unknown").astype(str).str.strip()
    df["product_name"] = df["product_name"].fillna("").astype(str).str.strip()

    # --- Reset index and add integer id ---
    df = df.reset_index(drop=True)
    df["idx"] = df.index

    logger.info("Preprocessing complete. Final dataset: %d rows", len(df))
    return df


def get_unique_brands(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique brand names."""
    return sorted(df["brand"].dropna().unique().tolist())


def get_unique_categories(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique top-level categories."""
    return sorted(df["category"].dropna().unique().tolist())


def filter_dataframe(
    df: pd.DataFrame,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    brands: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    min_rating: Optional[float] = None,
) -> pd.DataFrame:
    """
    Apply user-facing filters to the product DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    min_price, max_price : float, optional
        Inclusive price bounds on discounted_price.
    brands : list[str], optional
        Keep only rows matching these brand names.
    categories : list[str], optional
        Keep only rows matching these top-level categories.
    min_rating : float, optional
        Minimum product_rating threshold.

    Returns
    -------
    pd.DataFrame
        Filtered subset.
    """
    mask = pd.Series([True] * len(df), index=df.index)

    if min_price is not None:
        mask &= df["discounted_price"] >= min_price
    if max_price is not None:
        mask &= df["discounted_price"] <= max_price
    if brands:
        mask &= df["brand"].isin(brands)
    if categories:
        mask &= df["category"].isin(categories)
    if min_rating is not None:
        mask &= df["product_rating"].fillna(0) >= min_rating

    return df[mask].copy()


if __name__ == "__main__":
    import sys

    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/flipkart_com-ecommerce_sample.csv"
    df = load_and_preprocess(csv_file, max_rows=5000)
    print(df[["product_name", "brand", "category", "discounted_price", "product_rating", "image_url"]].head(10))
    print(f"\nShape: {df.shape}")
    print(f"Categories: {get_unique_categories(df)[:10]}")
