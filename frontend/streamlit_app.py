"""
Streamlit frontend for the Flipkart Multimodal RAG system.

Features:
  - Text search with natural language queries
  - Image upload for visual similarity search
  - Hybrid text + image search
  - Product comparison (powered by Groq)
  - Product feature explanation (powered by HuggingFace)
  - Filters: price range, brand, category, rating
  - Top-rated products browser
  - LLM provider status sidebar
"""

import io
import os
import sys
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Flipkart AI Shopping Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2874f0 0%, #fb641b 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.9; font-size: 1rem; }

    .product-card {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s;
    }
    .product-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.12); }

    .price-tag {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .retail-price {
        color: #9e9e9e;
        text-decoration: line-through;
        font-size: 0.9rem;
    }
    .discount-badge {
        background: #fff3e0;
        color: #e65100;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .rating-badge {
        background: #388e3c;
        color: white;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .provider-badge {
        background: #e3f2fd;
        color: #1565c0;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .answer-box {
        background: #f8f9fa;
        border-left: 4px solid #2874f0;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 1rem;
        line-height: 1.6;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2874f0, #1a5cc8);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
    }
    .stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_get(endpoint: str, params: dict = None) -> dict | None:
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API server. Make sure it's running on port 8000.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post_json(endpoint: str, data: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API server.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post_form(endpoint: str, data: dict, files: dict = None) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}{endpoint}", data=data, files=files, timeout=120
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API server.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🛒 Flipkart AI Shopping Assistant</h1>
        <p>Multimodal RAG · Text · Image · Hybrid Search · Powered by Ollama · HuggingFace · Groq</p>
    </div>
    """, unsafe_allow_html=True)


def render_product_card(product: dict, idx: int = 0, show_checkbox: bool = False) -> bool:
    """Render a product card. Returns True if checkbox is selected."""
    name = product.get("product_name", "Unknown Product")
    brand = product.get("brand", "N/A")
    price = product.get("discounted_price")
    retail = product.get("retail_price")
    rating = product.get("product_rating")
    category = product.get("category", "N/A")
    image_url = product.get("image_url", "")
    desc = str(product.get("description", ""))[:200].strip()
    score = product.get("score", None)
    uid = product.get("uniq_id", "")

    selected = False

    with st.container():
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        col_img, col_info = st.columns([1, 3])

        with col_img:
            if image_url:
                try:
                    st.image(image_url, width=130, use_container_width=False)
                except Exception:
                    st.markdown("🖼️ *No image*")
            else:
                st.markdown("🖼️ *No image*")

        with col_info:
            header_col, badge_col = st.columns([3, 1])
            with header_col:
                st.markdown(f"**{name}**")
                st.caption(f"Brand: {brand} | Category: {category}")

            with badge_col:
                if show_checkbox:
                    selected = st.checkbox("Compare", key=f"cmp_{uid}_{idx}")

            # Price row
            price_parts = []
            if price:
                price_parts.append(f'<span class="price-tag">Rs.{price:,.0f}</span>')
            if retail and price and retail > price:
                pct = int((retail - price) / retail * 100)
                price_parts.append(f'<span class="retail-price">Rs.{retail:,.0f}</span>')
                price_parts.append(f'<span class="discount-badge">{pct}% off</span>')
            if rating:
                price_parts.append(f'<span class="rating-badge">★ {rating}</span>')
            if score is not None:
                price_parts.append(f'<small style="color:#9e9e9e">score: {score:.3f}</small>')

            st.markdown(" &nbsp; ".join(price_parts), unsafe_allow_html=True)

            if desc:
                st.caption(desc + ("..." if len(str(product.get("description", ""))) > 200 else ""))

        st.markdown("</div>", unsafe_allow_html=True)

    return selected


def render_answer_box(answer: str, provider: str):
    st.markdown(
        f'<div class="answer-box">{answer}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="provider-badge">🤖 {provider}</span>',
        unsafe_allow_html=True,
    )


def render_sidebar_filters() -> dict:
    """Render filter controls in the sidebar. Returns filter dict."""
    st.sidebar.markdown("### 🔧 Filters")

    # Price range
    price_enabled = st.sidebar.checkbox("Filter by price", value=False)
    min_price, max_price = None, None
    if price_enabled:
        col1, col2 = st.sidebar.columns(2)
        min_price = col1.number_input("Min (Rs.)", min_value=0, value=0, step=100)
        max_price = col2.number_input("Max (Rs.)", min_value=0, value=50000, step=100)

    # Brand filter
    brand_input = st.sidebar.text_input("Brand (optional)", placeholder="e.g. Samsung")
    brand = brand_input.strip() or None

    # Category filter
    category_input = st.sidebar.text_input("Category (optional)", placeholder="e.g. Clothing")
    category = category_input.strip() or None

    # Rating filter
    rating_enabled = st.sidebar.checkbox("Minimum rating", value=False)
    min_rating = None
    if rating_enabled:
        min_rating = st.sidebar.slider("Min rating", 1.0, 5.0, 3.5, 0.5)

    # Number of results
    k = st.sidebar.slider("Number of results", 3, 15, 5)

    return {
        "min_price": min_price if price_enabled else None,
        "max_price": max_price if price_enabled else None,
        "brand": brand,
        "category": category,
        "min_rating": min_rating,
        "k": k,
    }


def render_provider_status():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 LLM Providers")
    health = api_get("/health")
    if health:
        providers = health.get("providers", {})
        icons = {
            "ollama": "🦙",
            "huggingface": "🤗",
            "groq": "⚡",
        }
        labels = {
            "ollama": "Ollama (llama3)",
            "huggingface": "HuggingFace (flan-t5)",
            "groq": "Groq (llama3-8b)",
        }
        for name, available in providers.items():
            icon = icons.get(name, "🤖")
            label = labels.get(name, name)
            status = "✅ Online" if available else "❌ Offline"
            st.sidebar.markdown(f"{icon} **{label}** — {status}")

        stats = health.get("index_stats", {})
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Index Stats")
        st.sidebar.markdown(f"📝 Text vectors: **{stats.get('text_index_vectors', 0):,}**")
        st.sidebar.markdown(f"🖼️ Image vectors: **{stats.get('image_index_vectors', 0):,}**")
    else:
        st.sidebar.warning("API not reachable")


# ---------------------------------------------------------------------------
# Page: Text Search
# ---------------------------------------------------------------------------

def page_text_search(filters: dict):
    st.markdown("## 📝 Text Search")
    st.markdown("Ask anything about products in natural language.")

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Your query",
            placeholder="e.g. Best phones under 20000, cheap running shoes, laptops for coding",
            label_visibility="collapsed",
        )
    with col2:
        search_btn = st.button("🔍 Search", use_container_width=True)

    # Quick example queries
    st.markdown("**Quick examples:**")
    examples = [
        "Best phones under 20000",
        "Cheap running shoes",
        "Laptops for coding",
        "Best rated headphones",
        "Suggest a sofa bed",
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}"):
            query = ex
            search_btn = True

    if search_btn and query:
        with st.spinner("Searching products and generating answer..."):
            payload = {
                "query": query,
                "k": filters["k"],
                "min_price": filters["min_price"],
                "max_price": filters["max_price"],
                "brand": filters["brand"],
                "category": filters["category"],
                "min_rating": filters["min_rating"],
            }
            result = api_post_json("/rag_query", payload)

        if result:
            st.markdown("### 💬 AI Answer")
            render_answer_box(result.get("answer", ""), result.get("provider", ""))

            products = result.get("products", [])
            if products:
                st.markdown(f"### 🛍️ Retrieved Products ({len(products)})")
                selected_ids = []
                for i, p in enumerate(products):
                    sel = render_product_card(p, idx=i, show_checkbox=True)
                    if sel:
                        selected_ids.append(p.get("uniq_id", ""))

                # Comparison trigger
                if len(selected_ids) >= 2:
                    st.markdown("---")
                    compare_query = st.text_input(
                        "Comparison question",
                        value="Compare these products and tell me which is best value",
                        key="cmp_q_text",
                    )
                    if st.button("⚡ Compare with Groq", key="cmp_btn_text"):
                        with st.spinner("Comparing with Groq (llama3-8b-8192)..."):
                            cmp_result = api_post_json(
                                "/compare_products",
                                {"product_ids": selected_ids, "query": compare_query},
                            )
                        if cmp_result:
                            st.markdown("### 📊 Comparison (Groq)")
                            render_answer_box(
                                cmp_result.get("answer", ""),
                                cmp_result.get("provider", ""),
                            )


# ---------------------------------------------------------------------------
# Page: Image Search
# ---------------------------------------------------------------------------

def page_image_search(filters: dict):
    st.markdown("## 🖼️ Image Search")
    st.markdown("Upload a product image to find visually similar items.")

    uploaded = st.file_uploader(
        "Upload product image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    text_hint = st.text_input(
        "Optional text hint",
        placeholder="e.g. Find cheaper alternatives, similar color, same brand",
    )

    if uploaded:
        col_prev, col_btn = st.columns([1, 2])
        with col_prev:
            st.image(uploaded, caption="Uploaded image", width=200)
        with col_btn:
            st.markdown("**Image ready.** Click search to find similar products.")
            search_btn = st.button("🔍 Find Similar Products", use_container_width=True)

        if search_btn:
            with st.spinner("Analyzing image and searching..."):
                img_bytes = uploaded.getvalue()
                files = {"file": (uploaded.name, img_bytes, uploaded.type)}
                data = {
                    "text_query": text_hint,
                    "k": filters["k"],
                }
                result = api_post_form("/rag_image_query", data=data, files=files)

            if result:
                st.markdown("### 💬 AI Answer")
                render_answer_box(result.get("answer", ""), result.get("provider", ""))

                products = result.get("products", [])
                if products:
                    st.markdown(f"### 🛍️ Similar Products ({len(products)})")
                    for i, p in enumerate(products):
                        render_product_card(p, idx=i)


# ---------------------------------------------------------------------------
# Page: Hybrid Search
# ---------------------------------------------------------------------------

def page_hybrid_search(filters: dict):
    st.markdown("## 🔀 Hybrid Search")
    st.markdown("Combine a text query **and** an image for the most precise results.")
    st.info(
        "Example: Upload a phone image and type 'Find a cheaper version of this phone under 15000'"
    )

    col_left, col_right = st.columns(2)

    with col_left:
        query = st.text_input(
            "Text query",
            placeholder="Find a cheaper version of this product",
        )
        text_weight = st.slider("Text weight", 0.0, 1.0, 0.5, 0.1)

    with col_right:
        uploaded = st.file_uploader(
            "Product image",
            type=["jpg", "jpeg", "png", "webp"],
            key="hybrid_upload",
        )
        image_weight = st.slider("Image weight", 0.0, 1.0, 0.5, 0.1)

    if uploaded:
        st.image(uploaded, caption="Query image", width=180)

    price_col1, price_col2 = st.columns(2)
    min_price = price_col1.number_input("Min price (Rs.)", value=0, step=100)
    max_price = price_col2.number_input("Max price (Rs.)", value=100000, step=1000)

    search_btn = st.button("🔀 Hybrid Search", use_container_width=True)

    if search_btn and query and uploaded:
        with st.spinner("Running hybrid search..."):
            img_bytes = uploaded.getvalue()
            files = {"file": (uploaded.name, img_bytes, uploaded.type)}
            data = {
                "query": query,
                "k": filters["k"],
                "text_weight": text_weight,
                "image_weight": image_weight,
                "min_price": min_price if min_price > 0 else None,
                "max_price": max_price if max_price < 100000 else None,
            }
            result = api_post_form("/rag_hybrid_query", data=data, files=files)

        if result:
            st.markdown("### 💬 AI Answer")
            render_answer_box(result.get("answer", ""), result.get("provider", ""))

            products = result.get("products", [])
            if products:
                st.markdown(f"### 🛍️ Hybrid Results ({len(products)})")
                for i, p in enumerate(products):
                    render_product_card(p, idx=i)

    elif search_btn and not uploaded:
        st.warning("Please upload an image for hybrid search.")
    elif search_btn and not query:
        st.warning("Please enter a text query.")


# ---------------------------------------------------------------------------
# Page: Compare Products
# ---------------------------------------------------------------------------

def page_compare(filters: dict):
    st.markdown("## ⚡ Product Comparison")
    st.markdown(
        "Search for products, select them, and get a detailed comparison powered by **Groq (llama3-8b-8192)**."
    )

    query = st.text_input(
        "Search for products to compare",
        placeholder="e.g. wireless headphones, gaming laptops",
    )

    if st.button("🔍 Search", key="cmp_search") and query:
        with st.spinner("Searching..."):
            result = api_post_json(
                "/search_text",
                {"query": query, "k": 10},
            )
        if result:
            st.session_state["cmp_products"] = result.get("results", [])

    products = st.session_state.get("cmp_products", [])
    if products:
        st.markdown(f"**Select 2–5 products to compare:**")
        selected_ids = []
        for i, p in enumerate(products):
            sel = render_product_card(p, idx=i, show_checkbox=True)
            if sel:
                selected_ids.append(p.get("uniq_id", ""))

        if selected_ids:
            st.info(f"{len(selected_ids)} product(s) selected")

        compare_query = st.text_input(
            "What do you want to know?",
            value="Which product offers the best value for money?",
            key="cmp_q",
        )

        if st.button("⚡ Compare with Groq", key="cmp_go") and len(selected_ids) >= 2:
            with st.spinner("Groq is analyzing products..."):
                result = api_post_json(
                    "/compare_products",
                    {"product_ids": selected_ids, "query": compare_query},
                )
            if result:
                st.markdown("### 📊 Groq Comparison Analysis")
                render_answer_box(result.get("answer", ""), result.get("provider", ""))
        elif st.button and len(selected_ids) < 2:
            st.warning("Select at least 2 products to compare.")


# ---------------------------------------------------------------------------
# Page: Explain Product
# ---------------------------------------------------------------------------

def page_explain():
    st.markdown("## 🤗 Product Explainer")
    st.markdown(
        "Enter a product ID to get a structured feature breakdown powered by **HuggingFace (flan-t5-base)**."
    )

    st.info(
        "Tip: Search for a product first, then copy its Product ID from the results."
    )

    # Quick search to find product IDs
    search_query = st.text_input("Search for a product first", placeholder="e.g. Samsung phone")
    if st.button("🔍 Search", key="explain_search") and search_query:
        with st.spinner("Searching..."):
            result = api_post_json("/search_text", {"query": search_query, "k": 5})
        if result:
            st.session_state["explain_products"] = result.get("results", [])

    explain_products = st.session_state.get("explain_products", [])
    if explain_products:
        st.markdown("**Click a product ID to explain it:**")
        for p in explain_products:
            uid = p.get("uniq_id", "")
            name = p.get("product_name", "Unknown")
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**{name}**  \n`{uid}`")
            if col2.button("Explain", key=f"exp_{uid}"):
                st.session_state["explain_id"] = uid

    product_id = st.text_input(
        "Or enter Product ID manually",
        value=st.session_state.get("explain_id", ""),
        placeholder="e.g. c2d766ca982eca8304150849735ffef9",
    )

    if st.button("🤗 Explain with HuggingFace", key="explain_go") and product_id:
        with st.spinner("HuggingFace flan-t5 is extracting features..."):
            result = api_post_json("/explain_product", {"product_id": product_id.strip()})
        if result:
            product = result.get("product")
            if product:
                render_product_card(product)
            st.markdown("### 🔍 Feature Extraction")
            render_answer_box(result.get("answer", ""), result.get("provider", ""))


# ---------------------------------------------------------------------------
# Page: Top Rated
# ---------------------------------------------------------------------------

def page_top_rated():
    st.markdown("## ⭐ Top Rated Products")
    st.markdown("Browse the highest-rated products with an AI-generated summary.")

    col1, col2 = st.columns(2)
    category = col1.text_input("Filter by category (optional)", placeholder="e.g. Clothing")
    k = col2.slider("Number of products", 5, 20, 10)

    if st.button("⭐ Get Top Rated", use_container_width=True):
        with st.spinner("Fetching top rated products..."):
            params = {"k": k}
            if category.strip():
                params["category"] = category.strip()
            result = api_get("/top_rated", params=params)

        if result:
            st.markdown("### 💬 AI Summary")
            render_answer_box(result.get("answer", ""), result.get("provider", ""))

            products = result.get("products", [])
            if products:
                st.markdown(f"### 🏆 Top {len(products)} Products")
                for i, p in enumerate(products):
                    render_product_card(p, idx=i)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    render_header()

    # Sidebar
    filters = render_sidebar_filters()
    render_provider_status()

    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Navigation")
    pages = {
        "📝 Text Search": "text",
        "🖼️ Image Search": "image",
        "🔀 Hybrid Search": "hybrid",
        "⚡ Compare Products": "compare",
        "🤗 Explain Product": "explain",
        "⭐ Top Rated": "top_rated",
    }
    selected_page = st.sidebar.radio(
        "Go to",
        list(pages.keys()),
        label_visibility="collapsed",
    )
    page_key = pages[selected_page]

    # Initialize session state
    if "cmp_products" not in st.session_state:
        st.session_state["cmp_products"] = []
    if "explain_products" not in st.session_state:
        st.session_state["explain_products"] = []
    if "explain_id" not in st.session_state:
        st.session_state["explain_id"] = ""

    # Render selected page
    if page_key == "text":
        page_text_search(filters)
    elif page_key == "image":
        page_image_search(filters)
    elif page_key == "hybrid":
        page_hybrid_search(filters)
    elif page_key == "compare":
        page_compare(filters)
    elif page_key == "explain":
        page_explain()
    elif page_key == "top_rated":
        page_top_rated()

    # Footer
    st.markdown("---")
    st.markdown(
        "<center><small>Flipkart Multimodal RAG · "
        "Ollama llama3 · HuggingFace flan-t5 · Groq llama3-8b-8192 · "
        "FAISS · CLIP · SentenceTransformers</small></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
