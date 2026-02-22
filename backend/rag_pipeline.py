"""
RAG Pipeline — orchestrates retrieval + LLM generation.

Provider routing per task:
  - search_text / rag_query   → Ollama llama3     (conversational Q&A)
  - explain_product           → HuggingFace flan-t5 (feature extraction)
  - compare_products          → Groq llama3-8b    (structured comparison)
  - search_image / hybrid     → Ollama llama3     (describe found products)
"""

import logging
import os
from pathlib import Path
from typing import Optional

from PIL import Image

from llm_providers import LLMRouter
from retriever import RetailRetriever
from vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

RAG_ANSWER_TEMPLATE = """You are a helpful retail shopping assistant for an Indian e-commerce platform.
Use the product information below to answer the customer's question accurately and helpfully.
If the answer is not in the context, say so honestly.

Customer Question: {query}

Retrieved Products:
{context}

Instructions:
- Recommend the best matching products
- Mention prices in Indian Rupees (Rs.)
- Highlight ratings where available
- Be concise and friendly
- If asked for comparison, compare clearly

Answer:"""


EXPLAIN_PRODUCT_TEMPLATE = """Extract and explain the key features of this product for a customer:

Product: {product_name}
Brand: {brand}
Price: Rs.{price}
Description: {description}

List the top 5 key features and who this product is best suited for:"""


COMPARISON_TEMPLATE = """You are an expert retail analyst. Compare these products for the customer.

Customer Query: {query}

Products to Compare:
{context}

Provide:
1. A comparison table (Name | Price | Rating | Key Feature)
2. Best value pick with reasoning
3. Best quality pick with reasoning
4. Final recommendation based on the query

Be specific and data-driven."""


IMAGE_SEARCH_TEMPLATE = """A customer uploaded a product image and we found these visually similar products.

Customer's additional request: {query}

Similar Products Found:
{context}

Help the customer by:
1. Describing what kind of products were found
2. Highlighting the best matches
3. Mentioning prices and ratings
4. Answering their specific request if any

Response:"""


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(products: list[dict], max_products: int = 5) -> str:
    """
    Format a list of product dicts into a readable context string for the LLM.
    """
    lines = []
    for i, p in enumerate(products[:max_products], 1):
        name = p.get("product_name", "Unknown Product")
        brand = p.get("brand", "N/A")
        price = p.get("discounted_price")
        retail = p.get("retail_price")
        rating = p.get("product_rating", "N/A")
        category = p.get("category", "N/A")
        desc = str(p.get("description", ""))[:300].strip()
        score = p.get("score", 0.0)

        price_str = f"Rs.{price:.0f}" if price else "N/A"
        retail_str = f"Rs.{retail:.0f}" if retail else ""
        discount_str = ""
        if price and retail and retail > price:
            pct = int((retail - price) / retail * 100)
            discount_str = f" ({pct}% off)"

        lines.append(
            f"[{i}] {name}\n"
            f"    Brand: {brand} | Category: {category}\n"
            f"    Price: {price_str}{discount_str}"
            + (f" (MRP: {retail_str})" if retail_str else "")
            + f"\n"
            f"    Rating: {rating} | Relevance Score: {score:.3f}\n"
            f"    Description: {desc}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG pipeline for the Flipkart retail assistant.

    Combines retrieval (RetailRetriever) with generation (LLMRouter) to
    answer product queries, explain products, and compare items.

    Parameters
    ----------
    retriever : RetailRetriever
    llm_router : LLMRouter
    default_k : int
        Default number of products to retrieve per query.
    """

    def __init__(
        self,
        retriever: RetailRetriever,
        llm_router: LLMRouter,
        default_k: int = 5,
    ):
        self.retriever = retriever
        self.router = llm_router
        self.default_k = default_k

    # ------------------------------------------------------------------
    # Core RAG methods
    # ------------------------------------------------------------------

    def answer_text_query(
        self,
        query: str,
        k: int = None,
        filters: Optional[dict] = None,
    ) -> dict:
        """
        Answer a natural-language product query using Ollama (llama3).

        Flow: embed query → FAISS text search → build context → Ollama → answer

        Parameters
        ----------
        query : str
        k : int, optional
        filters : dict, optional

        Returns
        -------
        dict with keys: answer, products, provider, query
        """
        k = k or self.default_k
        logger.info("RAG text query: '%s'", query)

        products = self.retriever.search_text(query, k=k, filters=filters)
        context = _build_context(products)

        prompt = RAG_ANSWER_TEMPLATE.format(query=query, context=context)
        answer = self.router.generate(prompt, task="rag_answer", max_tokens=600)

        return {
            "query": query,
            "answer": answer,
            "products": products,
            "provider": "Ollama (llama3)",
            "num_results": len(products),
        }

    def answer_image_query(
        self,
        image: Image.Image,
        text_query: str = "",
        k: int = None,
        filters: Optional[dict] = None,
    ) -> dict:
        """
        Answer a query based on an uploaded product image using Ollama (llama3).

        Flow: CLIP embed image → FAISS image search → build context → Ollama → answer

        Parameters
        ----------
        image : PIL.Image.Image
        text_query : str
            Optional text to accompany the image.
        k : int, optional
        filters : dict, optional

        Returns
        -------
        dict with keys: answer, products, provider, query
        """
        k = k or self.default_k
        logger.info("RAG image query (text='%s')", text_query)

        products = self.retriever.search_image(image, k=k, filters=filters)
        context = _build_context(products)

        prompt = IMAGE_SEARCH_TEMPLATE.format(
            query=text_query or "Find similar products",
            context=context,
        )
        answer = self.router.generate(prompt, task="rag_answer", max_tokens=500)

        return {
            "query": text_query or "(image search)",
            "answer": answer,
            "products": products,
            "provider": "Ollama (llama3)",
            "num_results": len(products),
        }

    def answer_hybrid_query(
        self,
        text: str,
        image: Image.Image,
        k: int = None,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
        filters: Optional[dict] = None,
    ) -> dict:
        """
        Answer a hybrid text+image query using Ollama (llama3).

        Example: "Find a cheaper version of this product" + uploaded image.

        Returns
        -------
        dict with keys: answer, products, provider, query
        """
        k = k or self.default_k
        logger.info("RAG hybrid query: '%s'", text)

        products = self.retriever.search_hybrid(
            text, image, k=k,
            text_weight=text_weight, image_weight=image_weight,
            filters=filters,
        )
        context = _build_context(products)

        prompt = IMAGE_SEARCH_TEMPLATE.format(query=text, context=context)
        answer = self.router.generate(prompt, task="rag_answer", max_tokens=500)

        return {
            "query": text,
            "answer": answer,
            "products": products,
            "provider": "Ollama (llama3)",
            "num_results": len(products),
        }

    def compare_products(
        self,
        product_ids: list[str],
        user_query: str = "Compare these products",
    ) -> dict:
        """
        Compare multiple products using Groq (llama3-8b-8192).

        Groq's ultra-fast inference handles the large multi-product context
        efficiently, producing a structured comparison table.

        Parameters
        ----------
        product_ids : list[str]
            List of uniq_id values to compare.
        user_query : str
            The user's comparison intent.

        Returns
        -------
        dict with keys: answer, products, provider, query
        """
        logger.info("Comparing %d products via Groq", len(product_ids))
        products = self.retriever.compare_products(product_ids)

        if not products:
            return {
                "query": user_query,
                "answer": "No products found for the given IDs.",
                "products": [],
                "provider": "Groq (llama3-8b-8192)",
            }

        context = _build_context(products, max_products=10)
        prompt = COMPARISON_TEMPLATE.format(query=user_query, context=context)
        answer = self.router.generate(prompt, task="comparison", max_tokens=1024)

        return {
            "query": user_query,
            "answer": answer,
            "products": products,
            "provider": "Groq (llama3-8b-8192)",
        }

    def explain_product(self, product_id: str) -> dict:
        """
        Generate a detailed feature explanation for a single product
        using HuggingFace flan-t5-base.

        Flan-T5 is ideal here: it's instruction-tuned for structured extraction
        and runs locally with no API cost.

        Parameters
        ----------
        product_id : str
            The uniq_id of the product to explain.

        Returns
        -------
        dict with keys: answer, product, provider
        """
        logger.info("Explaining product %s via HuggingFace", product_id)
        matches = self.retriever.compare_products([product_id])

        if not matches:
            return {
                "answer": "Product not found.",
                "product": None,
                "provider": "HuggingFace (flan-t5-base)",
            }

        product = matches[0]
        prompt = EXPLAIN_PRODUCT_TEMPLATE.format(
            product_name=product.get("product_name", "Unknown"),
            brand=product.get("brand", "N/A"),
            price=product.get("discounted_price", "N/A"),
            description=str(product.get("description", ""))[:400],
        )
        answer = self.router.generate(prompt, task="extraction", max_tokens=200)

        return {
            "answer": answer,
            "product": product,
            "provider": "HuggingFace (flan-t5-base)",
        }

    def get_top_rated(
        self,
        k: int = 10,
        category: Optional[str] = None,
    ) -> dict:
        """
        Return top-rated products with an Ollama-generated summary.
        """
        products = self.retriever.get_top_rated(k=k, category=category)
        context = _build_context(products)

        cat_str = f" in {category}" if category else ""
        query = f"Top rated products{cat_str}"
        prompt = RAG_ANSWER_TEMPLATE.format(query=query, context=context)
        answer = self.router.generate(prompt, task="rag_answer", max_tokens=400)

        return {
            "query": query,
            "answer": answer,
            "products": products,
            "provider": "Ollama (llama3)",
        }

    def provider_status(self) -> dict:
        """Return availability of all LLM providers."""
        return self.router.provider_status()


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_pipeline(
    text_index_path: str = "indexes/text_index.faiss",
    text_meta_path: str = "indexes/metadata.pkl",
    image_index_path: str = "indexes/image_index.faiss",
    image_meta_path: str = "indexes/image_metadata.pkl",
    ollama_model: str = None,
    groq_api_key: str = None,
    ollama_url: str = None,
) -> RAGPipeline:
    """
    Convenience factory that wires up VectorStore → RetailRetriever → RAGPipeline.

    Reads configuration from environment variables if not passed explicitly.
    """
    from image_embeddings import CLIPImageEmbedder
    from text_embeddings import TextEmbedder

    ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3")
    groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
    ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")

    store = VectorStore(
        text_index_path=Path(text_index_path),
        text_meta_path=Path(text_meta_path),
        image_index_path=Path(image_index_path),
        image_meta_path=Path(image_meta_path),
    )
    store.load_all()

    text_embedder = TextEmbedder()
    clip_embedder = CLIPImageEmbedder()
    retriever = RetailRetriever(store, text_embedder, clip_embedder)

    router = LLMRouter(
        ollama_model=ollama_model,
        groq_api_key=groq_api_key,
        ollama_url=ollama_url,
    )

    return RAGPipeline(retriever=retriever, llm_router=router)
