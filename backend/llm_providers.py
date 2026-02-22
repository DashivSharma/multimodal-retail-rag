"""
LLM Provider Router — manages three distinct LLM backends:

  1. Ollama (llama3)         — local inference, zero cost, used for main RAG Q&A
  2. HuggingFace (flan-t5)  — local transformers model, used for structured
                               feature extraction and product summarization
  3. Groq (llama3-8b-8192)  — cloud inference, ultra-fast, used for complex
                               multi-product comparison and analysis tasks

Each provider exposes a consistent `generate(prompt) -> str` interface.
The RAGPipeline selects the best provider per task type automatically.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import requests
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseLLMProvider(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate a text response for the given prompt."""

    def is_available(self) -> bool:
        """Return True if this provider is reachable / configured."""
        return True


# ---------------------------------------------------------------------------
# Provider 1 — Ollama (llama3, local)
# ---------------------------------------------------------------------------

class OllamaProvider(BaseLLMProvider):
    """
    Calls a locally running Ollama server.

    Best for: Main conversational RAG answers — product Q&A, recommendations,
    price comparisons. No API cost, runs fully offline.

    Requires: `ollama serve` running with llama3 pulled.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/generate"

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Send a prompt to Ollama and return the full response text.
        Uses the non-streaming endpoint for simplicity.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.3,
                "top_p": 0.9,
            },
        }
        try:
            resp = requests.post(self.api_url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            logger.error("Ollama server not reachable at %s", self.base_url)
            raise RuntimeError(
                "Ollama is not running. Start it with: ollama serve"
            )
        except Exception as e:
            logger.error("Ollama generation failed: %s", e)
            raise

    def stream_generate(self, prompt: str, max_tokens: int = 512):
        """
        Generator that yields text chunks from Ollama's streaming API.
        Used by the Streamlit frontend for real-time output.
        """
        import json

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": 0.3},
        }
        try:
            with requests.post(self.api_url, json=payload, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
        except Exception as e:
            logger.error("Ollama streaming failed: %s", e)
            yield f"\n[Error: {e}]"


# ---------------------------------------------------------------------------
# Provider 2 — HuggingFace Transformers (flan-t5-base, local)
# ---------------------------------------------------------------------------

class HuggingFaceProvider(BaseLLMProvider):
    """
    Runs google/flan-t5-base locally via HuggingFace Transformers.

    Best for: Structured product feature extraction, concise spec summaries,
    and "explain this product" tasks. Flan-T5 excels at instruction-following
    for short, structured outputs without needing a large model.

    The model is lazy-loaded on first use to avoid startup overhead.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSeq2SeqLM] = None

    def _load(self):
        if self._model is None:
            logger.info("Loading HuggingFace model: %s on %s", self.model_name, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            logger.info("HuggingFace model loaded.")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """
        Run inference with flan-t5. Truncates long inputs to model max length.
        """
        self._load()
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        result = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()

    def extract_features(self, product_text: str) -> str:
        """
        Convenience method: extract key product features as a bullet list.
        Wraps the product text in a structured extraction prompt.
        """
        prompt = (
            f"Extract the key features of this product as a short bullet list:\n\n"
            f"{product_text[:400]}\n\nKey features:"
        )
        return self.generate(prompt, max_tokens=150)

    def summarize_product(self, product_text: str) -> str:
        """
        Convenience method: generate a one-sentence product summary.
        """
        prompt = (
            f"Summarize this product in one sentence for a customer:\n\n"
            f"{product_text[:400]}\n\nSummary:"
        )
        return self.generate(prompt, max_tokens=80)


# ---------------------------------------------------------------------------
# Provider 3 — Groq (llama3-8b-8192, cloud)
# ---------------------------------------------------------------------------

class GroqProvider(BaseLLMProvider):
    """
    Calls the Groq cloud API using the openai-compatible endpoint.

    Best for: Multi-product comparison, complex analytical queries, and
    any task requiring fast processing of large context windows. Groq's
    LPU inference is 10-20x faster than typical cloud APIs, making it
    ideal for comparison tables and detailed analysis.

    Requires: GROQ_API_KEY environment variable.
    """

    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        model: str = "llama3-8b-8192",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Call the Groq chat completions API.
        Uses a system prompt tuned for retail product analysis.
        """
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert retail product analyst. "
                        "Provide clear, structured, and helpful product comparisons and analysis. "
                        "Use tables when comparing multiple products. Be concise and factual."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "top_p": 0.9,
        }

        try:
            resp = requests.post(self.GROQ_API_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            logger.error("Groq API HTTP error: %s — %s", e, resp.text)
            raise RuntimeError(f"Groq API error: {resp.status_code} — {resp.text}")
        except Exception as e:
            logger.error("Groq generation failed: %s", e)
            raise

    def compare_products(self, products: list[dict], user_query: str) -> str:
        """
        Convenience method: generate a structured comparison table for multiple products.
        Sends all product context to Groq for fast multi-doc analysis.
        """
        product_lines = []
        for i, p in enumerate(products, 1):
            name = p.get("product_name", "Unknown")
            brand = p.get("brand", "N/A")
            price = p.get("discounted_price", "N/A")
            rating = p.get("product_rating", "N/A")
            desc = str(p.get("description", ""))[:200]
            product_lines.append(
                f"Product {i}: {name}\n"
                f"  Brand: {brand} | Price: Rs.{price} | Rating: {rating}\n"
                f"  Description: {desc}"
            )

        context = "\n\n".join(product_lines)
        prompt = (
            f"User query: {user_query}\n\n"
            f"Compare the following products and answer the user's query:\n\n"
            f"{context}\n\n"
            f"Provide a detailed comparison table and recommendation."
        )
        return self.generate(prompt, max_tokens=1024)


# ---------------------------------------------------------------------------
# Provider factory / router
# ---------------------------------------------------------------------------

class LLMRouter:
    """
    Routes LLM requests to the appropriate provider based on task type.

    Task routing:
      - "rag_answer"    → Ollama (llama3)  — conversational product Q&A
      - "extraction"    → HuggingFace      — feature extraction / summarization
      - "comparison"    → Groq             — multi-product comparison tables
      - "analysis"      → Groq             — complex analytical queries
      - "fallback"      → tries each in order until one succeeds
    """

    TASK_PROVIDER_MAP = {
        "rag_answer": "ollama",
        "extraction": "huggingface",
        "summarize": "huggingface",
        "comparison": "groq",
        "analysis": "groq",
    }

    def __init__(
        self,
        ollama_model: str = "llama3",
        hf_model: str = "google/flan-t5-base",
        groq_model: str = "llama3-8b-8192",
        groq_api_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
    ):
        self.providers: dict[str, BaseLLMProvider] = {
            "ollama": OllamaProvider(model=ollama_model, base_url=ollama_url),
            "huggingface": HuggingFaceProvider(model_name=hf_model),
            "groq": GroqProvider(model=groq_model, api_key=groq_api_key),
        }

    def get_provider(self, task: str) -> BaseLLMProvider:
        """Return the best provider for the given task type."""
        provider_name = self.TASK_PROVIDER_MAP.get(task, "ollama")
        provider = self.providers[provider_name]

        # Graceful fallback if primary provider is unavailable
        if not provider.is_available():
            logger.warning(
                "Provider '%s' unavailable for task '%s'. Trying fallback.",
                provider_name, task,
            )
            for name, fallback in self.providers.items():
                if name != provider_name and fallback.is_available():
                    logger.info("Falling back to provider: %s", name)
                    return fallback
            raise RuntimeError("No LLM providers are available.")

        return provider

    def generate(self, prompt: str, task: str = "rag_answer", max_tokens: int = 512) -> str:
        """Route a prompt to the appropriate provider and return the response."""
        provider = self.get_provider(task)
        logger.info("Routing task='%s' to provider=%s", task, type(provider).__name__)
        return provider.generate(prompt, max_tokens=max_tokens)

    def provider_status(self) -> dict[str, bool]:
        """Return availability status for all providers."""
        return {name: p.is_available() for name, p in self.providers.items()}
