"""
FinBERT Sentiment Analyzer
===========================
Wraps ProsusAI/finbert for financial text sentiment analysis.

⚠  USAGE GUARD: Sentiment signals are RESEARCH INPUTS only.
   Do NOT use raw FinBERT output as direct trading signals.
   Always aggregate, validate, and combine with other signals.

Suitable for:
  - Earnings call transcript sentiment (quarterly)
  - Financial news headline sentiment
  - SEC filing tone analysis
  - Event impact direction estimation

Model: ProsusAI/finbert (~438MB)
       ahmedrachid/FinancialBERT-Sentiment-Analysis (alternative)

Reference: https://huggingface.co/ProsusAI/finbert
Paper: "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RESEARCH_ONLY = True

# ── Guard transformers import ─────────────────────────────────────────────────
try:
    from transformers import (
        BertTokenizer,
        BertForSequenceClassification,
        pipeline as hf_pipeline,
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
    logger.info("transformers loaded — FinBERT backend available.")
except ImportError:
    BertTokenizer = None
    BertForSequenceClassification = None
    hf_pipeline = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    torch = None
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers not installed. "
        "Run: pip install transformers>=4.40.0 torch>=2.0.0"
    )

# ── Sentiment label constants ─────────────────────────────────────────────────
POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL  = "neutral"

FINBERT_MODEL_IDS = {
    "finbert":     "ProsusAI/finbert",
    "financial":   "ahmedrachid/FinancialBERT-Sentiment-Analysis",
    "distilfinbert": "nickmuchi/distilroberta-finetuned-financial-text-classification",
}


class FinBERTSentiment:
    """
    Financial text sentiment analyzer using FinBERT.

    Provides both per-document and aggregated sentiment scoring
    suitable for use as a research input signal.

    Example
    -------
    >>> finbert = FinBERTSentiment()
    >>> headlines = [
    ...     "TCS beats Q3 estimates, revenue up 12%",
    ...     "Reliance faces antitrust probe in telecom sector",
    ...     "HDFC Bank reports stable NPA ratio"
    ... ]
    >>> scores = finbert.analyze_batch(headlines)
    >>> agg = finbert.aggregate_sentiment(scores)
    >>> print(agg['net_sentiment'])  # Range: -1 to +1
    """

    is_research_only: bool = RESEARCH_ONLY

    def __init__(
        self,
        model_name: str = "finbert",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
        cache_dir: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model_name : str
            Key from FINBERT_MODEL_IDS or full HuggingFace model path.
        device : str, optional
            'cuda', 'cpu', or None (auto-detect).
        max_length : int
            Maximum tokenizer length (FinBERT max: 512).
        batch_size : int
            Batch size for inference.
        """
        self.model_id = FINBERT_MODEL_IDS.get(model_name, model_name)
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        self._pipe = None      # HuggingFace pipeline (lazy-loaded)
        self._device = self._detect_device(device)

        logger.info(
            f"FinBERTSentiment configured | model={self.model_id} | "
            f"device={self._device} | batch_size={batch_size}"
        )

    @staticmethod
    def _detect_device(requested: Optional[str]) -> int:
        """Returns device index for HF pipeline (-1=CPU, 0=GPU)."""
        if requested == "cpu":
            return -1
        if requested == "cuda":
            return 0
        if TRANSFORMERS_AVAILABLE and torch is not None and torch.cuda.is_available():
            return 0
        return -1

    def _load_model(self) -> bool:
        """Lazy-load FinBERT pipeline. Returns True on success."""
        if self._pipe is not None:
            return True
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers package not available.")
            return False

        logger.info(f"Loading FinBERT from {self.model_id}...")
        t0 = time.time()
        try:
            self._pipe = hf_pipeline(
                "text-classification",
                model=self.model_id,
                tokenizer=self.model_id,
                device=self._device,
                top_k=None,           # Return all class probabilities
                truncation=True,
                max_length=self.max_length,
                batch_size=self.batch_size,
                cache_dir=self.cache_dir,
            )
            logger.info(f"FinBERT loaded in {time.time() - t0:.1f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            self._pipe = None
            return False

    # ── Core Analysis API ─────────────────────────────────────────────────────
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text snippet.

        Returns
        -------
        dict with:
            'positive', 'negative', 'neutral' : probability scores (0–1)
            'label' : dominant label
            'confidence' : max probability
            'net_score' : positive - negative (-1 to +1)
            'is_research_only' : True
        """
        results = self.analyze_batch([text])
        return results[0] if results else self._neutral_result(text)

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze a batch of texts.

        Parameters
        ----------
        texts : list of str
            List of financial text snippets (headlines, paragraphs).

        Returns
        -------
        list of dicts (same structure as analyze_text).
        """
        if not texts:
            return []

        if not self._load_model():
            logger.warning("FinBERT unavailable — returning neutral scores.")
            return [self._neutral_result(t) for t in texts]

        # Clean texts
        cleaned = [self._clean_text(t) for t in texts]

        try:
            raw_outputs = self._pipe(cleaned)
            results = []
            for text, output in zip(texts, raw_outputs):
                result = self._parse_output(text, output)
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"FinBERT inference error: {e}")
            return [self._neutral_result(t) for t in texts]

    def analyze_earnings_call(
        self,
        transcript: str,
        chunk_size: int = 400,
        overlap: int = 50,
    ) -> Dict:
        """
        Analyze full earnings call transcript by chunking into passages.

        Returns aggregated sentiment + per-chunk breakdown.

        Parameters
        ----------
        transcript : str
            Full transcript text.
        chunk_size : int
            Characters per chunk.
        overlap : int
            Overlap between chunks.
        """
        chunks = self._chunk_text(transcript, chunk_size, overlap)
        per_chunk = self.analyze_batch(chunks)
        aggregated = self.aggregate_sentiment(per_chunk)

        return {
            "aggregated":   aggregated,
            "per_chunk":    per_chunk,
            "n_chunks":     len(chunks),
            "is_research_only": RESEARCH_ONLY,
        }

    def aggregate_sentiment(
        self,
        sentiment_list: List[Dict],
        weights: Optional[List[float]] = None,
    ) -> Dict:
        """
        Aggregate a list of sentiment results into a single composite score.

        Parameters
        ----------
        sentiment_list : list of dicts (from analyze_batch)
        weights : list of floats, optional
            Per-item weights (e.g., recency weights).

        Returns
        -------
        dict with:
            'net_sentiment': float [-1, +1], positive = bullish
            'positive_pct', 'negative_pct', 'neutral_pct': proportions
            'avg_positive', 'avg_negative', 'avg_neutral': mean probs
            'bullish_signal', 'bearish_signal': bool flags
        """
        if not sentiment_list:
            return {"net_sentiment": 0.0, "positive_pct": 0.0,
                    "negative_pct": 0.0, "neutral_pct": 1.0}

        w = np.array(weights) if weights else np.ones(len(sentiment_list))
        w = w / w.sum()

        pos_probs  = np.array([s.get("positive", 0.0) for s in sentiment_list])
        neg_probs  = np.array([s.get("negative", 0.0) for s in sentiment_list])
        neut_probs = np.array([s.get("neutral", 0.0)  for s in sentiment_list])

        labels = [s.get("label", NEUTRAL) for s in sentiment_list]
        pos_pct  = sum(1 for l in labels if l == POSITIVE) / len(labels)
        neg_pct  = sum(1 for l in labels if l == NEGATIVE) / len(labels)
        neut_pct = sum(1 for l in labels if l == NEUTRAL) / len(labels)

        net = float(np.dot(w, pos_probs - neg_probs))

        return {
            "net_sentiment":    net,              # -1 to +1
            "positive_pct":     pos_pct,          # Fraction of positive docs
            "negative_pct":     neg_pct,
            "neutral_pct":      neut_pct,
            "avg_positive":     float(np.dot(w, pos_probs)),
            "avg_negative":     float(np.dot(w, neg_probs)),
            "avg_neutral":      float(np.dot(w, neut_probs)),
            "bullish_signal":   net > 0.15,       # Threshold for research signal
            "bearish_signal":   net < -0.15,
            "n_texts":          len(sentiment_list),
            "is_research_only": RESEARCH_ONLY,
        }

    # ── Utilities ─────────────────────────────────────────────────────────────
    def _parse_output(self, text: str, output: List[Dict]) -> Dict:
        """Parse HuggingFace pipeline output into standardised dict."""
        scores = {item["label"].lower(): item["score"] for item in output}
        pos  = scores.get("positive", scores.get("pos", 0.0))
        neg  = scores.get("negative", scores.get("neg", 0.0))
        neut = scores.get("neutral",  scores.get("neu", 1 - pos - neg))

        dominant = max(
            [(POSITIVE, pos), (NEGATIVE, neg), (NEUTRAL, neut)],
            key=lambda x: x[1]
        )[0]
        confidence = max(pos, neg, neut)

        return {
            "text":             text[:100] + "..." if len(text) > 100 else text,
            "positive":         float(pos),
            "negative":         float(neg),
            "neutral":          float(neut),
            "label":            dominant,
            "confidence":       float(confidence),
            "net_score":        float(pos - neg),
            "model":            self.model_id,
            "is_research_only": RESEARCH_ONLY,
        }

    def _neutral_result(self, text: str) -> Dict:
        return {
            "text":             text[:100] if text else "",
            "positive":         0.33,
            "negative":         0.33,
            "neutral":          0.34,
            "label":            NEUTRAL,
            "confidence":       0.34,
            "net_score":        0.0,
            "model":            "fallback_neutral",
            "is_research_only": RESEARCH_ONLY,
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        """Basic text cleaning for financial text."""
        import re
        text = re.sub(r"\s+", " ", text.strip())
        text = text[:2000]  # Hard cap before tokenization
        return text

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return [c for c in chunks if len(c.strip()) > 20]

    def unload(self) -> None:
        """Free memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("FinBERT pipeline unloaded.")


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== FinBERT Sentiment Smoke Test ===")
    print(f"transformers available: {TRANSFORMERS_AVAILABLE}")

    sample_headlines = [
        "TCS beats Q3 earnings estimates, raises full-year guidance",
        "SEBI probes Adani Group over alleged financial irregularities",
        "HDFC Bank NPA ratio stable at 1.1%, deposits grow 18% YoY",
        "India inflation rises to 6.8%, above RBI comfort band",
        "Infosys wins $2B multi-year deal with European bank",
    ]

    finbert = FinBERTSentiment(model_name="finbert")
    results = finbert.analyze_batch(sample_headlines)
    aggregated = finbert.aggregate_sentiment(results)

    print("\nPer-Headline Sentiment:")
    for r in results:
        label = r["label"].upper()
        net   = r["net_score"]
        print(f"  [{label:<8}] net={net:+.3f}  | {r['text']}")

    print(f"\nAggregated Sentiment:")
    print(f"  Net Score:    {aggregated['net_sentiment']:+.3f}")
    print(f"  Bullish:      {aggregated['bullish_signal']}")
    print(f"  Bearish:      {aggregated['bearish_signal']}")
    print(f"  +/−/= pct:    {aggregated['positive_pct']:.0%} / "
          f"{aggregated['negative_pct']:.0%} / {aggregated['neutral_pct']:.0%}")
    print("\n✓ FinBERT smoke test complete.")
