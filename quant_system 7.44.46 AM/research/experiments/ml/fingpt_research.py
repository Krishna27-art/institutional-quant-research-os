"""
FinGPT Research Summarizer
===========================
Wraps FinGPT (financial LLM) for research-grade financial text tasks.

⚠  USAGE GUARD: Research summarization ONLY.
   Do NOT use FinGPT output as trading signals.
   LLMs may hallucinate. All outputs require human review.

Suitable for:
  - Earnings call Q&A extraction
  - SEC filing risk factor summarization
  - Financial news narrative analysis
  - Research report summarization

Models (in order of resource requirements):
  - fingpt-sentiment (lightweight, ~7B LoRA on LLaMA2) ← Default
  - fingpt-mt (multi-task, larger)

⚡ Requires significant GPU RAM for 7B models. CPU inference is very slow.
   If no GPU: falls back to FinBERT for sentiment-only tasks.

Reference: https://huggingface.co/FinGPT
"""

import logging
import time
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RESEARCH_ONLY = True  # NEVER remove this flag

# ── Guard imports ─────────────────────────────────────────────────────────────
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline as hf_pipeline,
        BitsAndBytesConfig,
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
    logger.info("transformers loaded for FinGPT.")
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    hf_pipeline = None
    BitsAndBytesConfig = None
    torch = None
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. Run: pip install transformers>=4.40.0")

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PeftModel = None
    PEFT_AVAILABLE = False
    logger.info("peft not installed — LoRA models unavailable. Run: pip install peft")


# ── Model registry ────────────────────────────────────────────────────────────
FINGPT_MODELS = {
    # Lightweight sentiment classification (preferred for research pipelines)
    "sentiment": {
        "base_model":  "meta-llama/Llama-2-7b-hf",
        "peft_model":  "FinGPT/fingpt-sentiment_llama2-7b_lora",
        "task":        "sentiment",
        "gpu_gb_req":  14,    # ~14GB VRAM for 7B FP16
    },
    # Multi-task (summarization + sentiment + forecasting)
    "multitask": {
        "base_model":  "meta-llama/Llama-2-13b-hf",
        "peft_model":  "FinGPT/fingpt-mt_llama2-13b_lora",
        "task":        "multitask",
        "gpu_gb_req":  26,
    },
}

# Lighter alternative: use FinBERT for sentiment when GPU not available
FINBERT_FALLBACK = "ProsusAI/finbert"


class FinGPTResearch:
    """
    FinGPT wrapper for financial research summarization.

    Automatically selects the best available backend:
    ─────────────────────────────────────────────────
    1. FinGPT (7B or 13B) with LoRA — best quality, needs GPU
    2. FinBERT pipeline — lightweight, CPU-friendly
    3. Rule-based fallback — always available

    Example
    -------
    >>> fingpt = FinGPTResearch(model_name="sentiment")
    >>> filing_text = "The company faces significant liquidity risks..."
    >>> summary = fingpt.summarize_risk_factors(filing_text)
    >>> print(summary['summary'])
    >>> print(summary['sentiment'])
    """

    is_research_only: bool = RESEARCH_ONLY

    def __init__(
        self,
        model_name: str = "sentiment",
        use_4bit: bool = True,
        device: Optional[str] = None,
        fallback_to_finbert: bool = True,
        hf_token: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model_name : str
            Key from FINGPT_MODELS ('sentiment' or 'multitask').
        use_4bit : bool
            Use bitsandbytes 4-bit quantization to reduce VRAM (7B → ~4GB).
        device : str, optional
            'cuda', 'cpu', or None (auto-detect).
        fallback_to_finbert : bool
            If FinGPT fails to load (no GPU), use FinBERT instead.
        hf_token : str, optional
            HuggingFace token for gated models (LLaMA requires approval).
        """
        self.model_name  = model_name
        self.use_4bit    = use_4bit and TRANSFORMERS_AVAILABLE
        self.hf_token    = hf_token
        self.fallback    = fallback_to_finbert
        self._device     = self._detect_device(device)
        self._model_cfg  = FINGPT_MODELS.get(model_name, FINGPT_MODELS["sentiment"])

        self._model      = None
        self._tokenizer  = None
        self._finbert    = None  # Fallback pipeline
        self._backend    = None  # 'fingpt' | 'finbert' | 'fallback'

        logger.info(
            f"FinGPTResearch configured | model={model_name} | "
            f"4bit={use_4bit} | device={self._device} | "
            f"fallback_finbert={fallback_to_finbert}"
        )

    @staticmethod
    def _detect_device(requested: Optional[str]) -> str:
        if requested:
            return requested
        if TRANSFORMERS_AVAILABLE and torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _has_enough_gpu(self) -> bool:
        """Check if GPU has enough VRAM for the selected model."""
        if not TRANSFORMERS_AVAILABLE or torch is None:
            return False
        if not torch.cuda.is_available():
            return False
        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            required = self._model_cfg["gpu_gb_req"]
            if self.use_4bit:
                required = required * 0.35  # 4-bit reduces by ~65%
            return vram_gb >= required
        except Exception:
            return False

    def _load_model(self) -> bool:
        """Attempt to load FinGPT model. Returns True on success."""
        if self._backend is not None:
            return True
        if not TRANSFORMERS_AVAILABLE:
            self._backend = "fallback"
            return False

        # Check GPU availability
        if self._device == "cpu" or not self._has_enough_gpu():
            logger.warning(
                f"Insufficient GPU for FinGPT {self.model_name}. "
                f"Falling back to FinBERT."
            )
            return self._load_finbert_fallback()

        # Try FinGPT
        if not PEFT_AVAILABLE:
            logger.warning("peft not installed — cannot load LoRA model. "
                           "Run: pip install peft. Falling back to FinBERT.")
            return self._load_finbert_fallback()

        try:
            base_model_id = self._model_cfg["base_model"]
            peft_model_id = self._model_cfg["peft_model"]

            logger.info(f"Loading FinGPT: {peft_model_id} ...")
            t0 = time.time()

            quantization_cfg = None
            if self.use_4bit and BitsAndBytesConfig is not None:
                try:
                    quantization_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                except Exception:
                    quantization_cfg = None

            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id, token=self.hf_token
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                token=self.hf_token,
                quantization_config=quantization_cfg,
                device_map="auto" if self._device == "cuda" else None,
                torch_dtype=torch.float16 if not self.use_4bit else None,
            )
            model = PeftModel.from_pretrained(base, peft_model_id)
            model.eval()

            self._model     = model
            self._tokenizer = tokenizer
            self._backend   = "fingpt"
            logger.info(f"FinGPT loaded in {time.time() - t0:.1f}s")
            return True

        except Exception as e:
            logger.error(f"FinGPT load failed: {e}")
            return self._load_finbert_fallback()

    def _load_finbert_fallback(self) -> bool:
        """Load lightweight FinBERT as fallback."""
        if not TRANSFORMERS_AVAILABLE:
            self._backend = "fallback"
            return False
        try:
            self._finbert = hf_pipeline(
                "text-classification",
                model=FINBERT_FALLBACK,
                top_k=None,
                device=-1,  # CPU
            )
            self._backend = "finbert"
            logger.info("FinBERT fallback loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"FinBERT fallback also failed: {e}")
            self._backend = "fallback"
            return False

    # ── Core Research APIs ────────────────────────────────────────────────────
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Classify financial text sentiment.

        Returns
        -------
        dict with 'sentiment' ('positive'/'negative'/'neutral'),
        'confidence', 'net_score', 'is_research_only'.
        """
        self._load_model()

        if self._backend == "fingpt" and self._model is not None:
            return self._fingpt_sentiment(text)
        elif self._backend == "finbert" and self._finbert is not None:
            return self._finbert_sentiment(text)
        else:
            return self._rule_based_sentiment(text)

    def summarize_risk_factors(
        self, text: str, max_length: int = 150
    ) -> Dict:
        """
        Extract and summarize risk factors from SEC filings or analyst reports.

        Parameters
        ----------
        text : str
            Input financial text (SEC filing section, report paragraph).
        max_length : int
            Maximum summary length in tokens.

        Returns
        -------
        dict with 'summary', 'sentiment', 'key_risks' (list), 'is_research_only'.
        """
        sentiment_result = self.analyze_sentiment(text[:1000])  # First 1000 chars

        if self._backend == "fingpt" and self._model is not None:
            summary = self._fingpt_generate(
                prompt=f"Summarize the key financial risks in this text:\n\n{text[:800]}\n\nRisk Summary:",
                max_new_tokens=max_length,
            )
        else:
            # Fallback: keyword-based extraction
            summary = self._extract_risk_sentences(text)

        return {
            "summary":          summary,
            "sentiment":        sentiment_result.get("sentiment", "neutral"),
            "net_score":        sentiment_result.get("net_score", 0.0),
            "key_risks":        self._extract_risk_keywords(text),
            "is_research_only": RESEARCH_ONLY,
            "backend":          self._backend,
        }

    def analyze_earnings_call(
        self,
        transcript: str,
        questions: Optional[List[str]] = None,
    ) -> Dict:
        """
        Analyze earnings call transcript for sentiment and key themes.

        Parameters
        ----------
        transcript : str
            Full or partial earnings call transcript.
        questions : list of str, optional
            Specific questions to ask about the transcript.

        Returns
        -------
        dict with overall sentiment, guidance tone, key themes.
        """
        # Split into management remarks vs Q&A
        sections = self._split_earnings_sections(transcript)

        # Sentiment per section
        mgmt_sentiment  = self.analyze_sentiment(sections.get("management", transcript[:500]))
        qa_sentiment    = self.analyze_sentiment(sections.get("qa", transcript[-500:]))

        # Overall
        net_overall = (mgmt_sentiment.get("net_score", 0) * 0.6 +
                       qa_sentiment.get("net_score", 0) * 0.4)

        result = {
            "overall_net_sentiment": float(net_overall),
            "management_sentiment":  mgmt_sentiment,
            "qa_sentiment":          qa_sentiment,
            "guidance_bullish":      net_overall > 0.10,
            "guidance_bearish":      net_overall < -0.10,
            "key_themes":            self._extract_key_themes(transcript),
            "backend":               self._backend,
            "is_research_only":      RESEARCH_ONLY,
        }

        if questions and self._backend == "fingpt" and self._model is not None:
            result["qa_answers"] = {}
            for q in questions[:3]:  # Limit to 3 questions
                answer = self._fingpt_generate(
                    prompt=f"Based on this earnings call transcript:\n{transcript[:600]}\n\nQ: {q}\nA:",
                    max_new_tokens=100,
                )
                result["qa_answers"][q] = answer

        return result

    # ── FinGPT Inference ──────────────────────────────────────────────────────
    def _fingpt_sentiment(self, text: str) -> Dict:
        """Run FinGPT sentiment inference."""
        try:
            prompt = (
                "Instruction: What is the sentiment of this financial text? "
                "Options: positive, negative, neutral.\n"
                f"Input: {text[:400]}\n"
                "Answer:"
            )
            response = self._fingpt_generate(prompt, max_new_tokens=5)
            label    = self._parse_sentiment_label(response)
            net      = {"positive": 0.7, "negative": -0.7, "neutral": 0.0}.get(label, 0.0)
            return {
                "sentiment":        label,
                "confidence":       0.75,
                "net_score":        net,
                "model":            self._model_cfg["peft_model"],
                "is_research_only": RESEARCH_ONLY,
            }
        except Exception as e:
            logger.error(f"FinGPT sentiment error: {e}")
            return self._rule_based_sentiment(text)

    def _fingpt_generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Run FinGPT generation."""
        try:
            inputs = self._tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self._device if self._device == "cuda" else "cpu")
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            response = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            return response
        except Exception as e:
            logger.error(f"FinGPT generation error: {e}")
            return ""

    def _finbert_sentiment(self, text: str) -> Dict:
        """FinBERT sentiment inference."""
        try:
            output = self._finbert(text[:512])[0]
            scores = {item["label"].lower(): item["score"] for item in output}
            pos    = scores.get("positive", 0.33)
            neg    = scores.get("negative", 0.33)
            label  = max(scores, key=scores.get)
            return {
                "sentiment":        label,
                "confidence":       max(scores.values()),
                "net_score":        float(pos - neg),
                "model":            FINBERT_FALLBACK,
                "is_research_only": RESEARCH_ONLY,
            }
        except Exception as e:
            logger.error(f"FinBERT inference error: {e}")
            return self._rule_based_sentiment(text)

    # ── Utilities ─────────────────────────────────────────────────────────────
    @staticmethod
    def _rule_based_sentiment(text: str) -> Dict:
        """Simple keyword-based sentiment fallback."""
        text_lower = text.lower()
        pos_words = ["beat", "grew", "strong", "record", "raised guidance",
                     "outperform", "upgrade", "profit", "bullish", "upside"]
        neg_words = ["miss", "decline", "weak", "cut guidance", "downgrade",
                     "loss", "risk", "concern", "headwind", "bearish"]
        pos_score = sum(1 for w in pos_words if w in text_lower)
        neg_score = sum(1 for w in neg_words if w in text_lower)
        total     = pos_score + neg_score + 1
        net       = (pos_score - neg_score) / total
        label     = "positive" if net > 0.1 else "negative" if net < -0.1 else "neutral"
        return {
            "sentiment":        label,
            "confidence":       abs(net),
            "net_score":        float(net),
            "model":            "rule_based_fallback",
            "is_research_only": RESEARCH_ONLY,
        }

    @staticmethod
    def _parse_sentiment_label(response: str) -> str:
        resp_lower = response.lower()
        if "positive" in resp_lower:
            return "positive"
        if "negative" in resp_lower:
            return "negative"
        return "neutral"

    @staticmethod
    def _split_earnings_sections(transcript: str) -> Dict[str, str]:
        """Heuristically split transcript into management and Q&A sections."""
        q_markers = ["question-and-answer", "q&a session", "operator:", "q:"]
        split_idx = len(transcript) // 2
        for marker in q_markers:
            idx = transcript.lower().find(marker)
            if idx > 0:
                split_idx = idx
                break
        return {
            "management": transcript[:split_idx],
            "qa":         transcript[split_idx:],
        }

    @staticmethod
    def _extract_risk_sentences(text: str) -> str:
        """Extract sentences containing risk keywords."""
        import re
        sentences = re.split(r"[.!?]", text)
        risk_words = ["risk", "uncertainty", "decline", "headwind", "challenge",
                      "exposure", "volatility", "concern"]
        risk_sents = [s.strip() for s in sentences
                      if any(w in s.lower() for w in risk_words)]
        return ". ".join(risk_sents[:5]) + "."

    @staticmethod
    def _extract_risk_keywords(text: str) -> List[str]:
        """Extract risk-related keywords from text."""
        risk_keywords = [
            "liquidity risk", "credit risk", "market risk", "regulatory risk",
            "interest rate risk", "currency risk", "competition", "inflation",
            "supply chain", "geopolitical", "litigation",
        ]
        text_lower = text.lower()
        return [kw for kw in risk_keywords if kw in text_lower]

    @staticmethod
    def _extract_key_themes(text: str) -> List[str]:
        """Extract key financial themes mentioned."""
        themes = [
            "revenue growth", "margin expansion", "guidance", "capex",
            "free cash flow", "debt reduction", "buyback", "dividend",
            "new products", "market share", "cost optimization",
        ]
        text_lower = text.lower()
        return [t for t in themes if t in text_lower]

    def unload(self) -> None:
        """Free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if self._finbert is not None:
            del self._finbert
            self._finbert = None
        if TRANSFORMERS_AVAILABLE and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._backend = None
        logger.info("FinGPT/FinBERT model unloaded.")


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== FinGPT Research Smoke Test ===")
    print(f"transformers available: {TRANSFORMERS_AVAILABLE}")
    print(f"peft available:         {PEFT_AVAILABLE}")

    fingpt = FinGPTResearch(model_name="sentiment", fallback_to_finbert=True)

    texts = [
        "HDFC Bank reports record profit, NPA ratio improves to 1.1%.",
        "Adani Ports faces headwinds from slowdown in commodity exports.",
        "Infosys raises FY25 revenue guidance to 4-7% in constant currency.",
    ]
    print("\nSentiment Analysis:")
    for text in texts:
        result = fingpt.analyze_sentiment(text)
        print(f"  [{result['sentiment'].upper():<8}] {result['net_score']:+.2f}  | {text[:60]}...")

    risk_text = """
    The company faces significant liquidity risk due to high debt levels.
    Rising interest rates present a material headwind to profitability.
    Regulatory changes in the telecom sector create uncertainty for our projections.
    """
    risk_summary = fingpt.summarize_risk_factors(risk_text)
    print(f"\nRisk Summary: {risk_summary['key_risks']}")
    print(f"Sentiment:    {risk_summary['sentiment']}")
    print(f"Backend used: {risk_summary['backend']}")
    print("\n✓ FinGPT smoke test complete.")
