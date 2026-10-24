
import os
import logging
from typing import Dict, Any

logger = logging.getLogger("FactGuard.BertVerifier")

# Cache directory inside backend/models
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "models",
    "bert_cache",
)

from config import HUGGINGFACE_TEXT_MODEL, HUGGINGFACE_API_TOKEN

# Preferred high-accuracy models in order of precision and calibration
_CANDIDATE_MODELS = [
    HUGGINGFACE_TEXT_MODEL,
    "jy46604790/Fake-News-Bert-Detect",
    "hamzab/roberta-fake-news-classification",
]


class BertTextVerifier:
    """Wraps Hugging Face pipeline for text-based fake news classification."""

    def __init__(self):
        self.pipeline = None
        self.available = False
        self.model_name = ""
        self._load()

    def _load(self):
        """Attempt to load transformer pipeline. Tries RoBERTa first, falls back to BERT."""
        try:
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            import torch  # noqa: F401
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

            os.makedirs(_CACHE_DIR, exist_ok=True)

            loaded = False
            for model_id in _CANDIDATE_MODELS:
                try:
                    tokenizer = None
                    model = None
                    for local_flag in (True, False):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(
                                model_id,
                                cache_dir=_CACHE_DIR,
                                local_files_only=local_flag,
                            )
                            model = AutoModelForSequenceClassification.from_pretrained(
                                model_id,
                                cache_dir=_CACHE_DIR,
                                local_files_only=local_flag,
                            )
                            if tokenizer is not None and model is not None:
                                break
                        except Exception:
                            if local_flag:
                                continue
                            raise
                    self.pipeline = pipeline(
                        "text-classification",
                        model=model,
                        tokenizer=tokenizer,
                        truncation=True,
                        max_length=512,
                        device=-1,  # CPU execution
                    )
                    self.available = True
                    self.model_name = model_id
                    logger.info("Transformer text classifier '%s' initialized and ready.", model_id)
                    loaded = True
                    break
                except Exception as load_err:
                    logger.warning("Could not load candidate model '%s': %s", model_id, load_err)

            if not loaded:
                self.available = False
                logger.warning("No candidate transformer models could be loaded. Using TF-IDF fallback.")

        except ImportError as e:
            logger.info("PyTorch / Transformers not installed (%s). Transformer disabled; using TF-IDF fallback.", e)
            self.available = False
        except Exception as e:
            logger.warning("Transformer initialization error (%s). Using TF-IDF fallback.", e)
            self.available = False

    def _predict_hf_api(self, text: str):
        """Attempt inference via Hugging Face Inference API if HUGGINGFACE_API_TOKEN is configured."""
        if not HUGGINGFACE_API_TOKEN:
            return None
        try:
            import requests
            model_target = self.model_name or HUGGINGFACE_TEXT_MODEL
            url = f"https://router.huggingface.co/hf-inference/models/{model_target}"
            headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
            payload = {"inputs": text[:1500]}
            resp = requests.post(url, headers=headers, json=payload, timeout=2.5)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    entries = data[0] if isinstance(data[0], list) else data
                    if isinstance(entries, list) and len(entries) > 0 and isinstance(entries[0], dict):
                        best = max(entries, key=lambda x: x.get("score", 0.0))
                        raw_label = str(best.get("label", "")).upper()
                        score = float(best.get("score", 0.0))
                        score = max(0.0, min(1.0, score))
                        label = _normalise_label(raw_label, score)
                        if label == "FAKE":
                            probs = {"FAKE": score, "REAL": float(1.0 - score)}
                        else:
                            probs = {"REAL": score, "FAKE": float(1.0 - score)}
                        logger.info("Inference succeeded via Hugging Face Serverless API (%s).", model_target)
                        return {
                            "label": label,
                            "confidence": round(score, 4),
                            "probabilities": probs,
                            "model_name": model_target,
                            "method": "huggingface_api",
                            "is_free_huggingface_model": True,
                            "hf_engine_status": "Hugging Face Cloud API answered successfully",
                        }
            elif resp.status_code == 402:
                logger.info("Hugging Face API returned 402 (Payment/Credits required on HF Serverless). Immediately running free local Hugging Face model.")
            elif resp.status_code == 410:
                logger.info("Model '%s' is deprecated on HF Serverless provider (410). Immediately running free local Hugging Face model.", model_target)
            else:
                logger.warning("Hugging Face API returned status %s (%s). Using free local Hugging Face model.", resp.status_code, resp.text[:100])
        except Exception as e:
            logger.info("Hugging Face Cloud API call skipped (%s); seamlessly using free local Hugging Face model.", e)
        return None

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classify text as FAKE or REAL.
        Priority:
          1. Hugging Face Inference API (if token provided)
          2. Local Free Hugging Face Pipeline (cached, CPU/PyTorch, zero cost)
          3. Returns UNKNOWN for TF-IDF fallback in TextVerifier
        """
        # 1. Try Hugging Face Cloud Inference API if token configured
        api_result = self._predict_hf_api(text)
        if api_result is not None:
            return api_result

        # 2. Local free Hugging Face model (offline cached pipeline)
        if not self.available or self.pipeline is None:
            return {
                "label": "UNKNOWN",
                "confidence": 0.0,
                "probabilities": {},
                "model_name": "",
                "method": "none",
                "is_free_huggingface_model": False,
            }

        try:
            # Allow up to 2500 characters for rich article context before tokenizer truncation
            truncated = text[:2500]
            result = self.pipeline(truncated)[0]

            raw_label = str(result.get("label", "")).upper()
            score = float(result.get("score", 0.0))
            score = max(0.0, min(1.0, score))

            label = _normalise_label(raw_label, score)

            if label == "FAKE":
                probs = {"FAKE": score, "REAL": float(1.0 - score)}
            else:
                probs = {"REAL": score, "FAKE": float(1.0 - score)}

            return {
                "label": label,
                "confidence": round(score, 4),
                "probabilities": probs,
                "model_name": self.model_name,
                "method": "huggingface_local",
                "is_free_huggingface_model": True,
                "hf_engine_status": "Free Hugging Face model answered via local cached pipeline (0 cost, unlimited)",
            }

        except Exception as e:
            logger.error("Transformer inference exception: %s", e)
            return {
                "label": "UNKNOWN",
                "confidence": 0.0,
                "probabilities": {},
                "model_name": self.model_name,
                "method": "none",
                "is_free_huggingface_model": False,
            }


def _normalise_label(raw_label: str, score: float) -> str:
    """Normalize raw Hugging Face model output to FAKE or REAL."""
    mapping = {
        "LABEL_0": "FAKE",   # Fake News Bert: 0 = FAKE
        "LABEL_1": "REAL",   # Fake News Bert: 1 = REAL
        "FAKE": "FAKE",
        "REAL": "REAL",
        "FALSE": "FAKE",
        "TRUE": "REAL",
    }
    return mapping.get(raw_label, "FAKE" if score > 0.5 else "REAL")
