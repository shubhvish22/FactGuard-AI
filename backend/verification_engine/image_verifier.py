
import os
import io
import sqlite3
import logging
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageChops
import imagehash

from config import (
    HASH_DB_PATH,
    PHASH_SIMILARITY_THRESHOLD,
    AI_DETECTION_NOISE_THRESHOLD,
    AI_COMPOSITE_THRESHOLD,
    AI_HIGH_CONFIDENCE_THRESHOLD,
    FLORIDA_MODEL_PATH,
    HUGGINGFACE_IMAGE_MODEL,
    HUGGINGFACE_API_TOKEN,
)

logger = logging.getLogger("FactGuard.ImageVerifier")

# HuggingFace model for AI image detection
_HF_MODEL_NAME = HUGGINGFACE_IMAGE_MODEL or "umm-maybe/AI-image-detector"
_HF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "hf_img_cache")


def _load_hf_image_classifier():
    """Load Hugging Face AI image detector pipeline once; returns pipeline or None."""
    try:
        os.environ["TRANSFORMERS_CACHE"] = _HF_CACHE_DIR
        os.makedirs(_HF_CACHE_DIR, exist_ok=True)

        logger.info("Initializing HuggingFace AI image detector '%s'...", _HF_MODEL_NAME)
        processor = None
        for local_flag in (True, False):
            try:
                try:
                    from transformers import AutoImageProcessor
                    processor = AutoImageProcessor.from_pretrained(_HF_MODEL_NAME, cache_dir=_HF_CACHE_DIR, local_files_only=local_flag)
                except Exception:
                    from transformers import AutoFeatureExtractor
                    processor = AutoFeatureExtractor.from_pretrained(_HF_MODEL_NAME, cache_dir=_HF_CACHE_DIR, local_files_only=local_flag)
                if processor is not None:
                    break
            except Exception:
                continue

        from transformers import AutoModelForImageClassification, pipeline as hf_pipeline
        model = None
        for local_flag in (True, False):
            try:
                model = AutoModelForImageClassification.from_pretrained(
                    _HF_MODEL_NAME, cache_dir=_HF_CACHE_DIR, local_files_only=local_flag
                )
                if model is not None:
                    break
            except Exception:
                continue

        clf = hf_pipeline(
            "image-classification",
            model=model,
            image_processor=processor,
            device=-1,  # CPU execution
        )
        logger.info("HuggingFace AI image detector loaded successfully.")
        return clf
    except ImportError:
        logger.info("Transformers / PyTorch not installed. Using forensic heuristics for AI detection.")
        return None
    except Exception as e:
        logger.warning("Could not load HuggingFace AI model (%s). Using forensic heuristics fallback.", e)
        return None


def _load_florida_classifier():
    """Load calibrated classifier trained on the FLORIDA dataset."""
    if os.path.isfile(FLORIDA_MODEL_PATH):
        try:
            import joblib
            clf = joblib.load(FLORIDA_MODEL_PATH)
            logger.info("FLORIDA-calibrated image classifier loaded successfully from %s.", FLORIDA_MODEL_PATH)
            return clf
        except Exception as e:
            logger.warning("Could not load FLORIDA model (%s).", e)
    return None


class ImageVerifier:
    """Handles image verification: pHash reuse detection, synthetic AI analysis, and tampering detection."""

    def __init__(self):
        self._init_hash_db()
        self._hf_classifier = _load_hf_image_classifier()
        self._florida_classifier = _load_florida_classifier()
        self.model_name = _HF_MODEL_NAME

    def _get_connection(self) -> sqlite3.Connection:
        """Create a thread-safe SQLite connection with a 20s timeout."""
        return sqlite3.connect(HASH_DB_PATH, timeout=20.0)

    def _init_hash_db(self):
        """Initialize SQLite database with WAL mode for safe multi-client concurrency."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_hashes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phash TEXT NOT NULL,
                    source_url TEXT DEFAULT '',
                    label TEXT DEFAULT 'unknown',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_hashes_phash ON image_hashes(phash);")
            conn.commit()
            logger.info("Hash database initialized with WAL mode enabled.")
        except Exception as e:
            logger.error("Error initializing hash database: %s", e)
        finally:
            if conn:
                conn.close()

    def verify(
        self,
        image: Image.Image,
        source_url: str = "",
        webpage_url: str = "",
        webpage_title: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run complete image verification pipeline:
          1. Compute pHash and check database similarity (excluding self-matches).
          2. Run synthetic AI detection (ViT primary, forensics fallback).
          3. Run digital manipulation / edit forensics (multi-quality ELA, noise maps, blocking).
          4. Query web reverse search (Google Vision / SerpApi / Lens).
          5. Synthesize 4-way classification verdict (ai_generated | edited | fake | real).
          6. Record hash into database.
        """
        try:
            # Step 1: Compute Perceptual Hash
            phash = self._compute_phash(image)

            # Step 2: Check for Existing Similar Hashes in DB
            similarity_result = self._check_hash_similarity(phash, source_url)

            # Step 3: Synthetic AI Detection
            ai_result = self._detect_ai_generated(image)

            # Step 4: Manipulation / Tampering Forensics
            edit_result = self._analyze_edit_manipulation(image)

            # Step 5: Web Reverse Image Search
            from utils import search_image_on_web
            web_search = search_image_on_web(image, source_url)

            pages_count = len(web_search.get("pages_with_matching_images", []))
            full_matches = len(web_search.get("full_matching_images", []))
            partial_matches = len(web_search.get("partial_matching_images", []))
            total_web_matches = full_matches + partial_matches

            # Scan matching domains for fact-checking / debunking sites
            debunks = [
                p.get("url", "") for p in web_search.get("pages_with_matching_images", [])
                if any(site in p.get("url", "").lower() for site in ["factcheck", "snopes", "politifact", "hoax", "boomlive", "altnews"])
            ]
            has_debunks = len(debunks) > 0

            ai_confidence = float(ai_result.get("composite_score", 0.0))
            edit_score = float(edit_result.get("edit_score", 0.0))
            ai_method = str(ai_result.get("method", "heuristics"))
            has_api_results = not web_search.get("is_simulated", True) and not web_search.get("api_error")
            context_text = f"{webpage_title} {webpage_url} {source_url}".lower()
            vit_raw = float(ai_result.get("details", {}).get("vit_raw_score", 0.0))
            ela_val = float(edit_result.get("details", {}).get("ela_score", 0.0))

            # Step 6: Multi-Factor Category Classification
            category_result = self._classify_image_category(
                ai_confidence=ai_confidence,
                edit_score=edit_score,
                is_reused=bool(similarity_result.get("is_reused", False)),
                has_debunks=has_debunks,
                has_api_results=has_api_results,
                web_match_count=total_web_matches,
                ai_method=ai_method,
                context_text=context_text,
                vit_raw_score=vit_raw,
                ela_score=ela_val,
            )

            # Step 7: Store Hash (excluding duplicates)
            self._store_hash(str(phash), source_url)

            is_suspicious = category_result["category"] in ("ai_generated", "edited", "fake", "unverified")

            # Step 8: Verification Mode & Hugging Face Free Model Transparency
            if has_api_results:
                verification_mode = "api_hybrid"
            elif ai_method == "huggingface_api":
                verification_mode = "huggingface_free_api"
            elif ai_method in ("huggingface", "florida_calibrated_ml"):
                verification_mode = "huggingface_free_local"
            else:
                verification_mode = "heuristics_local"

            is_free_huggingface = bool(ai_method in ("huggingface", "huggingface_api", "florida_calibrated_ml"))

            # Step 9: OCR Text Extraction & Content Claims Fact-Checking
            from utils import extract_image_text_ocr
            from claim_extractor import extract_claims
            from evidence_retriever import EvidenceRetriever
            from claim_cross_referencer import cross_reference_all_claims

            ocr_text = extract_image_text_ocr(image)
            claims_verdicts = []
            if ocr_text and len(ocr_text.strip()) >= 10:
                extracted_claims = extract_claims(ocr_text)
                retriever = EvidenceRetriever()
                evidence_by_claim = {}
                for cl in extracted_claims:
                    cid = cl.get("claim_id", 1)
                    q = cl.get("search_query") or cl.get("statement", "")
                    evidence_by_claim[cid] = retriever.retrieve_evidence(q, cl.get("statement", ""), max_sources=5)

                claims_verdicts = cross_reference_all_claims(extracted_claims, evidence_by_claim)

            # Record in Persistent History
            try:
                from history_manager import HistoryManager
                hm = HistoryManager()
                ev_sources = []
                for cv in claims_verdicts:
                    ev_sources.extend(cv.get("supporting_evidence", []))
                    ev_sources.extend(cv.get("contradicting_evidence", []))
                for m in web_search.get("pages_with_matching_images", []):
                    if m.get("url"):
                        ev_sources.append({"url": m.get("url"), "title": m.get("pageTitle", "Matching Web Page"), "tier": 4, "tier_label": "Web Match"})

                record_source_url = webpage_url or (source_url if not source_url.startswith("upload:") else "")
                img_entry_id = hm.save_entry(
                    submission_type="image",
                    content_preview=f"Image ({image.width}x{image.height}) - OCR: {ocr_text[:80] if ocr_text else 'No text'}",
                    source_url=record_source_url,
                    page_title=webpage_title or "",
                    direct_image_url=source_url if source_url.startswith(("http://", "https://")) else "",
                    claims=claims_verdicts,
                    overall_verdict=category_result["category"],
                    ml_prediction=category_result["category"],
                    ml_confidence=float(category_result["confidence"]),
                    evidence_sources=ev_sources[:15],
                    source_discovery_status="sources_found" if (ev_sources or total_web_matches > 0) else "insufficient_evidence",
                )
            except Exception as hist_err:
                logger.warning("History save error on image verification: %s", hist_err)
                img_entry_id = None

            return {
                "history_id": img_entry_id,
                "phash": str(phash),
                "is_reused": bool(similarity_result.get("is_reused", False)),
                "similarity_score": float(similarity_result.get("similarity_score", 0.0)),
                "similar_count": int(similarity_result.get("similar_count", 0)),
                "local_matching_urls": similarity_result.get("matching_urls", []),
                "closest_match_distance": int(similarity_result.get("closest_distance", -1)),
                "is_ai_generated": bool(category_result["category"] == "ai_generated" or ai_result.get("is_ai_generated", False)),
                "ai_confidence": round(ai_confidence, 4),
                "ai_details": ai_result.get("details", {}),
                "ai_method": ai_method,
                "verification_mode": verification_mode,
                "is_free_huggingface_model": is_free_huggingface,
                "hf_engine_status": ai_result.get("hf_engine_status", "Free Hugging Face ViT model active via local pipeline (0 cost, unlimited)"),
                "model_name": getattr(self, "model_name", _HF_MODEL_NAME),
                "edit_score": round(edit_score, 4),
                "edit_details": edit_result.get("details", {}),
                "image_category": category_result["category"],
                "category_confidence": round(float(category_result["confidence"]), 4),
                "category_reason": category_result["reason"],
                "image_dimensions": f"{image.width}x{image.height}",
                "web_search": web_search,
                "web_pages_count": pages_count,
                "web_full_matches": full_matches,
                "is_suspicious": is_suspicious,
                "ocr_text": ocr_text,
                "has_image_text": bool(ocr_text),
                "claims": claims_verdicts,
                "visual_forensics": {
                    "visual_category": category_result["category"],
                    "visual_confidence": round(float(category_result["confidence"]), 4),
                    "visual_reason": category_result["reason"],
                    "is_ai_generated": bool(category_result["category"] == "ai_generated" or ai_result.get("is_ai_generated", False)),
                    "ai_confidence": round(ai_confidence, 4),
                    "ai_method": ai_method,
                    "edit_score": round(edit_score, 4),
                    "is_reused": bool(similarity_result.get("is_reused", False)),
                },
            }
        except Exception as e:
            logger.error("Exception during image verification: %s", e, exc_info=True)
            return {
                "error": str(e),
                "phash": "",
                "is_reused": False,
                "similarity_score": 0.0,
                "similar_count": 0,
                "closest_match_distance": -1,
                "is_ai_generated": False,
                "ai_confidence": 0.0,
                "ai_details": {},
                "edit_score": 0.0,
                "edit_details": {},
                "image_category": "real",
                "category_confidence": 0.0,
                "category_reason": "Error during analysis",
                "image_dimensions": "unknown",
                "web_search": {},
                "web_pages_count": 0,
                "web_full_matches": 0,
                "is_suspicious": False,
                "verification_mode": "none",
                "is_free_huggingface_model": False,
                "model_name": getattr(self, "model_name", _HF_MODEL_NAME),
            }

    def _compute_phash(self, image: Image.Image) -> imagehash.ImageHash:
        """Compute perceptual hash (8x8 standard size for 64 bits)."""
        return imagehash.phash(image, hash_size=8)

    def _check_hash_similarity(self, phash: imagehash.ImageHash, current_source_url: str = "") -> Dict[str, Any]:
        """
        Compare current pHash against database of stored hashes.
        Filters out self-matches so re-checking the same URL does not trigger false 'reused' flags.
        """
        conn = None
        rows = []
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT phash, source_url FROM image_hashes")
            rows = cursor.fetchall()
        except Exception as e:
            logger.error("Database read error during similarity check: %s", e)
            return {
                "is_reused": False,
                "similarity_score": 0.0,
                "similar_count": 0,
                "closest_distance": -1,
                "matching_urls": [],
            }
        finally:
            if conn:
                conn.close()

        if not rows:
            return {
                "is_reused": False,
                "similarity_score": 0.0,
                "similar_count": 0,
                "closest_distance": -1,
                "matching_urls": [],
            }

        distances = []
        matching_urls = []

        for row in rows:
            try:
                stored_hash = imagehash.hex_to_hash(row[0])
                stored_url = row[1] or ""

                # Skip self-match if checking the exact same URL
                if current_source_url and stored_url == current_source_url:
                    continue

                distance = phash - stored_hash
                distances.append(distance)

                if distance <= PHASH_SIMILARITY_THRESHOLD:
                    if stored_url and stored_url != current_source_url:
                        matching_urls.append(stored_url)
            except Exception:
                continue

        if not distances:
            return {
                "is_reused": False,
                "similarity_score": 0.0,
                "similar_count": 0,
                "closest_distance": -1,
                "matching_urls": [],
            }

        min_distance = min(distances)
        similar_count = sum(1 for d in distances if d <= PHASH_SIMILARITY_THRESHOLD)

        max_distance = 64  # Standard 64-bit pHash (8x8)
        similarity_score = max(0.0, 1.0 - (min_distance / max_distance))

        return {
            "is_reused": min_distance <= PHASH_SIMILARITY_THRESHOLD,
            "similarity_score": round(float(similarity_score), 4),
            "similar_count": similar_count,
            "closest_distance": int(min_distance),
            "matching_urls": list(set(matching_urls)),
        }

    def _query_hf_image_api(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """Attempt query to Hugging Face Inference API if HUGGINGFACE_API_TOKEN is set."""
        if not HUGGINGFACE_API_TOKEN:
            return None
        try:
            import requests
            img_rgb = image.convert("RGB")
            img_rgb.thumbnail((512, 512), Image.LANCZOS)
            buf = io.BytesIO()
            img_rgb.save(buf, format="JPEG", quality=85)
            img_bytes = buf.getvalue()

            url = f"https://router.huggingface.co/hf-inference/models/{_HF_MODEL_NAME}"
            headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
            resp = requests.post(url, headers=headers, data=img_bytes, timeout=2.5)
            if resp.status_code == 200:
                results = resp.json()
                if isinstance(results, list):
                    ai_score = 0.0
                    for r in results:
                        lbl = str(r.get("label", "")).lower()
                        score = float(r.get("score", 0.0))
                        if any(k in lbl for k in ("artificial", "fake", "ai", "synthetic")):
                            ai_score = score
                            break
                        elif any(k in lbl for k in ("human", "real", "authentic")):
                            ai_score = 1.0 - score
                            break
                    ai_score = max(0.0, min(1.0, ai_score))
                    logger.info("Image classified via Hugging Face Serverless API (score: %.4f).", ai_score)
                    return {
                        "is_ai_generated": bool(ai_score >= AI_COMPOSITE_THRESHOLD),
                        "composite_score": round(ai_score, 4),
                        "method": "huggingface_api",
                        "hf_engine_status": "Hugging Face Cloud API answered successfully",
                        "details": {
                            "model": f"Hugging Face Inference API ({_HF_MODEL_NAME})",
                            "hf_ai_score": round(ai_score, 4),
                            "raw_results": [{"label": r.get("label", ""), "score": round(float(r.get("score", 0.0)), 4)} for r in results],
                        },
                    }
            elif resp.status_code == 402:
                logger.info("Hugging Face Image API returned 402 (Payment/Credits required on HF Serverless). Immediately running local free model.")
            elif resp.status_code == 410:
                logger.info("Model '%s' is deprecated on HF Serverless provider (410). Immediately running local free model.", _HF_MODEL_NAME)
            else:
                logger.warning("Hugging Face Image API returned %s. Using local free model.", resp.status_code)
        except Exception as e:
            logger.info("Hugging Face Image API request skipped (%s); seamlessly using local free model.", e)
        return None

    def _detect_ai_generated(self, image: Image.Image) -> Dict[str, Any]:
        """Detect synthetic AI generation using Hugging Face free models, FLORIDA ensemble, or heuristics."""
        # 1. Try Hugging Face Cloud Inference API if token configured
        api_res = self._query_hf_image_api(image)
        if api_res is not None:
            return api_res

        # 2. FLORIDA Calibrated Ensemble (uses local Hugging Face ViT features + spatial/spectral sensors)
        if self._florida_classifier is not None:
            return self._florida_classify(image)

        # 3. Direct Local Hugging Face ViT Model
        if self._hf_classifier is not None:
            return self._hf_classify(image)

        # 4. Pure Forensic Heuristics fallback
        return self._heuristic_classify(image)

    def _florida_classify(self, image: Image.Image) -> Dict[str, Any]:
        """Classify using multimodal ensemble calibrated on FLORIDA (Fake-Looking Real Images Dataset)."""
        try:
            from train_image_model import extract_features_from_pil
            features = extract_features_from_pil(image, self._hf_classifier)
            if features is not None:
                probs = self._florida_classifier.predict_proba([features])[0]
                # prob[0] is REAL (FLORIDA), prob[1] is AI-GENERATED
                p_ai = float(probs[1])
                vit_raw = float(features[0]) if len(features) > 0 else 0.0

                # Multi-model consensus: Don't let tabular classifier suppress strong neural ViT suspicion
                if vit_raw >= 0.40:
                    p_ai = max(p_ai, vit_raw * 0.90)
                else:
                    p_ai = 0.5 * p_ai + 0.5 * vit_raw

                p_ai = max(0.0, min(1.0, p_ai))
                is_ai = p_ai >= AI_COMPOSITE_THRESHOLD

                return {
                    "is_ai_generated": is_ai,
                    "composite_score": round(p_ai, 4),
                    "method": "florida_calibrated_ml",
                    "details": {
                        "model": "FLORIDA Calibrated Ensemble (RandomForest + ViT + Forensics)",
                        "florida_ai_probability": round(float(probs[1]), 4),
                        "florida_real_probability": round(float(probs[0]), 4),
                        "vit_raw_score": round(vit_raw, 4),
                    },
                }
        except Exception as e:
            logger.warning("FLORIDA classifier inference failed (%s). Falling back to ViT.", e)

        if self._hf_classifier is not None:
            return self._hf_classify(image)
        return self._heuristic_classify(image)

    def _hf_classify(self, image: Image.Image) -> Dict[str, Any]:
        """Deep learning AI image classifier (umm-maybe/AI-image-detector)."""
        try:
            img_rgb = image.convert("RGB")
            img_rgb.thumbnail((512, 512), Image.LANCZOS)

            results = self._hf_classifier(img_rgb)
            ai_score = 0.0

            for r in results:
                lbl = r.get("label", "").lower()
                score = float(r.get("score", 0.0))
                if any(k in lbl for k in ("artificial", "fake", "ai", "synthetic")):
                    ai_score = score
                    break
                elif any(k in lbl for k in ("human", "real", "authentic")):
                    ai_score = 1.0 - score
                    break

            ai_score = max(0.0, min(1.0, ai_score))
            is_ai = ai_score >= AI_COMPOSITE_THRESHOLD

            return {
                "is_ai_generated": is_ai,
                "composite_score": round(ai_score, 4),
                "method": "huggingface",
                "details": {
                    "hf_model": _HF_MODEL_NAME,
                    "hf_ai_score": round(ai_score, 4),
                    "raw_results": [{"label": r["label"], "score": round(float(r["score"]), 4)} for r in results],
                },
            }
        except Exception as e:
            logger.warning("HuggingFace ViT inference error: %s. Falling back to heuristics.", e)
            return self._heuristic_classify(image)

    def _heuristic_classify(self, image: Image.Image) -> Dict[str, Any]:
        """Multi-stage forensic heuristic detection for AI-generated artifacts."""
        img_array = np.array(image).astype(np.float64)

        if image.width < 100 or image.height < 100:
            return {
                "is_ai_generated": False,
                "composite_score": 0.20,
                "method": "heuristics_skipped_small",
                "details": {"note": "Image too small for reliable forensic heuristics"},
            }

        noise_score = self._analyze_noise(image)
        texture_score = self._analyze_texture(img_array)
        color_score = self._analyze_color_distribution(img_array)
        edge_score = self._analyze_edge_coherence(image)
        frequency_score = self._analyze_frequency(img_array)
        ela_score = self._analyze_ela(image)

        composite = (
            0.20 * noise_score +
            0.22 * texture_score +
            0.15 * color_score +
            0.18 * edge_score +
            0.13 * frequency_score +
            0.12 * ela_score
        )
        composite = max(0.0, min(1.0, composite))

        return {
            "is_ai_generated": bool(composite >= AI_COMPOSITE_THRESHOLD),
            "composite_score": round(float(composite), 4),
            "method": "heuristics",
            "details": {
                "noise_score": round(float(noise_score), 4),
                "texture_score": round(float(texture_score), 4),
                "color_score": round(float(color_score), 4),
                "edge_score": round(float(edge_score), 4),
                "frequency_score": round(float(frequency_score), 4),
                "ela_score": round(float(ela_score), 4),
                "noise_interpretation": "Unusually uniform noise" if noise_score > 0.5 else "Natural noise pattern",
                "texture_interpretation": "Low texture variation" if texture_score > 0.5 else "Normal texture variation",
            },
        }

    def _analyze_noise(self, image: Image.Image) -> float:
        """Laplacian noise analysis (AI images often exhibit synthetic uniform noise)."""
        try:
            gray = image.convert("L")
            laplacian = gray.filter(ImageFilter.Kernel(
                size=(3, 3),
                kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
                scale=1,
                offset=128
            ))
            lap_array = np.array(laplacian).astype(np.float64)
            variance = np.var(lap_array)

            if variance < AI_DETECTION_NOISE_THRESHOLD:
                return float(min(1.0, 1.0 - (variance / AI_DETECTION_NOISE_THRESHOLD)))
            return float(max(0.0, 0.3 - (variance - AI_DETECTION_NOISE_THRESHOLD) / 500))
        except Exception:
            return 0.5

    def _analyze_ela(self, image: Image.Image) -> float:
        """Error Level Analysis measuring recompression error distribution."""
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=90)
            buffer.seek(0)
            compressed_img = Image.open(buffer)

            ela_image = ImageChops.difference(image, compressed_img)
            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])

            if max_diff == 0:
                return 0.6

            scale = 255.0 / max_diff
            ela_image = Image.eval(ela_image, lambda x: x * scale)
            ela_array = np.array(ela_image).astype(np.float64)
            variance = np.var(ela_array)

            if variance < 1500:
                return float(min(1.0, 1.0 - (variance / 1500)))
            return float(max(0.0, 0.4 - (variance - 1500) / 10000))
        except Exception:
            return 0.5

    def _analyze_texture(self, img_array: np.ndarray) -> float:
        """Patch-level texture variance (AI synthesis often creates plastic/smooth zones)."""
        try:
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            h, w = gray.shape
            patch_size = max(16, min(h, w) // 8)
            variances = []

            for i in range(0, h - patch_size, patch_size):
                for j in range(0, w - patch_size, patch_size):
                    patch = gray[i:i + patch_size, j:j + patch_size]
                    variances.append(np.var(patch))

            if not variances:
                return 0.5

            mean_var = np.mean(variances)
            std_var = np.std(variances)

            if mean_var == 0:
                return 0.8

            cv = std_var / mean_var
            if cv < 0.5:
                return float(min(1.0, 0.8 - cv))
            elif cv < 1.5:
                return float(max(0.1, 0.5 - (cv - 0.5) * 0.3))
            return 0.1
        except Exception:
            return 0.5

    def _analyze_color_distribution(self, img_array: np.ndarray) -> float:
        """Color entropy analysis."""
        try:
            scores = []
            for channel in range(3):
                hist, _ = np.histogram(img_array[:, :, channel].flatten(), bins=256, range=(0, 256))
                hist = hist.astype(np.float64)
                hist /= hist.sum() + 1e-10
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                normalized_entropy = entropy / 8.0

                if normalized_entropy < 0.7:
                    scores.append(0.6)
                elif normalized_entropy > 0.95:
                    scores.append(0.4)
                else:
                    scores.append(0.2)
            return float(np.mean(scores))
        except Exception:
            return 0.5

    def _analyze_edge_coherence(self, image: Image.Image) -> float:
        """Edge contour pattern analysis."""
        try:
            gray = image.convert("L")
            edges_find = gray.filter(ImageFilter.FIND_EDGES)
            edges_array = np.array(edges_find).astype(np.float64)
            edge_density = np.mean(edges_array > 30)
            edge_variance = np.var(edges_array)

            density_score = 0.6 if edge_density < 0.1 else (0.5 if edge_density > 0.5 else 0.2)
            variance_score = 0.6 if edge_variance < 500 else max(0.1, 0.4 - (edge_variance - 500) / 5000)

            return float((density_score + variance_score) / 2.0)
        except Exception:
            return 0.5

    def _analyze_frequency(self, img_array: np.ndarray) -> float:
        """FFT frequency domain analysis."""
        try:
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log1p(np.abs(f_shift))

            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2
            radius = min(center_h, center_w) // 4
            low_freq = magnitude[center_h - radius:center_h + radius, center_w - radius:center_w + radius]
            total_energy = np.sum(magnitude)
            low_energy = np.sum(low_freq)

            if total_energy == 0:
                return 0.5

            high_freq_ratio = 1.0 - (low_energy / total_energy)
            if high_freq_ratio < 0.6:
                return 0.6
            elif high_freq_ratio > 0.85:
                return 0.3
            return 0.4
        except Exception:
            return 0.5

    def _analyze_edit_manipulation(self, image: Image.Image) -> Dict[str, Any]:
        """Detect digital manipulation, compositing, splicing, and double compression."""
        try:
            if image.width < 80 or image.height < 80:
                return {"edit_score": 0.20, "details": {"note": "Image too small for edit detection"}}

            img_rgb = image.convert("RGB")
            ela_score = self._ela_multi_quality(img_rgb)
            region_score = self._region_noise_variance(img_rgb)
            blocking_score = self._jpeg_blocking_artifacts(img_rgb)

            edit_score = 0.50 * ela_score + 0.30 * region_score + 0.20 * blocking_score
            edit_score = max(0.0, min(1.0, edit_score))

            return {
                "edit_score": round(float(edit_score), 4),
                "details": {
                    "ela_score": round(float(ela_score), 4),
                    "region_score": round(float(region_score), 4),
                    "blocking_score": round(float(blocking_score), 4),
                    "ela_interpretation": "High ELA variance (possible splice/alteration)" if ela_score > 0.55 else "Normal compression levels",
                    "region_interpretation": "Inconsistent noise patches (possible composite)" if region_score > 0.55 else "Uniform noise pattern",
                },
            }
        except Exception as e:
            logger.warning("Edit detection analysis error: %s", e)
            return {"edit_score": 0.25, "details": {"error": str(e)}}

    def _ela_multi_quality(self, img_rgb: Image.Image) -> float:
        """Multi-quality Error Level Analysis across JPEG quality 75, 90, 95."""
        try:
            scores = []
            for quality in (75, 90, 95):
                buf = io.BytesIO()
                img_rgb.save(buf, "JPEG", quality=quality)
                buf.seek(0)
                recompressed = Image.open(buf).convert("RGB")

                diff = ImageChops.difference(img_rgb, recompressed)
                extrema = diff.getextrema()
                max_diff = max(e[1] for e in extrema)
                if max_diff == 0:
                    continue

                # Forensic contrast normalization
                scale = 255.0 / max_diff
                scaled_diff = Image.eval(diff, lambda x: x * scale)
                diff_array = np.array(scaled_diff).astype(np.float64)

                h, w = diff_array.shape[:2]
                patch = max(16, min(h, w) // 8)
                region_vars = []
                for y in range(0, h - patch, patch):
                    for x in range(0, w - patch, patch):
                        block = diff_array[y:y + patch, x:x + patch]
                        region_vars.append(np.var(block))

                if not region_vars:
                    continue

                mean_var = np.mean(region_vars)
                std_var = np.std(region_vars)
                cv = std_var / (mean_var + 1e-5)

                # Composited/spliced images have sharp local variance discrepancies across patches
                scores.append(min(1.0, cv / 1.1))

            return float(np.mean(scores)) if scores else 0.3
        except Exception:
            return 0.3

    def _region_noise_variance(self, img_rgb: Image.Image) -> float:
        """Patch-level noise variation mapping to spot copy-paste splicing."""
        try:
            gray = np.array(img_rgb.convert("L")).astype(np.float64)
            h, w = gray.shape
            patch = max(32, min(h, w) // 6)

            patch_noise = []
            for y in range(0, h - patch, patch):
                for x in range(0, w - patch, patch):
                    block = gray[y:y + patch, x:x + patch]
                    lap = (
                        np.roll(block, 1, 0) + np.roll(block, -1, 0) +
                        np.roll(block, 1, 1) + np.roll(block, -1, 1) - 4 * block
                    )
                    patch_noise.append(np.var(lap))

            if len(patch_noise) < 4:
                return 0.3

            mean_noise = np.mean(patch_noise)
            std_noise = np.std(patch_noise)

            if mean_noise < 1e-6:
                return 0.5

            cv = std_noise / (mean_noise + 1e-6)
            return float(min(1.0, cv / 3.0))
        except Exception:
            return 0.3

    def _jpeg_blocking_artifacts(self, img_rgb: Image.Image) -> float:
        """Measures 8x8 JPEG grid blocking seams to identify double-compression."""
        try:
            gray = np.array(img_rgb.convert("L")).astype(np.float64)
            h, w = gray.shape
            h_diffs, v_diffs = [], []

            for y in range(8, h - 8, 8):
                row_diff = np.abs(gray[y, :] - gray[y - 1, :]).mean()
                interior = np.abs(gray[y - 1, :] - gray[y - 2, :]).mean()
                h_diffs.append(row_diff - interior)

            for x in range(8, w - 8, 8):
                col_diff = np.abs(gray[:, x] - gray[:, x - 1]).mean()
                interior = np.abs(gray[:, x - 1] - gray[:, x - 2]).mean()
                v_diffs.append(col_diff - interior)

            all_diffs = h_diffs + v_diffs
            if not all_diffs:
                return 0.3

            mean_blocking = np.mean(all_diffs)
            return float(min(1.0, max(0.0, mean_blocking / 4.0)))
        except Exception:
            return 0.3

    def _store_hash(self, phash_str: str, source_url: str = ""):
        """Store image perceptual hash in SQLite database with duplicate avoidance."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM image_hashes WHERE phash = ? AND source_url = ?",
                (phash_str, source_url)
            )
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    "INSERT INTO image_hashes (phash, source_url) VALUES (?, ?)",
                    (phash_str, source_url)
                )
                conn.commit()
        except Exception as e:
            logger.warning("Error recording hash to database: %s", e)
        finally:
            if conn:
                conn.close()

    # =========================================================
    #  MULTI-FACTOR 4-WAY VERDICT CLASSIFIER
    # =========================================================

    def _classify_image_category(
        self,
        ai_confidence: float,
        edit_score: float,
        is_reused: bool,
        has_debunks: bool,
        has_api_results: bool,
        web_match_count: int = 0,
        ai_method: str = "heuristics",
        context_text: str = "",
        vit_raw_score: float = 0.0,
        ela_score: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Synthesize multi-factor decision hierarchy across 5 categories:
          - ai_generated : Image was generated by AI synthesis or 3D CGI concept rendering.
          - edited       : Authentic base image with detected tampering, splicing, or filters.
          - fake         : Authentic photo reused in misleading context or identified as debunked.
          - unverified   : Media with zero web corroboration or ambiguous/synthetic baseline.
          - real         : Authenticated image with verified web publication history and clean forensics.
        """
        EDIT_THRESHOLD = 0.50

        # ── 1. FACT-CHECK / DEBUNK PLATFORMS DETECTED ─────────────────────
        if has_debunks:
            if ai_confidence >= 0.75 or vit_raw_score >= 0.75:
                return {
                    "category": "ai_generated",
                    "confidence": round(max(0.88, ai_confidence, vit_raw_score), 4),
                    "reason": f"Identified as a debunked AI-generated image on fact-checking websites (AI confidence: {max(ai_confidence, vit_raw_score):.0%}).",
                }
            return {
                "category": "fake",
                "confidence": 0.88,
                "reason": "Image matches confirmed fact-checking and debunking reports. Flagged as misleading or fabricated context.",
            }

        # ── 2. CONCEPT ART / SCI-FI ENTITIES / STOCK CGI DETECTION ────────
        ctx = (context_text or "").lower()
        concept_terms = [
            "flying car", "flying cars", "robotic car", "concept car", "concept art",
            "vecteezy", "freepik", "deviantart", "artstation", "civitai", "midjourney",
            "stable diffusion", "3d render", "cgi", "ai generated", "cyberpunk cityscape",
            "futuristic city", "autonomous transportation robotic cars"
        ]
        matched_concept = next((t for t in concept_terms if t in ctx), None)
        if matched_concept and (ai_confidence >= 0.30 or vit_raw_score >= 0.35 or ela_score >= 0.45 or edit_score >= 0.35):
            confidence = round(max(0.78, ai_confidence, vit_raw_score), 4)
            return {
                "category": "ai_generated",
                "confidence": confidence,
                "reason": (
                    f"Identified as synthetic/CGI concept media. Visual patterns and source context ('{matched_concept}') "
                    f"reflect 3D computer graphics or AI generative synthesis rather than authentic camera photography."
                ),
            }

        # ── 3. HIGH CONFIDENCE AI GENERATION (ViT / Strong Forensics) ─────
        if ai_confidence >= AI_HIGH_CONFIDENCE_THRESHOLD or vit_raw_score >= 0.85:
            conf = max(ai_confidence, vit_raw_score)
            if web_match_count >= 5:
                reason = (
                    f"High-confidence AI-generated image ({conf:.0%}) actively circulating online "
                    f"across {web_match_count} webpages. Widespread internet presence reflects viral distribution "
                    f"of synthetic media rather than authentic camera capture."
                )
            else:
                reason = f"AI image detection model scored {conf:.0%} — characteristic structural patterns of synthetic media detected."

            return {
                "category": "ai_generated",
                "confidence": round(conf, 4),
                "reason": reason,
            }

        # ── 4. TAMPERING & EDIT DETECTION ─────────────────────────────────
        if edit_score >= EDIT_THRESHOLD or (ela_score >= 0.65 and edit_score >= 0.42):
            return {
                "category": "edited",
                "confidence": round(max(edit_score, ela_score), 4),
                "reason": (
                    f"Forensic analysis detected image manipulation (edit score: {edit_score:.0%}, ELA variance: {ela_score:.0%}). "
                    f"Error Level Analysis and region noise variance indicate spliced, composited, or filtered areas."
                ),
            }

        # ── 5. MODERATE AI SUSPICION (0.50 to 0.79) ───────────────────────
        if ai_confidence >= AI_COMPOSITE_THRESHOLD or vit_raw_score >= 0.55:
            if has_api_results and web_match_count >= 15 and edit_score < 0.35 and vit_raw_score < 0.55:
                confidence = round(max(0.65, 1.0 - ai_confidence), 4)
                return {
                    "category": "real",
                    "confidence": confidence,
                    "reason": (
                        f"Widespread web publication ({web_match_count} matches) and clean forensic baseline indicate "
                        f"an authentic photograph. Moderate AI suspicion ({ai_confidence:.0%}) resolved in favor of "
                        f"documented real-world origin."
                    ),
                }

            eff_ai = max(ai_confidence, vit_raw_score)
            return {
                "category": "ai_generated",
                "confidence": round(eff_ai, 4),
                "reason": f"AI detection score ({eff_ai:.0%}) exceeds synthetic content threshold.",
            }

        # ── 6. REUSED / MISLEADING RECYCLING ──────────────────────────────
        if is_reused:
            return {
                "category": "fake",
                "confidence": 0.75,
                "reason": "Perceptual hash match in local database indicates this image has been recycled across different stories or sources.",
            }

        # ── 7. CORROBORATED AUTHENTIC MEDIA (Documented Web Matches) ──────
        if has_api_results and web_match_count >= 3:
            confidence = round(max(0.70, 1.0 - max(ai_confidence, edit_score)), 4)
            return {
                "category": "real",
                "confidence": confidence,
                "reason": f"Forensic indicators are normal and {web_match_count} internet matches corroborate genuine publication history.",
            }

        if has_api_results and web_match_count >= 1 and ai_confidence < 0.25 and edit_score < 0.30:
            confidence = round(max(0.60, 1.0 - max(ai_confidence, edit_score)), 4)
            return {
                "category": "real",
                "confidence": confidence,
                "reason": f"Corroborated by {web_match_count} online publication match(es) with clean forensic baseline.",
            }

        # ── 8. UNCORROBORATED / UNVERIFIED (Zero Web Evidence or Ambiguous Forensics) ──
        if web_match_count == 0 or not has_api_results:
            if ai_confidence >= 0.25 or edit_score >= 0.32 or vit_raw_score >= 0.35 or ela_score >= 0.48:
                uncertainty_conf = round(max(0.55, ai_confidence, edit_score, vit_raw_score), 4)
                return {
                    "category": "unverified",
                    "confidence": uncertainty_conf,
                    "reason": (
                        f"Uncorroborated media: Zero documented web appearances or photographic provenance found, "
                        f"while forensic sensors detected non-trivial anomalies (AI score: {max(ai_confidence, vit_raw_score):.0%}, "
                        f"Edit/ELA variance: {max(edit_score, ela_score):.0%}). Cannot be certified as authentic media."
                    ),
                }

            # If clean baseline, but zero web corroboration and unconfigured external search
            return {
                "category": "unverified",
                "confidence": 0.52,
                "reason": (
                    "Uncorroborated media: No synthetic AI artifacts detected locally, but zero web matches "
                    "or photographic provenance records were found. Cannot be certified as authentic media without external corroboration."
                ),
            }

        return {
            "category": "unverified",
            "confidence": 0.50,
            "reason": "Inconclusive forensic analysis and lack of independent external corroboration.",
        }
