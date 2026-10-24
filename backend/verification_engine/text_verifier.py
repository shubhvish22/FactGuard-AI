
import os
import logging
from typing import Dict, Any

import joblib
import numpy as np

from config import (
    TFIDF_MODEL_PATH,
    LOGREG_MODEL_PATH,
    MIN_SIMILARITY_SCORE,
    MAX_SOURCES_RETURNED,
    HIGH_CONFIDENCE_THRESHOLD,
)
from utils import (
    preprocess_text,
    compute_text_similarity,
    fetch_google_news_rss,
    fetch_factcheck_results,
)
from verification_engine.bert_text_verifier import BertTextVerifier

logger = logging.getLogger("FactGuard.TextVerifier")


DEBUNK_HEADLINE_KEYWORDS = (
    "fact check", "fact-check", "factcheck", "debunk", "hoax",
    "false claim", "fake claim", "rumor", "rumour", "myth",
    "misinformation", "disinformation", "busted", "untrue", "distorted"
)


def fuse_text_signals(
    ml_result: Dict[str, Any],
    source_result: Dict[str, Any],
    factcheck_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Pure decision fusion function combining ML, News Corroboration, and Fact-Checking.
    Designed as an independent, testable function.

    Hierarchy of Evidence:
      1. Accredited Fact-Check Debunk: Official fact-checker debunking is authoritative (verdict: FAKE).
      2. Ground-Truth News Corroboration: Multiple accredited news outlets actively reporting
         an event with high semantic similarity and zero debunks confirms a real news event (verdict: REAL).
         This prevents stylistic NLP false-positives on breaking or business headlines.
      3. BERT Guard on Weak Evidence: When external evidence is weak (0 sources or low similarity < 0.35)
         and no fact-checks exist, high-confidence ML predictions (>= 0.85) are firmly preserved.
      4. Balanced Linear Fusion: For moderate or uncertain ML predictions, signals are weighted smoothly.
      5. Conflict Transparency: Contradictory evidence between ML and external reporting is explicitly surfaced.
    """
    ml_confidence = float(ml_result.get("confidence", 0.0))
    ml_confidence = max(0.0, min(1.0, ml_confidence))
    ml_label = str(ml_result.get("label", "UNKNOWN")).upper()

    sources = source_result.get("sources", [])
    matching_sources = int(source_result.get("matching_sources", len(sources)))
    source_corroboration = float(source_result.get("max_similarity", 0.0))
    has_fact_checks = bool(factcheck_result.get("has_fact_checks", False))
    fact_checks = factcheck_result.get("fact_checks", [])

    # Evaluate Fact-Check signals
    false_keywords = {
        "false", "fake", "incorrect", "pants on fire", "misleading",
        "untrue", "hoax", "satire", "misinformation", "fabricated", "distorted"
    }
    true_keywords = {
        "true", "correct", "accurate", "verified", "mostly true", "confirmed", "legitimate"
    }

    ratings = [str(fc.get("rating", "")).lower() for fc in fact_checks if fc.get("rating")]
    has_debunk = any(any(k in r for k in false_keywords) for r in ratings)
    has_verified = any(any(k in r for k in true_keywords) for r in ratings)

    # Check if news articles are actively reporting a debunk rather than straight news
    news_debunk_detected = any(
        any(dk in str(s.get("title", "")).lower() for dk in DEBUNK_HEADLINE_KEYWORDS)
        for s in sources
        if float(s.get("similarity", 0.0)) >= 0.25
    )

    # Filter clean sources (straightforward news reporting without debunk flags)
    clean_sources = [
        s for s in sources
        if not any(dk in str(s.get("title", "")).lower() for dk in DEBUNK_HEADLINE_KEYWORDS)
    ]
    solid_matches = [s for s in clean_sources if float(s.get("similarity", 0.0)) >= 0.25]
    strong_matches = [s for s in clean_sources if float(s.get("similarity", 0.0)) >= 0.40]

    # Strong live news corroboration: multiple news reports or high similarity reporting
    is_strong_news_corroboration = (
        not has_debunk
        and not news_debunk_detected
        and (
            len(strong_matches) >= 2
            or (source_corroboration >= 0.45 and (matching_sources >= 2 or len(solid_matches) >= 2))
            or (len(solid_matches) >= 3 and source_corroboration >= 0.35)
            or (source_corroboration >= 0.55 and (matching_sources >= 1 or len(solid_matches) >= 1))
        )
    )

    has_conflict = False

    # --- DECISION RULES ---

    # Rule 1: Accredited Fact-Check Debunk Found (or news reports active debunk)
    if has_debunk or news_debunk_detected:
        final_prediction = "FAKE"
        final_confidence = max(0.85, min(0.98, 0.85 + 0.03 * len(fact_checks)))
        if ml_label == "REAL":
            has_conflict = True

    # Rule 2: Accredited Fact-Check Confirmed True
    elif has_verified:
        final_prediction = "REAL"
        final_confidence = 0.90
        if ml_label == "FAKE":
            has_conflict = True

    # Rule 3: Strong Live News Corroboration from Recognized Media
    elif is_strong_news_corroboration:
        final_prediction = "REAL"
        if ml_label == "FAKE":
            # The static NLP model produced a stylistic false positive on live news
            has_conflict = True
            # Confidence scaled smoothly based on the strength of news matches
            final_confidence = min(0.90, max(0.75, 0.55 + 0.45 * source_corroboration))
        elif ml_label == "REAL":
            has_conflict = False
            final_confidence = min(0.98, max(ml_confidence, 0.70 + 0.25 * source_corroboration))
        else:
            final_confidence = min(0.90, max(0.70, 0.50 + 0.40 * source_corroboration))

    # Rule 4: BERT Guard Protection on High-Confidence ML with Weak External Evidence
    elif ml_confidence >= HIGH_CONFIDENCE_THRESHOLD and ml_label != "UNKNOWN":
        # External evidence is weak (< 0.35 similarity or 0 sources) and no debunks exist:
        # BERT Guard preserves the high-confidence model prediction!
        final_prediction = ml_label
        final_confidence = max(0.80, min(0.98, ml_confidence))
        has_conflict = False

    # Rule 5: Uncertain / Moderate ML with moderate or weak news signals
    else:
        if ml_label == "REAL":
            ml_realness = ml_confidence
        elif ml_label == "FAKE":
            ml_realness = 1.0 - ml_confidence
        else:
            ml_realness = 0.50

        news_bonus = 0.0
        if len(solid_matches) >= 1:
            news_bonus = min(0.30, 0.10 + 0.20 * source_corroboration)

        combined_realness = 0.60 * ml_realness + 0.40 * (0.50 + news_bonus)
        combined_realness = max(0.0, min(1.0, combined_realness))

        if combined_realness >= 0.50:
            final_prediction = "REAL"
            final_confidence = combined_realness
        else:
            final_prediction = "FAKE"
            final_confidence = 1.0 - combined_realness

    return {
        "prediction": final_prediction,
        "confidence": round(float(final_confidence), 4),
        "ml_label": ml_label,
        "ml_score": round(float(ml_confidence), 4),
        "sources": sources,
        "fact_checks": fact_checks,
        "source_corroboration": round(float(source_corroboration), 4),
        "matching_sources": matching_sources,
        "has_conflicting_evidence": has_conflict,
    }


class TextVerifier:
    """Handles end-to-end text verification with cached models and automatic fallbacks."""

    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.bert = None
        self.model_type = "none"
        self._load_models()

    def _load_models(self):
        """Try loading BERT first; fall back to TF-IDF + Logistic Regression."""
        # 1. Try BERT
        try:
            bert = BertTextVerifier()
            if bert.available:
                self.bert = bert
                self.model_type = "bert"
                logger.info("Using BERT transformer for primary text classification.")
                return
        except Exception as e:
            logger.warning("BERT initialization error (%s). Trying TF-IDF fallback...", e)

        # 2. Try TF-IDF + Logistic Regression
        if os.path.exists(TFIDF_MODEL_PATH) and os.path.exists(LOGREG_MODEL_PATH):
            try:
                self.vectorizer = joblib.load(TFIDF_MODEL_PATH)
                self.model = joblib.load(LOGREG_MODEL_PATH)
                self.model_type = "tfidf"
                logger.info("Using TF-IDF + Logistic Regression fallback model.")
            except Exception as e:
                logger.error("Failed to load TF-IDF models: %s", e)
                self.model_type = "none"
        else:
            logger.warning("No pre-trained text models found. Run train_model.py to build TF-IDF model.")
            self.model_type = "none"

    def predict(self, text: str, source_url: str = "", page_title: str = "") -> Dict[str, Any]:
        """Execute the full text verification pipeline with claim-by-claim evidence checking."""
        if not text or len(text.strip()) < 10:
            return {
                "prediction": "UNKNOWN",
                "confidence": 0.0,
                "ml_score": 0.0,
                "ml_label": "UNKNOWN",
                "sources": [],
                "fact_checks": [],
                "source_corroboration": 0.0,
                "matching_sources": 0,
                "has_conflicting_evidence": False,
                "verification_mode": "none",
                "model_name": getattr(getattr(self, "bert", None), "model_name", "none"),
                "is_free_huggingface_model": (self.model_type == "bert"),
                "has_factcheck_api": False,
                "claims": [],
                "error": "Text too short for analysis (minimum 10 characters).",
            }

        # Step 1: ML Model Classification (Primary: Hugging Face free model; Secondary: TF-IDF)
        ml_result = self._ml_predict(text)

        # Step 2: Live News Corroboration (Google News RSS)
        source_result = self._verify_sources(text)

        # Step 3: Google Fact Check Tools Lookup
        factcheck_result = self._factcheck_lookup(text)

        # Step 4: Decision Fusion with BERT Guard
        combined = fuse_text_signals(ml_result, source_result, factcheck_result)

        # Step 5: Verification Mode & Hugging Face Free Model Transparency
        is_hf = bool(ml_result.get("is_free_huggingface_model", self.model_type == "bert"))
        ml_method = ml_result.get("method", "huggingface_local" if self.model_type == "bert" else "tfidf")
        has_fc = bool(factcheck_result.get("has_fact_checks", False))

        if has_fc:
            verification_mode = "factcheck_api"
        elif ml_method == "huggingface_api":
            verification_mode = "huggingface_free_api"
        elif is_hf:
            verification_mode = "huggingface_free_local"
        else:
            verification_mode = "tfidf_fallback"

        model_name = ml_result.get("model_name") or getattr(self.bert, "model_name", "") or "TF-IDF Logistic Regression"

        combined["verification_mode"] = verification_mode
        combined["model_name"] = model_name
        combined["ml_method"] = ml_method
        combined["is_free_huggingface_model"] = is_hf
        combined["has_factcheck_api"] = has_fc

        # Step 6: Claim-by-Claim Extraction and Evidence Verification
        from claim_extractor import extract_claims
        from evidence_retriever import EvidenceRetriever
        from claim_cross_referencer import cross_reference_all_claims

        extracted_claims = extract_claims(text)
        retriever = EvidenceRetriever()
        evidence_by_claim = {}
        all_evidence_sources = []
        for cl in extracted_claims:
            cid = cl.get("claim_id", 1)
            q = cl.get("search_query") or cl.get("statement", "")
            ev_list = retriever.retrieve_evidence(q, cl.get("statement", ""), max_sources=5)
            evidence_by_claim[cid] = ev_list
            all_evidence_sources.extend(ev_list)

        claim_verdicts = cross_reference_all_claims(extracted_claims, evidence_by_claim)
        combined["claims"] = claim_verdicts

        # Step 7: Persistent History Recording
        try:
            from history_manager import HistoryManager
            hm = HistoryManager()
            history_id = hm.save_entry(
                submission_type="text",
                content_preview=text[:250],
                source_url=source_url,
                page_title=page_title,
                claims=claim_verdicts,
                overall_verdict=combined["prediction"],
                ml_prediction=combined["ml_label"],
                ml_confidence=combined["ml_score"],
                evidence_sources=all_evidence_sources[:15],
                source_discovery_status="sources_found" if all_evidence_sources else "insufficient_evidence",
            )
            combined["history_id"] = history_id
        except Exception as hist_err:
            logger.warning("History save error on text verification: %s", hist_err)
            combined["history_id"] = None

        return combined

    def _ml_predict(self, text: str) -> Dict[str, Any]:
        """Classify text using the best available model (BERT or TF-IDF)."""
        if self.bert is not None and self.bert.available:
            return self.bert.predict(text)

        if self.vectorizer is None or self.model is None:
            return {
                "label": "UNKNOWN",
                "confidence": 0.0,
                "probabilities": {},
            }

        try:
            processed = preprocess_text(text)
            features = self.vectorizer.transform([processed])
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]

            classes = self.model.classes_.tolist()
            pred_index = classes.index(prediction)
            confidence = float(probabilities[pred_index])
            confidence = max(0.0, min(1.0, confidence))

            label = "REAL" if str(prediction).upper() in ["REAL", "1", "TRUE"] else "FAKE"
            prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}

            return {
                "label": label,
                "confidence": round(confidence, 4),
                "probabilities": prob_dict,
            }
        except Exception as e:
            logger.error("TF-IDF prediction exception: %s", e)
            return {
                "label": "UNKNOWN",
                "confidence": 0.0,
                "probabilities": {},
            }

    def _verify_sources(self, text: str) -> Dict[str, Any]:
        """Fetch news articles and compute high-fidelity cosine similarity."""
        try:
            articles = fetch_google_news_rss(text, max_results=MAX_SOURCES_RETURNED)
            if not articles:
                return {
                    "sources": [],
                    "avg_similarity": 0.0,
                    "max_similarity": 0.0,
                    "matching_sources": 0,
                }

            # Compute similarities for both titles alone and title+summary combined
            titles = [a.get("title", "") for a in articles]
            title_sims = compute_text_similarity(text, titles)

            combined_texts = [f"{a.get('title', '')} {a.get('summary', '')}" for a in articles]
            combined_sims = compute_text_similarity(text, combined_texts)

            # High fidelity match: take max of title match and body match.
            # This prevents short headline queries from being artificially diluted by long descriptions.
            similarities = [max(ts, cs) for ts, cs in zip(title_sims, combined_sims)]

            sources = []
            for article, sim in zip(articles, similarities):
                sources.append({
                    "title": article.get("title", ""),
                    "url": article.get("link", ""),
                    "similarity": round(float(sim), 4),
                    "published": article.get("published", ""),
                })

            sources.sort(key=lambda x: x["similarity"], reverse=True)
            top_sources = sources[:MAX_SOURCES_RETURNED]
            matching = [s for s in sources if s["similarity"] >= MIN_SIMILARITY_SCORE]

            max_sim = max(similarities) if similarities else 0.0
            avg_sim = float(np.mean(similarities)) if similarities else 0.0

            return {
                "sources": top_sources,
                "avg_similarity": round(avg_sim, 4),
                "max_similarity": round(max_sim, 4),
                "matching_sources": len(matching),
            }
        except Exception as e:
            logger.warning("Source verification failed: %s", e)
            return {
                "sources": [],
                "avg_similarity": 0.0,
                "max_similarity": 0.0,
                "matching_sources": 0,
            }

    def _factcheck_lookup(self, text: str) -> Dict[str, Any]:
        """Query fact-checking databases for matching claims."""
        try:
            results = fetch_factcheck_results(text)
            fact_checks = []
            for r in results:
                fact_checks.append({
                    "claim": r.get("claim", ""),
                    "claimant": r.get("claimant", "Unknown"),
                    "rating": r.get("rating", ""),
                    "url": r.get("url", ""),
                    "publisher": r.get("publisher", "Unknown"),
                })
            return {
                "fact_checks": fact_checks,
                "has_fact_checks": len(fact_checks) > 0,
            }
        except Exception as e:
            logger.warning("Fact check lookup error: %s", e)
            return {
                "fact_checks": [],
                "has_fact_checks": False,
            }
