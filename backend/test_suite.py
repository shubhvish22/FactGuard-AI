
import io
import os
import sys
import unittest
import threading
import tempfile
import numpy as np
from PIL import Image, ImageDraw

# Add backend directory to path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from config import PHASH_SIMILARITY_THRESHOLD, VERDICT_STATUSES
from verification_engine.text_verifier import fuse_text_signals, TextVerifier
from verification_engine.image_verifier import ImageVerifier
from explanation_engine import explain_text_result, explain_image_result
from utils import extract_search_keywords
from claim_extractor import extract_claims, generate_targeted_query
from evidence_retriever import EvidenceRetriever, EvidenceItem
from claim_cross_referencer import cross_reference_claim
from history_manager import HistoryManager
from security import is_safe_url
from app import app


class TestTextSignalFusion(unittest.TestCase):
    """Tests for the decision fusion engine and refined BERT Guard."""

    def test_bert_guard_protects_high_confidence_against_weak_evidence(self):
        """When BERT confidence >= 0.85, weak or absent external signals cannot flip verdict."""
        # Case A: High-confidence REAL with 0 news sources
        ml_real = {"label": "REAL", "confidence": 0.92}
        src_empty = {"matching_sources": 0, "max_similarity": 0.0, "sources": []}
        fc_empty = {"has_fact_checks": False, "fact_checks": []}

        fused_real = fuse_text_signals(ml_real, src_empty, fc_empty)
        self.assertEqual(fused_real["prediction"], "REAL")
        self.assertGreaterEqual(fused_real["confidence"], 0.85)

        # Case B: High-confidence FAKE with only 1 weak match (18% similarity)
        ml_fake = {"label": "FAKE", "confidence": 0.90}
        src_weak = {
            "matching_sources": 1,
            "max_similarity": 0.18,
            "sources": [{"title": "Unrelated article", "similarity": 0.18}],
        }
        fused_fake = fuse_text_signals(ml_fake, src_weak, fc_empty)
        self.assertEqual(fused_fake["prediction"], "FAKE")
        self.assertGreaterEqual(fused_fake["confidence"], 0.85)

    def test_factcheck_debunk_overrides_linguistic_model(self):
        """Accredited fact-check debunks are ground truth and override the model."""
        ml_result = {"label": "REAL", "confidence": 0.92}
        source_result = {"matching_sources": 0, "max_similarity": 0.0, "sources": []}
        factcheck_result = {
            "has_fact_checks": True,
            "fact_checks": [{"claim": "Sample claim", "rating": "false", "publisher": "PolitiFact"}],
        }

        fused = fuse_text_signals(ml_result, source_result, factcheck_result)
        self.assertEqual(fused["prediction"], "FAKE")
        self.assertGreaterEqual(fused["confidence"], 0.85)
        self.assertTrue(fused["has_conflicting_evidence"])

    def test_strong_news_corroboration_overrides_stylistic_fake(self):
        """Multiple accredited news reports with high similarity confirm real events (e.g. ITC news)."""
        ml_result = {"label": "FAKE", "confidence": 0.99}
        source_result = {
            "matching_sources": 4,
            "max_similarity": 0.64,
            "sources": [
                {"title": "ITC increases cigarette prices of popular brands - India.com", "similarity": 0.64},
                {"title": "ITC cigarette prices hiked - Moneycontrol.com", "similarity": 0.28},
                {"title": "ITC raises cigarette prices - NDTV Profit", "similarity": 0.26},
            ],
        }
        factcheck_result = {"has_fact_checks": False, "fact_checks": []}

        fused = fuse_text_signals(ml_result, source_result, factcheck_result)
        self.assertEqual(fused["prediction"], "REAL")
        self.assertGreaterEqual(fused["confidence"], 0.75)
        self.assertTrue(fused["has_conflicting_evidence"])

    def test_uncertain_model_benefits_from_factcheck(self):
        """When model is uncertain (<= 0.60), verified fact-checks flip verdict to FAKE."""
        ml_result = {"label": "REAL", "confidence": 0.52}
        source_result = {"matching_sources": 0, "max_similarity": 0.0, "sources": []}
        factcheck_result = {
            "has_fact_checks": True,
            "fact_checks": [{"claim": "Fake story", "rating": "fabricated hoax", "publisher": "Snopes"}]
        }

        fused = fuse_text_signals(ml_result, source_result, factcheck_result)
        self.assertEqual(fused["prediction"], "FAKE")

    def test_confidence_normalization_bounds(self):
        """All output confidence scores must remain within strictly normalized [0.0, 1.0]."""
        extreme_cases = [
            ({"label": "REAL", "confidence": 1.0}, {"matching_sources": 10, "max_similarity": 1.0}, {"has_fact_checks": False}),
            ({"label": "FAKE", "confidence": 1.0}, {"matching_sources": 0, "max_similarity": 0.0}, {"has_fact_checks": True, "fact_checks": [{"rating": "pants on fire"}]}),
            ({"label": "UNKNOWN", "confidence": 0.0}, {"matching_sources": 0, "max_similarity": 0.0}, {"has_fact_checks": False}),
        ]
        for ml, src, fc in extreme_cases:
            fused = fuse_text_signals(ml, src, fc)
            self.assertGreaterEqual(fused["confidence"], 0.0)
            self.assertLessEqual(fused["confidence"], 1.0)


class TestTextVerifierPipeline(unittest.TestCase):
    """Tests for TextVerifier input handling and fallbacks."""

    def setUp(self):
        self.verifier = TextVerifier()

    def test_short_input_validation(self):
        """Inputs shorter than 10 characters should return UNKNOWN with clear error message."""
        res = self.verifier.predict("Too short")
        self.assertEqual(res["prediction"], "UNKNOWN")
        self.assertIn("error", res)

    def test_empty_input_validation(self):
        """Empty or whitespace input should be rejected gracefully."""
        res = self.verifier.predict("          ")
        self.assertEqual(res["prediction"], "UNKNOWN")

    def test_keyword_extraction(self):
        """Salient keyword extraction filters common stopwords."""
        headline = "According to breaking reports yesterday, the government launched a major inquiry into renewable energy."
        keywords = extract_search_keywords(headline, max_words=6)
        self.assertNotIn("according", keywords)
        self.assertNotIn("the", keywords)
        self.assertTrue(len(keywords.split("+")) > 2)


class TestImageClassificationAndViralAIFix(unittest.TestCase):
    """Tests for image classification hierarchy and the Viral AI bug fix."""

    def setUp(self):
        self.verifier = ImageVerifier()

    def test_viral_ai_image_not_marked_real(self):
        """CRITICAL FIX: High confidence AI image (0.90) found on 25 websites MUST NOT be marked REAL."""
        verdict = self.verifier._classify_image_category(
            ai_confidence=0.90,
            edit_score=0.20,
            is_reused=False,
            has_debunks=False,
            has_api_results=True,
            web_match_count=25,
            ai_method="huggingface"
        )
        self.assertEqual(verdict["category"], "ai_generated")
        self.assertIn("viral", verdict["reason"].lower())

    def test_debunked_image_marked_fake_or_ai(self):
        """Images matched to fact-checking debunk sites must be flagged as fake/ai_generated."""
        verdict = self.verifier._classify_image_category(
            ai_confidence=0.30,
            edit_score=0.20,
            is_reused=False,
            has_debunks=True,
            has_api_results=True,
            web_match_count=8,
        )
        self.assertEqual(verdict["category"], "fake")

    def test_tampered_image_marked_edited(self):
        """Image with high ELA / region noise tampering score is classified as edited."""
        verdict = self.verifier._classify_image_category(
            ai_confidence=0.20,
            edit_score=0.72,
            is_reused=False,
            has_debunks=False,
            has_api_results=True,
            web_match_count=5,
        )
        self.assertEqual(verdict["category"], "edited")

    def test_authentic_image_marked_real(self):
        """Clean forensic indicators with corroborating web matches must be classified as real."""
        verdict = self.verifier._classify_image_category(
            ai_confidence=0.15,
            edit_score=0.18,
            is_reused=False,
            has_debunks=False,
            has_api_results=True,
            web_match_count=8,
        )
        self.assertEqual(verdict["category"], "real")
        self.assertGreaterEqual(verdict["confidence"], 0.65)


class TestImagePerceptualHashing(unittest.TestCase):
    """Tests for pHash calculations, transformations, and self-match exclusion."""

    def setUp(self):
        self.verifier = ImageVerifier()

        # Create a base test image with realistic texture and color variation
        np.random.seed(42)
        arr = np.random.randint(60, 210, (250, 250, 3), dtype=np.uint8)
        self.base_img = Image.fromarray(arr)
        draw = ImageDraw.Draw(self.base_img)
        draw.rectangle([40, 40, 180, 180], fill=(220, 70, 70))
        draw.ellipse([80, 80, 160, 160], fill=(70, 220, 100))

    def test_identical_image_hash_distance(self):
        """Identical images must have Hamming distance = 0."""
        h1 = self.verifier._compute_phash(self.base_img)
        h2 = self.verifier._compute_phash(self.base_img)
        self.assertEqual(h1 - h2, 0)

    def test_resized_image_matches(self):
        """Resized version of the same image must be recognized as similar (distance <= 10)."""
        resized = self.base_img.resize((120, 120))
        h1 = self.verifier._compute_phash(self.base_img)
        h2 = self.verifier._compute_phash(resized)
        self.assertLessEqual(h1 - h2, PHASH_SIMILARITY_THRESHOLD)

    def test_compressed_jpeg_matches(self):
        """Re-compressed JPEG must preserve perceptual hash distance <= 10."""
        buf = io.BytesIO()
        self.base_img.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        compressed = Image.open(buf)

        h1 = self.verifier._compute_phash(self.base_img)
        h2 = self.verifier._compute_phash(compressed)
        self.assertLessEqual(h1 - h2, PHASH_SIMILARITY_THRESHOLD)

    def test_different_image_does_not_match(self):
        """A completely different image must exceed similarity threshold distance > 10."""
        np.random.seed(999)
        diff_arr = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
        diff_img = Image.fromarray(diff_arr)

        h1 = self.verifier._compute_phash(self.base_img)
        h2 = self.verifier._compute_phash(diff_img)
        self.assertGreater(h1 - h2, PHASH_SIMILARITY_THRESHOLD)

    def test_self_match_exclusion(self):
        """Re-checking the same URL must exclude itself and not falsely flag reuse."""
        test_url = "https://example.com/unique_test_image.jpg"
        phash = self.verifier._compute_phash(self.base_img)

        # Store hash under test_url
        self.verifier._store_hash(str(phash), test_url)

        # Checking with same URL should exclude it
        sim_result = self.verifier._check_hash_similarity(phash, current_source_url=test_url)
        self.assertNotIn(test_url, sim_result.get("matching_urls", []))


class TestSQLiteConcurrency(unittest.TestCase):
    """Verifies that SQLite WAL mode safely handles concurrent worker threads without locking."""

    def test_concurrent_hash_storage(self):
        verifier = ImageVerifier()
        errors = []

        def worker(thread_id):
            try:
                for i in range(10):
                    dummy_hash = f"abcdef1234567890_{thread_id}_{i}"
                    verifier._store_hash(dummy_hash, f"https://example.com/img_{thread_id}_{i}.jpg")
                    verifier._check_hash_similarity(
                        verifier._compute_phash(Image.new("RGB", (100, 100), color=(thread_id * 20, i * 20, 50))),
                        current_source_url=""
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent database operations produced errors: {errors}")


class TestExplanationConsistency(unittest.TestCase):
    """Ensures that explanations never contradict the top-level verdict."""

    def test_fake_verdict_does_not_say_authentic(self):
        result = {
            "image_category": "fake",
            "category_confidence": 0.85,
            "category_reason": "Matched debunk database.",
            "is_reused": True,
            "similarity_score": 0.95,
            "similar_count": 2,
            "ai_confidence": 0.10,
            "edit_score": 0.15,
            "image_dimensions": "500x500",
            "web_search": {},
        }
        explanations = " ".join(explain_image_result(result)).lower()
        self.assertNotIn("appears authentic", explanations)
        self.assertIn("misleading", explanations)

    def test_ai_verdict_does_not_say_authentic(self):
        result = {
            "image_category": "ai_generated",
            "category_confidence": 0.92,
            "category_reason": "High AI score.",
            "is_reused": False,
            "ai_confidence": 0.92,
            "edit_score": 0.10,
            "image_dimensions": "500x500",
            "web_search": {"full_matching_images": [{"url": "http://example.com/1.jpg"}] * 10},
        }
        explanations = " ".join(explain_image_result(result)).lower()
        self.assertNotIn("appears authentic", explanations)
        self.assertIn("ai-generated", explanations)


class TestFlaskAPIEndpoints(unittest.TestCase):
    """Tests for Flask HTTP endpoints, status codes, and input validation."""

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        """GET /health must return 200 with service and engine status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("engines", data)

    def test_verify_text_valid_input(self):
        """POST /verify with valid text returns 200 and prediction."""
        response = self.client.post("/verify", json={
            "text": "The Federal Reserve announced an adjustment in interest rates today."
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)
        self.assertIn("explanations", data)

    def test_verify_text_missing_text_returns_400(self):
        """POST /verify without text field must return 400 (not 500)."""
        response = self.client.post("/verify", json={})
        self.assertEqual(response.status_code, 400)

    def test_verify_text_short_input_returns_400(self):
        """POST /verify with text < 10 characters must return 400."""
        response = self.client.post("/verify", json={"text": "Hello"})
        self.assertEqual(response.status_code, 400)

    def test_verify_image_url_invalid_format_returns_400(self):
        """POST /verify-image-url with non-http URL must return 400."""
        response = self.client.post("/verify-image-url", json={"url": "ftp://not-http.jpg"})
        self.assertEqual(response.status_code, 400)

    def test_upload_image_no_file_returns_400(self):
        """POST /upload-image with empty body must return 400."""
        response = self.client.post("/upload-image", data={})
        self.assertEqual(response.status_code, 400)

    def test_upload_image_corrupt_data_returns_400(self):
        """POST /upload-image with non-image bytes must return 400 (not 500)."""
        data = {
            "image": (io.BytesIO(b"Not a real image file"), "corrupt.jpg")
        }
        response = self.client.post("/upload-image", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_upload_image_valid_binary(self):
        """POST /upload-image with a valid PNG returns 200 and category verdict."""
        img = Image.new("RGB", (100, 100), color=(73, 109, 137))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        data = {
            "image": (buf, "valid_test.png")
        }
        response = self.client.post("/upload-image", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertIn("image_category", result)
        self.assertIn("category_confidence", result)
        self.assertIn("explanations", result)


class TestHuggingFaceFreeModelIntegration(unittest.TestCase):
    """Tests ensuring Hugging Face free models serve as primary engines and reliable API fallbacks."""

    def setUp(self):
        self.client = app.test_client()
        self.text_verifier = TextVerifier()
        self.image_verifier = ImageVerifier()

    def test_text_verifier_returns_hf_mode_and_metadata(self):
        """Text verifier must report HF model name, is_free_huggingface_model, and verification_mode."""
        result = self.text_verifier.predict("The stock market experienced steady growth throughout the afternoon session.")
        self.assertIn("verification_mode", result)
        self.assertIn("is_free_huggingface_model", result)
        self.assertTrue(result["is_free_huggingface_model"])
        self.assertEqual(result["verification_mode"], "huggingface_free_local")
        self.assertIn("Fake-News-Bert", result.get("model_name", ""))

    def test_text_explanation_highlights_hf_fallback(self):
        """When fact-check API is unconfigured, explanation must highlight Hugging Face free fallback."""
        sample_result = {
            "prediction": "REAL",
            "confidence": 0.88,
            "ml_label": "REAL",
            "ml_score": 0.88,
            "sources": [],
            "fact_checks": [],
            "source_corroboration": 0.0,
            "matching_sources": 0,
            "has_conflicting_evidence": False,
            "verification_mode": "huggingface_free_local",
            "model_name": "jy46604790/Fake-News-Bert-Detect",
            "is_free_huggingface_model": True,
            "has_factcheck_api": False,
        }
        explanations = explain_text_result(sample_result)
        combined_text = " ".join(explanations).lower()
        self.assertIn("hugging face", combined_text)

    def test_image_verifier_returns_hf_mode_and_metadata(self):
        """Image verifier must report HF metadata and use huggingface_free_local when web API is absent."""
        img = Image.new("RGB", (150, 150), color=(120, 160, 200))
        result = self.image_verifier.verify(img, source_url="test:sample.jpg")
        self.assertIn("verification_mode", result)
        self.assertIn("is_free_huggingface_model", result)
        self.assertTrue(result["is_free_huggingface_model"])
        self.assertEqual(result["verification_mode"], "huggingface_free_local")
        self.assertIn("AI-image-detector", result.get("model_name", ""))

    def test_image_explanation_highlights_hf_fallback(self):
        """When external web search API is simulated/absent, explanation highlights HF ViT fallback."""
        sample_result = {
            "image_category": "real",
            "category_confidence": 0.70,
            "category_reason": "No synthetic AI or editing patterns detected by free Hugging Face ViT model.",
            "is_reused": False,
            "ai_confidence": 0.15,
            "ai_method": "huggingface",
            "edit_score": 0.10,
            "image_dimensions": "300x300",
            "web_search": {"is_simulated": True},
            "web_pages_count": 0,
            "verification_mode": "huggingface_free_local",
            "is_free_huggingface_model": True,
        }
        explanations = explain_image_result(sample_result)
        combined_text = " ".join(explanations).lower()
        self.assertIn("hugging face", combined_text)

    def test_health_reports_hf_status(self):
        """Health check endpoint must explicitly report Hugging Face free fallback status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        engines = data.get("engines", {})
        self.assertTrue(engines.get("huggingface_free_fallback_active"))
        self.assertIn("Fake-News-Bert", engines.get("text_model_name", ""))
        self.assertIn("AI-image-detector", engines.get("image_model_name", ""))


class TestClaimExtractor(unittest.TestCase):
    """Tests for claim decomposition, opinion filtering, and query formulation."""

    def test_extract_multiple_factual_claims(self):
        """Paragraph with multiple factual statements is separated into individual claims."""
        text = (
            "The European Union approved a new artificial intelligence regulation on Wednesday. "
            "Meanwhile, the European Central Bank kept interest rates unchanged at 4.5%. "
            "In my personal opinion, technology will always change everything."
        )
        claims = extract_claims(text)
        self.assertGreaterEqual(len(claims), 2)
        for c in claims:
            self.assertIn("claim", c)
            self.assertIn("claim_id", c)
            self.assertIn("targeted_query", c)
            self.assertIn("original_context", c)
            self.assertTrue(len(c["targeted_query"]) > 5)

    def test_pure_opinion_filtered(self):
        """Pure subjective opinions should not be treated as verifiable factual claims."""
        text = "I think chocolate ice cream is the best dessert ever invented in the world."
        claims = extract_claims(text)
        # Should be filtered out or return 0 factual claims
        self.assertEqual(len(claims), 0)

    def test_targeted_query_focuses_on_entities_and_action(self):
        """Targeted query strips fluff and focuses on core entities and verbs."""
        claim = "According to recent statements, the World Health Organization reported an increase in cases."
        query = generate_targeted_query(claim)
        self.assertNotIn("according", query.lower())
        self.assertNotIn("recent", query.lower())
        self.assertIn("world health organization", query.lower().replace("+", " "))



class TestEvidenceRetriever(unittest.TestCase):
    """Tests for multi-provider retrieval, deduplication, and source tiering."""

    def setUp(self):
        self.retriever = EvidenceRetriever()

    def test_source_tier_classification(self):
        """Source URLs are mapped strictly to Tiers 1-5 based on authority."""
        self.assertEqual(self.retriever._classify_source_tier("https://www.cdc.gov/flu/weekly"), 1)
        self.assertEqual(self.retriever._classify_source_tier("https://www.navy.mil/news"), 1)
        self.assertEqual(self.retriever._classify_source_tier("https://www.harvard.edu/research"), 2)
        self.assertEqual(self.retriever._classify_source_tier("https://www.snopes.com/fact-check/moon"), 3)
        self.assertEqual(self.retriever._classify_source_tier("https://www.politifact.com/factchecks/1"), 3)
        self.assertEqual(self.retriever._classify_source_tier("https://www.reuters.com/world/us"), 4)
        self.assertEqual(self.retriever._classify_source_tier("https://www.bbc.com/news/world"), 4)
        self.assertEqual(self.retriever._classify_source_tier("https://random-unverified-blog.xyz/post"), 5)

    def test_evidence_deduplication_by_canonical_url(self):
        """Duplicate URLs from multiple search providers are deduplicated cleanly."""
        item1 = EvidenceItem(
            claim="Sample claim",
            title="Article One",
            url="https://example.com/article-1",
            snippet="Snippet 1",
            publisher="Example",
            source_tier=4
        )
        item2 = EvidenceItem(
            claim="Sample claim",
            title="Article One Duplicate",
            url="https://example.com/article-1",
            snippet="Snippet 1 dup",
            publisher="Example",
            source_tier=4
        )
        deduped = self.retriever._deduplicate_evidence([item1, item2])
        self.assertEqual(len(deduped), 1)


class TestClaimCrossReferencer(unittest.TestCase):
    """Tests for claim verification, stance analysis, and 5-status verdicts."""

    def test_verdict_statuses_adhere_to_specification(self):
        """Only the 5 specified statuses are allowed in the system."""
        expected_statuses = {"Supported", "Contradicted", "Mixed Evidence", "Insufficient Evidence", "Not Verifiable"}
        self.assertEqual(set(VERDICT_STATUSES), expected_statuses)

    def test_empty_evidence_yields_insufficient_evidence(self):
        """CRITICAL: Do NOT classify claim as false when no evidence found."""
        claim_obj = {
            "claim_id": "c1",
            "claim": "A rare bird species was observed in the Amazon canopy.",
            "original_context": "Full text",
            "targeted_query": "rare bird species Amazon canopy"
        }
        result = cross_reference_claim(claim_obj, evidence=[])
        self.assertEqual(result["status"], "Insufficient Evidence")
        self.assertNotEqual(result["status"], "Contradicted")
        self.assertEqual(len(result["supporting_evidence"]), 0)
        self.assertEqual(len(result["contradicting_evidence"]), 0)
        self.assertIn("insufficient", result["explanation"].lower())
        self.assertTrue(len(result["uncertainty_or_limitations"]) > 0)

    def test_supported_verdict_when_tier1_confirms(self):
        """Primary/official sources affirming claim produce Supported verdict."""
        claim_obj = {
            "claim_id": "c2",
            "claim": "The James Webb Space Telescope was launched in December 2021.",
            "original_context": "Full text",
            "targeted_query": "James Webb Space Telescope launch December 2021"
        }
        ev = EvidenceItem(
            claim=claim_obj["claim"],
            title="NASA launches James Webb Space Telescope in December 2021",
            url="https://www.nasa.gov/jwst-launch",
            snippet="The James Webb Space Telescope successfully launched in December 2021.",
            publisher="NASA",
            source_tier=1
        )
        result = cross_reference_claim(claim_obj, evidence=[ev])
        self.assertEqual(result["status"], "Supported")
        self.assertEqual(len(result["supporting_evidence"]), 1)
        self.assertEqual(len(result["contradicting_evidence"]), 0)

    def test_contradicted_verdict_when_factcheck_debunks(self):
        """Accredited fact-check debunking claim produces Contradicted verdict."""
        claim_obj = {
            "claim_id": "c3",
            "claim": "Drinking bleach cures viral infections.",
            "original_context": "Full text",
            "targeted_query": "drinking bleach cures viral infections"
        }
        ev = EvidenceItem(
            claim=claim_obj["claim"],
            title="Fact Check: Drinking bleach is hazardous and does not cure infections",
            url="https://www.snopes.com/fact-check/bleach-cure",
            snippet="False: Health authorities confirm drinking bleach is toxic and does not cure any virus.",
            publisher="Snopes",
            source_tier=2
        )
        result = cross_reference_claim(claim_obj, evidence=[ev])
        self.assertEqual(result["status"], "Contradicted")
        self.assertEqual(len(result["contradicting_evidence"]), 1)
        self.assertEqual(len(result["supporting_evidence"]), 0)


class TestHistoryManager(unittest.TestCase):
    """Tests for SQLite history persistence, pagination, search, and deletion."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_history.db")
        self.manager = HistoryManager(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_retrieve_history_entry(self):
        """History manager correctly stores and retrieves entries with claims."""
        claims = [
            {
                "claim_id": "c1",
                "claim": "Test claim statement",
                "status": "Supported",
                "supporting_evidence": [{"url": "https://example.com/source", "publisher": "Test Pub"}],
                "contradicting_evidence": []
            }
        ]
        entry_id = self.manager.save_entry(
            item_type="text",
            submitted_text="This is a test submission containing factual claims.",
            source_url="https://example.com/original-article",
            webpage_title="Original Article Title",
            claims=claims,
            top_verdict="Supported"
        )
        self.assertIsInstance(entry_id, int)

        # Retrieve entry by ID
        entry = self.manager.get_entry(entry_id)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["item_type"], "text")
        self.assertEqual(entry["source_url"], "https://example.com/original-article")
        self.assertEqual(entry["webpage_title"], "Original Article Title")
        self.assertEqual(len(entry["claims"]), 1)
        self.assertEqual(entry["claims"][0]["claim"], "Test claim statement")

    def test_history_filtering_and_pagination(self):
        """History filters by item_type and supports text query search and pagination."""
        self.manager.save_entry(item_type="text", submitted_text="Renewable energy records in Europe.")
        self.manager.save_entry(item_type="text", submitted_text="Central bank interest rate report.")
        self.manager.save_entry(item_type="image", image_identifier="mars_rover_photo.jpg")

        # Filter by item_type
        text_results = self.manager.get_history(item_type="text", page=1, per_page=10)
        self.assertEqual(text_results["total"], 2)

        image_results = self.manager.get_history(item_type="image", page=1, per_page=10)
        self.assertEqual(image_results["total"], 1)

        # Search query
        query_results = self.manager.get_history(query="renewable", page=1, per_page=10)
        self.assertEqual(query_results["total"], 1)
        self.assertIn("Renewable", query_results["items"][0]["submitted_text"])

    def test_history_deletion(self):
        """Deleting an entry removes it from the database."""
        entry_id = self.manager.save_entry(item_type="text", submitted_text="Entry to delete.")
        self.assertIsNotNone(self.manager.get_entry(entry_id))

        deleted = self.manager.delete_entry(entry_id)
        self.assertTrue(deleted)
        self.assertIsNone(self.manager.get_entry(entry_id))

    def test_history_clear_all(self):
        """Clearing all entries wipes verification_history table."""
        self.manager.save_entry(item_type="text", submitted_text="Entry 1")
        self.manager.save_entry(item_type="text", submitted_text="Entry 2")
        cleared_count = self.manager.clear_all()
        self.assertGreaterEqual(cleared_count, 2)
        history_data = self.manager.get_history(page=1, per_page=10)
        self.assertEqual(history_data["total"], 0)
        self.assertEqual(len(history_data["items"]), 0)


class TestSecurityAndSSRF(unittest.TestCase):
    """Tests SSRF prevention, IP restriction, and URL sanitization."""

    def test_blocks_private_and_loopback_ips(self):
        """SSRF protection blocks localhost, loopback, RFC1918 private IPs, and metadata servers."""
        blocked_urls = [
            "http://127.0.0.1:5000/health",
            "http://localhost:8000/",
            "http://192.168.1.1/router",
            "http://10.0.0.1/internal",
            "http://172.16.0.1/private",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]:8080/",
        ]
        for url in blocked_urls:
            self.assertFalse(is_safe_url(url), f"Should have blocked unsafe URL: {url}")

    def test_blocks_dangerous_schemes(self):
        """SSRF protection blocks non-http schemes like file://, gopher://, javascript:."""
        unsafe_schemes = [
            "file:///etc/passwd",
            "file:///C:/Windows/System32",
            "gopher://evil.com/",
            "ftp://files.example.com/",
            "javascript:alert(1)",
        ]
        for url in unsafe_schemes:
            self.assertFalse(is_safe_url(url), f"Should have blocked scheme: {url}")

    def test_allows_valid_public_web_urls(self):
        """Valid public HTTPS web URLs are permitted."""
        valid_urls = [
            "https://www.reuters.com/world",
            "https://www.cdc.gov/flu",
            "https://en.wikipedia.org/wiki/Earth",
        ]
        for url in valid_urls:
            self.assertTrue(is_safe_url(url), f"Should have allowed valid URL: {url}")


class TestNewAPIEndpoints(unittest.TestCase):
    """Tests for new REST endpoints: /fact-check, /history, and source tracking."""

    def setUp(self):
        self.client = app.test_client()

    def test_fact_check_endpoint_returns_decomposed_claims(self):
        """POST /fact-check extracts claims, performs verification, and returns 200."""
        text = "The World Bank published its annual global economic forecast today."
        response = self.client.post("/fact-check", json={"text": text})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("claims", data)
        self.assertIn("fact_check_summary", data)
        self.assertIn("history_id", data)

    def test_history_list_endpoint_pagination(self):
        """GET /history returns paginated items and metadata."""
        response = self.client.get("/history?page=1&per_page=5")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("items", data)
        self.assertIn("total", data)
        self.assertIn("page", data)
        self.assertIn("total_pages", data)

    def test_verify_preserves_source_metadata(self):
        """POST /verify records source_url and webpage_title in history."""
        sample_url = "https://example.com/verified-news-story"
        sample_title = "Verified News Story Headline"
        response = self.client.post("/verify", json={
            "text": "The European Union announced updated climate targets for 2030.",
            "source_url": sample_url,
            "webpage_title": sample_title
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        history_id = data.get("history_id")
        self.assertIsNotNone(history_id)

        # Retrieve from history endpoint
        hist_res = self.client.get(f"/history/{history_id}")
        self.assertEqual(hist_res.status_code, 200)
        entry = hist_res.get_json()["entry"]
        self.assertEqual(entry["source_url"], sample_url)
        self.assertEqual(entry["webpage_title"], sample_title)


if __name__ == "__main__":
    unittest.main()

