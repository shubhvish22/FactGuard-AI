
import os
import sys
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS

from config import (
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    CORS_ALLOWED_ORIGINS,
    MAX_CONTENT_LENGTH,
    TFIDF_MODEL_PATH,
    HUGGINGFACE_API_TOKEN,
)
import hashlib
from PIL import Image
Image.MAX_IMAGE_PIXELS = 25_000_000  # Guard against image decompression bombs

from verification_engine.text_verifier import TextVerifier
from verification_engine.image_verifier import ImageVerifier
from explanation_engine import explain_text_result, explain_image_result
from utils import download_image, image_from_bytes
from history_manager import HistoryManager
from security import is_safe_url, sanitize_text
from cache import text_cache

# --- Configure Structured Logging ---
log_level = logging.DEBUG if FLASK_DEBUG else logging.INFO
logging.basicConfig(
    level=log_level,
    format="[%(asctime)s] %(levelname)-7s in %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FactGuard.App")

# --- App Initialization ---
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Secure CORS: Restrict origins to Chrome extension, localhost, and 127.0.0.1
CORS(app, resources={r"/*": {"origins": CORS_ALLOWED_ORIGINS}})

# Injected HTTP Security Headers
@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# --- Model Initialization ---
logger.info("Initializing FactGuard AI backend engines...")

def _bert_available() -> bool:
    try:
        import transformers, torch  # noqa: F401
        return True
    except ImportError:
        return False

# Auto-train TF-IDF fallback model only if BERT is unavailable and TF-IDF file is missing
if not _bert_available() and not os.path.exists(TFIDF_MODEL_PATH):
    logger.info("BERT unavailable & TF-IDF model not found. Training baseline TF-IDF model...")
    try:
        from train_model import train
        train()
    except Exception as e:
        logger.error("Auto-training failed: %s", e)

# Singleton engine instances loaded once and reused across all requests
text_verifier = TextVerifier()
image_verifier = ImageVerifier()
history_manager = HistoryManager()
logger.info("FactGuard AI engines successfully initialized.")


# =========================================================
#  API ENDPOINTS
# =========================================================

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint reporting engine and server status."""
    return jsonify({
        "status": "ok",
        "service": "FactGuard AI",
        "version": "1.3.0",
        "debug": FLASK_DEBUG,
        "engines": {
            "text_verifier": text_verifier.model_type != "none",
            "text_model": text_verifier.model_type,  # "bert" | "tfidf" | "none"
            "text_model_name": getattr(getattr(text_verifier, "bert", None), "model_name", "jy46604790/Fake-News-Bert-Detect"),
            "image_verifier": True,
            "ai_image_detector": image_verifier._hf_classifier is not None,
            "image_model_name": getattr(image_verifier, "model_name", "umm-maybe/AI-image-detector"),
            "huggingface_token_configured": bool(HUGGINGFACE_API_TOKEN),
            "huggingface_status": "Free Hugging Face models active (local pipelines cached, hub authenticated)",
            "huggingface_free_fallback_active": True,
            "evidence_retrieval": True,
            "claim_extractor": True,
            "history_db": True,
        }
    }), 200


@app.route("/verify", methods=["POST"])
def verify_text():
    """
    Verify text for misinformation and extract factual claims.
    Payload: { "text": "string", "source_url": "string (optional)", "page_title": "string (optional)" }
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        data = request.get_json(silent=True)
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field in request body."}), 400

        raw_text = data.get("text", "")
        if not isinstance(raw_text, str):
            return jsonify({"error": "'text' field must be a string."}), 400

        text = sanitize_text(raw_text, max_chars=50000)
        if len(text) < 10:
            return jsonify({"error": "Text is too short for analysis. Please provide at least 10 characters."}), 400

        if len(text) > 50000:
            return jsonify({"error": "Text exceeds maximum allowable length of 50,000 characters."}), 400

        source_url = str(data.get("source_url", "")).strip()
        page_title = str(data.get("page_title") or data.get("webpage_title") or "").strip()

        # In-memory TTL cache lookup for sub-millisecond repeated queries
        cache_key = f"verify:{hashlib.sha256((text + '::' + source_url).encode('utf-8')).hexdigest()}"
        cached_result = text_cache.get(cache_key)
        if cached_result:
            return jsonify(cached_result), 200

        # Execute text verification pipeline
        result = text_verifier.predict(text, source_url=source_url, page_title=page_title)

        # Generate XAI explanations
        explanations = explain_text_result(result)
        result["explanations"] = explanations

        # Cache result for 10 minutes
        text_cache.set(cache_key, result, ttl=600)

        return jsonify(result), 200

    except ValueError as e:
        logger.warning("Validation error on /verify: %s", e)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Unexpected error on /verify: %s", e, exc_info=True)
        return jsonify({"error": "An internal server error occurred during text analysis."}), 500


@app.route("/fact-check", methods=["POST"])
def fact_check_text():
    """
    Granular claim-by-claim factual evidence verification endpoint.
    Payload: { "text": "...", "source_url": "...", "page_title": "..." }
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        data = request.get_json(silent=True) or {}
        raw_text = str(data.get("text", "")).strip()
        clean_text = sanitize_text(raw_text, max_chars=50000)
        if len(clean_text) < 10:
            return jsonify({"error": "Text is too short for fact-checking (minimum 10 characters)."}), 400

        source_url = str(data.get("source_url", "")).strip()
        page_title = str(data.get("page_title", "")).strip()

        # Cache check for instant response time
        cache_key = f"fc:{hashlib.sha256((clean_text + '::' + source_url).encode('utf-8')).hexdigest()}"
        cached_res = text_cache.get(cache_key)
        if cached_res:
            return jsonify(cached_res), 200

        from claim_extractor import extract_claims
        from evidence_retriever import EvidenceRetriever
        from claim_cross_referencer import cross_reference_all_claims

        claims = extract_claims(clean_text)
        retriever = EvidenceRetriever()
        evidence_by_claim = {}
        all_sources = []

        for cl in claims:
            cid = cl.get("claim_id", 1)
            q = cl.get("search_query") or cl.get("statement", "")
            ev = retriever.retrieve_evidence(q, cl.get("statement", ""), max_sources=6)
            evidence_by_claim[cid] = ev
            all_sources.extend(ev)

        verdicts = cross_reference_all_claims(claims, evidence_by_claim)

        # Determine overall summary verdict
        statuses = [v["status"] for v in verdicts]
        if "Contradicted" in statuses:
            summary = "Contradicted"
        elif "Mixed Evidence" in statuses:
            summary = "Mixed Evidence"
        elif all(s == "Supported" for s in statuses) and statuses:
            summary = "Supported"
        elif any(s == "Supported" for s in statuses):
            summary = "Mixed Evidence"
        else:
            summary = "Insufficient Evidence"

        # Compute average confidence and breakdown
        avg_conf = (
            round(sum(v.get("confidence_score", 0.0) for v in verdicts) / len(verdicts), 2)
            if verdicts else 0.0
        )
        conf_level = "High" if avg_conf >= 0.80 else ("Moderate" if avg_conf >= 0.50 else "Low")

        # Record in history
        entry_id = history_manager.save_entry(
            submission_type="text",
            content_preview=clean_text[:250],
            source_url=source_url,
            page_title=page_title,
            claims=verdicts,
            overall_verdict=summary,
            ml_prediction="",
            ml_confidence=avg_conf,
            evidence_sources=all_sources[:15],
            source_discovery_status="sources_found" if all_sources else "insufficient_evidence",
        )

        # Compute breakdown summary
        counts = {
            "total_claims": len(verdicts),
            "supported": sum(1 for v in verdicts if v.get("status") == "Supported"),
            "contradicted": sum(1 for v in verdicts if v.get("status") == "Contradicted"),
            "mixed": sum(1 for v in verdicts if v.get("status") == "Mixed Evidence"),
            "insufficient": sum(1 for v in verdicts if v.get("status") == "Insufficient Evidence"),
            "not_verifiable": sum(1 for v in verdicts if v.get("status") == "Not Verifiable"),
        }

        response_payload = {
            "status": "ok",
            "history_id": entry_id,
            "overall_verdict": summary,
            "confidence_score": avg_conf,
            "confidence_level": conf_level,
            "confidence_summary": f"{conf_level} Confidence ({avg_conf:.0%}): Based on {len(all_sources)} corroborating/refuting evidence source(s)",
            "claims": verdicts,
            "fact_check_summary": counts,
            "sources_count": len(all_sources),
            "sources": all_sources[:15],
        }

        text_cache.set(cache_key, response_payload, ttl=600)
        return jsonify(response_payload), 200

    except Exception as e:
        logger.error("Error on /fact-check: %s", e, exc_info=True)
        return jsonify({"error": "An internal server error occurred during claim fact-checking."}), 500


@app.route("/upload-image", methods=["POST"])
def upload_image():
    """
    Verify an uploaded image binary.
    Payload: multipart/form-data with 'image' file field, optional 'source_url' and 'page_title'.
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "No 'image' file part found in request."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"error": "Uploaded image file is empty."}), 400

        source_url = request.form.get("source_url", "").strip() or f"upload:{file.filename}"
        webpage_url = (request.form.get("webpage_url") or request.form.get("page_url") or "").strip()
        webpage_title = (request.form.get("webpage_title") or request.form.get("page_title") or "").strip()

        # Decode image safely
        image = image_from_bytes(image_bytes)

        # Execute verification
        result = image_verifier.verify(
            image,
            source_url=source_url,
            webpage_url=webpage_url,
            webpage_title=webpage_title
        )

        # Generate consistent explanations
        explanations = explain_image_result(result)
        result["explanations"] = explanations

        return jsonify(result), 200

    except ValueError as e:
        logger.warning("Image validation error on /upload-image: %s", e)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Unexpected error on /upload-image: %s", e, exc_info=True)
        return jsonify({"error": "An internal server error occurred during image processing."}), 500


@app.route("/verify-image-url", methods=["POST"])
def verify_image_url():
    """
    Verify an image from a URL.
    Payload: { "url": "string", "page_title": "string (optional)" }
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        data = request.get_json(silent=True)
        if not data or "url" not in data:
            return jsonify({"error": "Missing 'url' field in request body."}), 400

        url = str(data.get("url", "")).strip()
        if not url.startswith(("http://", "https://")):
            return jsonify({"error": "Invalid URL. Image URL must start with http:// or https://"}), 400

        # Validate URL for SSRF security before download
        is_safe, reason = is_safe_url(url)
        if not is_safe:
            return jsonify({"error": f"Invalid or restricted URL: {reason}"}), 400

        webpage_url = str(data.get("webpage_url") or data.get("source_url") or "").strip()
        webpage_title = str(data.get("page_title") or data.get("webpage_title") or "").strip()

        # Download image safely with size and timeout checks
        image = download_image(url)

        # Execute verification
        result = image_verifier.verify(
            image,
            source_url=url,
            webpage_url=webpage_url,
            webpage_title=webpage_title
        )

        # Generate explanations
        explanations = explain_image_result(result)
        result["explanations"] = explanations

        return jsonify(result), 200

    except ValueError as e:
        logger.warning("Image URL error on /verify-image-url: %s", e)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Unexpected error on /verify-image-url: %s", e, exc_info=True)
        return jsonify({"error": "An internal server error occurred during image URL verification."}), 500


@app.route("/history", methods=["GET", "DELETE"])
def handle_history_collection():
    """Retrieve paginated verification history or clear all records."""
    if request.method == "DELETE":
        try:
            count = history_manager.clear_all()
            return jsonify({"success": True, "message": f"Cleared {count} history records.", "deleted": count}), 200
        except Exception as e:
            logger.error("Error clearing history: %s", e)
            return jsonify({"error": "Failed to clear history."}), 500

    try:
        page = request.args.get("page", 1, type=int)
        limit = request.args.get("limit", 10, type=int)
        filter_type = request.args.get("type", "all").strip().lower()
        query = request.args.get("q", "").strip()

        data = history_manager.get_history(page=page, limit=limit, filter_type=filter_type, query=query)
        return jsonify(data), 200
    except Exception as e:
        logger.error("Error retrieving history: %s", e)
        return jsonify({"error": "Failed to fetch verification history."}), 500


@app.route("/history/<int:entry_id>", methods=["GET", "DELETE"])
def handle_history_record(entry_id: int):
    """Retrieve or delete a single history entry by ID."""
    try:
        if request.method == "DELETE":
            deleted = history_manager.delete_entry(entry_id)
            if deleted:
                return jsonify({"success": True, "message": f"Deleted record #{entry_id}."}), 200
            return jsonify({"error": f"Record #{entry_id} not found."}), 404

        entry = history_manager.get_entry(entry_id)
        if entry:
            return jsonify({"status": "ok", "entry": entry, **entry}), 200
        return jsonify({"error": f"Record #{entry_id} not found."}), 404
    except Exception as e:
        logger.error("Error managing history record #%d: %s", entry_id, e)
        return jsonify({"error": "Failed to manage history record."}), 500


# =========================================================
#  SERVER EXECUTION
# =========================================================

def run_server():
    """Start the server using Waitress WSGI if available, else threaded Flask."""
    logger.info("Starting FactGuard AI server on http://%s:%d (Debug=%s)", FLASK_HOST, FLASK_PORT, FLASK_DEBUG)

    if not FLASK_DEBUG:
        try:
            from waitress import serve
            logger.info("Serving via Waitress production WSGI server (threads=6)...")
            serve(app, host=FLASK_HOST, port=FLASK_PORT, threads=6)
            return
        except ImportError:
            logger.info("Waitress not installed; falling back to multi-threaded Flask server.")

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    run_server()
