
import os
import logging

logger = logging.getLogger("FactGuard.Config")

# --- Base Directory Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- Safe Environment Loading (.env) ---
def _load_env_configuration():
    """Load .env file if available, trying python-dotenv first, with manual fallback."""
    env_loaded = False
    try:
        from dotenv import load_dotenv
        # Look in backend/ then workspace root
        for path in [os.path.join(BASE_DIR, ".env"), os.path.join(os.path.dirname(BASE_DIR), ".env")]:
            if os.path.isfile(path):
                load_dotenv(path, override=False)
                env_loaded = True
                break
    except ImportError:
        pass

    if not env_loaded:
        # Fallback manual parser for zero-dependency execution
        for path in [os.path.join(BASE_DIR, ".env"), os.path.join(os.path.dirname(BASE_DIR), ".env")]:
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                k, v = k.strip(), v.strip().strip("'\"")
                                if k and k not in os.environ:
                                    os.environ[k] = v
                except Exception:
                    pass

_load_env_configuration()

# --- Model & Database File Paths ---
TFIDF_MODEL_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
LOGREG_MODEL_PATH = os.path.join(MODELS_DIR, "logistic_model.pkl")
FLORIDA_MODEL_PATH = os.path.join(MODELS_DIR, "florida_image_classifier.pkl")
FLORIDA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "FLORIDA"))
DATA_2_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data_2"))
DATA_3_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data_3"))
TEXT_DATASET_DIR = os.path.join(BASE_DIR, "dataset")
HASH_DB_PATH = os.path.join(BASE_DIR, "image_hashes.db")

# --- ML & Verification Thresholds ---
FAKE_CONFIDENCE_THRESHOLD = 0.60   # Above this = confident fake/real
LOW_CONFIDENCE_THRESHOLD = 0.55    # Below this = uncertain prediction
HIGH_CONFIDENCE_THRESHOLD = 0.85   # Above this = high-confidence prediction

# --- Image Forensics Thresholds ---
PHASH_SIMILARITY_THRESHOLD = 10      # Hamming distance <= 10 out of 64 bits (standard 8x8 pHash) indicates similar/reused image
AI_DETECTION_NOISE_THRESHOLD = 50.0  # Laplacian variance threshold
AI_DETECTION_TEXTURE_THRESHOLD = 15.0 # Texture variance threshold
AI_COMPOSITE_THRESHOLD = 0.52        # Composite score threshold for synthetic AI detection
AI_HIGH_CONFIDENCE_THRESHOLD = 0.80  # ViT confidence above this indicates viral AI image

# --- External API Credentials (Read from environment / .env ONLY) ---
# NOTE: Secrets are read from environment / .env only; never hardcoded in source.
GOOGLE_VISION_API_KEY = os.environ.get("VISION_API_KEY", "").strip() 
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "").strip()  
HUGGINGFACE_API_TOKEN = (
    os.environ.get("HUGGINGFACE_API_TOKEN", "") 
    or os.environ.get("HF_TOKEN", "")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
).strip()
if HUGGINGFACE_API_TOKEN:
    os.environ["HF_TOKEN"] = HUGGINGFACE_API_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HUGGINGFACE_API_TOKEN


# --- Hugging Face Free Models ---
HUGGINGFACE_TEXT_MODEL = os.environ.get("HUGGINGFACE_TEXT_MODEL", "jy46604790/Fake-News-Bert-Detect").strip()
HUGGINGFACE_IMAGE_MODEL = os.environ.get("HUGGINGFACE_IMAGE_MODEL", "umm-maybe/AI-image-detector").strip()

# --- Google Fact Check & News RSS ---
GOOGLE_FACTCHECK_API_KEY = os.environ.get("FACTCHECK_API_KEY", "").strip()
GOOGLE_FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

# --- Server Security & Host Settings ---
# Defaults to localhost (127.0.0.1) for secure local execution
FLASK_HOST = os.environ.get("FLASK_HOST", "127.0.0.1").strip()
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5000"))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "false").strip().lower() in ("true", "1", "yes")
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload

# Permitted CORS Origins (Restrict to local extension & local development)
DEFAULT_ORIGINS = [
    r"^chrome-extension://[a-z]+$",
    r"^http://localhost(:\d+)?$",
    r"^http://127\.0\.0\.1(:\d+)?$",
]
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "")
if CORS_ORIGINS:
    CORS_ALLOWED_ORIGINS = [orig.strip() for orig in CORS_ORIGINS.split(",") if orig.strip()]
else:
    CORS_ALLOWED_ORIGINS = DEFAULT_ORIGINS

# --- Source Similarity Settings ---
MIN_SIMILARITY_SCORE = 0.10
MAX_SOURCES_RETURNED = 15

# --- Google Gemini Grounding & API Key (Optional) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# --- Claim Verification Verdict Statuses ---
VERDICT_STATUSES = (
    "Supported",
    "Contradicted",
    "Mixed Evidence",
    "Insufficient Evidence",
    "Not Verifiable",
)

# --- Source Quality Tiers ---
# Tier 1: Official government publications and original documents
TIER_1_TLDS = (".gov", ".mil")
TIER_1_DOMAINS = {
    "who.int", "un.org", "cdc.gov", "nih.gov", "fda.gov", "nasa.gov",
    "rbi.org.in", "pib.gov.in", "eci.gov.in", "isro.gov.in", "whitehouse.gov",
    "gov.uk", "europa.eu", "unicef.org", "worldbank.org", "imf.org"
}

# Tier 2: Primary research, institutional reports, court records, official statistics
TIER_2_TLDS = (".edu", ".ac.uk", ".ac.in")
TIER_2_DOMAINS = {
    "nature.com", "sciencedirect.com", "ncbi.nlm.nih.gov", "thelancet.com",
    "nejm.org", "arxiv.org", "jstor.org", "springer.com", "wiley.com",
    "science.org", "pnas.org", "cell.com", "bmj.com", "oup.com"
}

# Tier 3: Established fact-checking organizations
TIER_3_DOMAINS = {
    "snopes.com", "politifact.com", "factcheck.org", "boomlive.in",
    "altnews.in", "fullfact.org", "afp.com", "apnews.com", "leadstories.com",
    "checkyourfact.com", "vishvasnews.com", "thequint.com", "factly.in"
}

# Tier 4: Reputable news publications
TIER_4_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "wsj.com", "theguardian.com", "washingtonpost.com", "bloomberg.com",
    "ft.com", "thehindu.com", "indianexpress.com", "hindustantimes.com",
    "ndtv.com", "timesofindia.indiatimes.com", "cnbc.com", "economist.com",
    "cbsnews.com", "nbcnews.com", "abcnews.go.com", "aljazeera.com"
}
