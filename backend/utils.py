
import io
import re
import base64
import logging
import urllib.parse
from typing import List, Dict, Any

import requests
import feedparser
from PIL import Image, UnidentifiedImageError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("FactGuard.Utils")

# Common English stop words for RSS query distillation
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "says", "said", "breaking", "urgent", "report", "reports", "according",
    "recent", "statements", "statement"
}


def preprocess_text(text: str) -> str:
    """Clean and normalize text for ML processing."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_search_keywords(text: str, max_words: int = 8) -> str:
    """
    Extract salient search terms by removing boilerplate and stopwords,
    providing far better Google News RSS and Fact Check query relevance.
    """
    clean = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = clean.split()
    salient = [w for w in tokens if w not in STOPWORDS and len(w) > 2]
    
    # If filtering removed too much, fall back to first distinct tokens
    if len(salient) < 3:
        seen = set()
        salient = [w for w in tokens if not (w in seen or seen.add(w))]
    
    selected = salient[:max_words]
    return "+".join(selected) if selected else "news"


def download_image(url: str, timeout: int = 10, max_size_bytes: int = 15 * 1024 * 1024) -> Image.Image:
    """Download an image from a URL safely with SSRF protection, size checks and timeout."""
    if not url or not url.startswith(("http://", "https://")):
        raise ValueError("Invalid URL format. Image URL must start with http:// or https://")

    from security import is_safe_url
    is_safe, reason = is_safe_url(url)
    if not is_safe:
        raise ValueError(f"Blocked unsafe or restricted URL: {reason}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise ValueError(f"Connection timed out while fetching image from {url[:60]}...")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to download image: {str(e)}")

    content_type = response.headers.get("Content-Type", "").lower()
    # Accept valid image types or stream/octet-stream if URL looks like an image
    if content_type and not (content_type.startswith("image/") or "octet-stream" in content_type):
        raise ValueError(f"The URL returned non-image content (Content-Type: {content_type}). Please provide a direct link to an image.")

    content = bytearray()
    for chunk in response.iter_content(chunk_size=65536):
        content.extend(chunk)
        if len(content) > max_size_bytes:
            raise ValueError(f"Image exceeds maximum allowable size ({max_size_bytes // (1024 * 1024)}MB).")

    try:
        image = Image.open(io.BytesIO(content))
        image.load()
        return image.convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("The provided file could not be recognized as a valid image format.")
    except Exception as e:
        raise ValueError(f"Corrupt or unreadable image data: {str(e)}")


def image_from_bytes(file_bytes: bytes) -> Image.Image:
    """Convert raw upload bytes to a PIL Image safely."""
    if not file_bytes:
        raise ValueError("No image data provided.")
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
        return image.convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("Uploaded file is not a supported image format (JPEG, PNG, WebP, etc.).")
    except Exception as e:
        raise ValueError(f"Could not decode image: {str(e)}")


def pil_to_base64_jpeg(image: Image.Image, quality: int = 85) -> str:
    """Safely convert a PIL Image into a base64-encoded JPEG string."""
    buffered = io.BytesIO()
    img_rgb = image.convert("RGB")
    img_rgb.save(buffered, format="JPEG", quality=quality)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def compute_text_similarity(text1: str, texts: Any) -> Any:
    """Compute TF-IDF cosine similarity between text1 and texts (str or list of str)."""
    if not text1 or not text1.strip():
        return 0.0 if isinstance(texts, str) else ([0.0] * len(texts) if texts else [])

    if not texts:
        return 0.0 if isinstance(texts, str) else []
    is_single = isinstance(texts, str)
    text_list = [texts] if is_single else list(texts)
    if not text_list:
        return 0.0 if is_single else []

    all_texts = [text1] + text_list
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        res = [float(max(0.0, min(1.0, s))) for s in similarities]
        return res[0] if is_single else res
    except Exception as e:
        logger.debug("Cosine similarity calculation error: %s", e)
        return 0.0 if is_single else [0.0] * len(text_list)



def fetch_google_news_rss(query: str, max_results: int = 15) -> List[Dict[str, Any]]:
    """Fetch relevant news articles from Google News RSS using keyword extraction."""
    from config import GOOGLE_NEWS_RSS_URL

    search_query = extract_search_keywords(query, max_words=8)
    url = GOOGLE_NEWS_RSS_URL.format(query=search_query)

    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:max_results]:
            raw_summary = entry.get("summary", "")
            # Strip HTML tags and bare URLs from summary to avoid cosine similarity dilution
            clean_summary = re.sub(r'<[^>]+>', ' ', raw_summary)
            clean_summary = re.sub(r'https?://\S+', ' ', clean_summary)
            clean_summary = re.sub(r'\s+', ' ', clean_summary).strip()

            articles.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "summary": clean_summary,
                "published": entry.get("published", ""),
            })
        return articles
    except Exception as e:
        logger.warning("Google News RSS fetch failed: %s", e)
        return []


def fetch_factcheck_results(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Query Google Fact Check Tools API for related claims with graceful fallback."""
    from config import GOOGLE_FACTCHECK_API_KEY, GOOGLE_FACTCHECK_URL

    if not GOOGLE_FACTCHECK_API_KEY:
        logger.debug("Fact Check API key not configured. Skipping fact check lookup.")
        return []

    params = {
        "key": GOOGLE_FACTCHECK_API_KEY,
        "query": query[:200],
        "languageCode": "en",
    }

    try:
        response = requests.get(GOOGLE_FACTCHECK_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for claim in data.get("claims", [])[:max_results]:
            reviews = claim.get("claimReview", [])
            review = reviews[0] if reviews else {}
            results.append({
                "claim": claim.get("text", ""),
                "claimant": claim.get("claimant", "Unknown"),
                "rating": review.get("textualRating", ""),
                "url": review.get("url", ""),
                "publisher": review.get("publisher", {}).get("name", "Unknown"),
            })
        return results
    except Exception as e:
        logger.warning("Fact Check API lookup error: %s", e)
        return []


def fetch_targeted_factcheck_rss(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """
    Search Google News RSS specifically scoped to accredited fact-checking domains
    (Snopes, PolitiFact, FactCheck.org, Reuters Fact Check, AP News, Lead Stories).
    Discovers existing debunks and verifications with zero API key dependencies.
    """
    from config import GOOGLE_NEWS_RSS_URL
    search_keywords = extract_search_keywords(query, max_words=6)
    if not search_keywords:
        return []

    factcheck_scope = "site:snopes.com OR site:politifact.com OR site:factcheck.org OR site:reuters.com OR site:apnews.com OR site:leadstories.com"
    full_query = f"{search_keywords} ({factcheck_scope})"
    url = GOOGLE_NEWS_RSS_URL.format(query=urllib.parse.quote(full_query))

    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:max_results]:
            raw_summary = entry.get("summary", "")
            clean_summary = re.sub(r'<[^>]+>', ' ', raw_summary)
            clean_summary = re.sub(r'https?://\S+', ' ', clean_summary)
            clean_summary = re.sub(r'\s+', ' ', clean_summary).strip()

            source_name = entry.get("source", {}).get("title") if isinstance(entry.get("source"), dict) else (entry.get("source") or "")
            articles.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "summary": clean_summary,
                "source": source_name,
                "publisher": source_name,
                "published": entry.get("published", ""),
                "is_factcheck_source": True,
            })
        return articles
    except Exception as e:
        logger.warning("Targeted fact-check RSS lookup error: %s", e)
        return []


def fetch_wikipedia_summary(query: str, max_results: int = 2) -> List[Dict[str, Any]]:
    """
    Query Wikipedia OpenSearch & REST summary endpoints to retrieve authoritative,
    zero-key encyclopedic ground-truth summaries (Tier 2 Academic / Primary Reference).
    """
    search_terms = extract_search_keywords(query, max_words=5)
    if not search_terms:
        return []

    clean_search = search_terms.replace("+", " ")
    opensearch_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(clean_search)}&limit={max_results}&namespace=0&format=json"

    headers = {
        "User-Agent": "FactGuardAI/1.3.0 (https://github.com/factguard-ai; verification-bot)"
    }
    results = []
    try:
        resp = requests.get(opensearch_url, headers=headers, timeout=5)
        if resp.status_code != 200:
            return []
        data = resp.json()
        if len(data) >= 4 and data[1]:
            titles = data[1]
            links = data[3]
            for title, link in zip(titles[:max_results], links[:max_results]):
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                try:
                    s_resp = requests.get(summary_url, headers=headers, timeout=4)
                    if s_resp.status_code == 200:
                        s_data = s_resp.json()
                        extract = s_data.get("extract", "")
                        page_url = s_data.get("content_urls", {}).get("desktop", {}).get("page", link)
                        results.append({
                            "title": s_data.get("title", title),
                            "link": page_url,
                            "summary": extract,
                            "publisher": "Wikipedia",
                            "published": s_data.get("timestamp", ""),
                            "tier": 2,
                            "tier_label": "Academic / Reference",
                        })
                except Exception as inner_e:
                    logger.debug("Wikipedia page summary fetch error for '%s': %s", title, inner_e)
    except Exception as e:
        logger.debug("Wikipedia OpenSearch lookup error: %s", e)

    return results


def search_image_on_web(image: Image.Image, image_url: str = "") -> Dict[str, Any]:
    """
    Search image across web via Google Cloud Vision API or SerpApi.
    Priority: Google Vision -> SerpApi -> simulated fallback.
    Never exposes API keys in logs or errors.
    """
    from config import GOOGLE_VISION_API_KEY, SERPAPI_API_KEY

    # 1. Try Google Vision API if key is present
    if GOOGLE_VISION_API_KEY:
        try:
            img_str = pil_to_base64_jpeg(image, quality=85)
            url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
            payload = {
                "requests": [{"image": {"content": img_str}, "features": [{"type": "WEB_DETECTION"}]}]
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                responses = data.get("responses", [])
                if responses and "webDetection" in responses[0]:
                    web_detection = responses[0]["webDetection"]
                    logger.info("Google Vision Web Detection succeeded.")
                    return {
                        "web_entities": web_detection.get("webEntities", []),
                        "full_matching_images": web_detection.get("fullMatchingImages", []),
                        "partial_matching_images": web_detection.get("partialMatchingImages", []),
                        "pages_with_matching_images": web_detection.get("pagesWithMatchingImages", []),
                        "is_simulated": False
                    }
            elif response.status_code == 403:
                logger.warning("Google Vision API error 403 (Check API key or billing). Falling back to SerpApi.")
            else:
                logger.warning("Google Vision API returned status %s. Falling back to SerpApi.", response.status_code)
        except Exception as e:
            logger.warning("Google Vision request failed: %s. Falling back to SerpApi.", e)

    # 2. Try SerpApi if key is present
    if SERPAPI_API_KEY and image_url:
        try:
            params = {
                "engine": "google_lens",
                "url": image_url,
                "api_key": SERPAPI_API_KEY
            }
            res = requests.get("https://serpapi.com/search.json", params=params, timeout=15)
            if res.status_code == 200:
                serp_data = res.json()
                visual_matches = serp_data.get("visual_matches", [])
                pages = [
                    {"url": m.get("link"), "pageTitle": m.get("title")}
                    for m in visual_matches if m.get("link")
                ]
                images = [
                    {"url": m.get("thumbnail")}
                    for m in visual_matches if m.get("thumbnail")
                ]
                logger.info("SerpApi Google Lens succeeded with %d matches.", len(visual_matches))
                return {
                    "web_entities": [],
                    "full_matching_images": images,
                    "partial_matching_images": [],
                    "pages_with_matching_images": pages,
                    "is_simulated": False
                }
            elif res.status_code in (401, 403):
                logger.warning("SerpApi authentication or quota limit reached.")
        except Exception as e:
            logger.warning("SerpApi request failed: %s", e)

    # 3. Fallback when keys are missing or services unavailable
    return {
        "web_entities": [],
        "full_matching_images": [],
        "partial_matching_images": [],
        "pages_with_matching_images": [],
        "is_simulated": True,
        "api_error": "External search APIs unavailable or not configured. Use the Free Web Search button for Google Lens."
    }


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length characters with an ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def extract_image_text_ocr(image: Image.Image) -> str:
    """
    Extract visible text from an image using Google Cloud Vision OCR if configured.
    Falls back gracefully to empty string if API key is absent or request fails.
    """
    from config import GOOGLE_VISION_API_KEY
    if not GOOGLE_VISION_API_KEY:
        return ""

    try:
        img_str = pil_to_base64_jpeg(image, quality=85)
        url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
        payload = {
            "requests": [{
                "image": {"content": img_str},
                "features": [{"type": "TEXT_DETECTION"}]
            }]
        }
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            responses = data.get("responses", [])
            if responses and "textAnnotations" in responses[0]:
                annotations = responses[0]["textAnnotations"]
                if annotations:
                    full_text = annotations[0].get("description", "").strip()
                    logger.info("Google Vision OCR extracted %d characters from image.", len(full_text))
                    return full_text
        elif response.status_code == 403:
            logger.warning("Google Vision OCR returned 403 (Verify API key and Cloud Vision enabled).")
        else:
            logger.warning("Google Vision OCR returned status code %s.", response.status_code)
    except Exception as e:
        logger.warning("Google Vision OCR error: %s", e)

    return ""
