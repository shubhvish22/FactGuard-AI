
import re
import logging
from typing import List, Dict, Any, Optional

from config import GEMINI_API_KEY
from utils import extract_search_keywords

logger = logging.getLogger("FactGuard.ClaimExtractor")

# Patterns indicating subjective non-factual claims (opinions, personal musings)
_OPINION_MARKERS = [
    r"^\s*i\s+(think|feel|believe|suppose|guess|hope|wish)\b",
    r"^\s*in\s+my\s+(opinion|view|humble\s+opinion)\b",
    r"^\s*it\s+seems\s+to\s+me\b",
    r"^\s*many\s+people\s+(feel|hope|fear)\b",
    r"^\s*what\s+do\s+you\s+think\b",
]

# Patterns for questions and conversational greetings
_CONVERSATIONAL_MARKERS = [
    r"^\s*(hello|hi|hey|good\s+morning|good\s+evening|welcome|dear\s+friends)\b",
    r"^\s*thanks?\s+(you|a\s+lot|very\s+much)?\b",
    r"^\s*(please\s+like|subscribe|share\s+this|follow\s+me)\b",
]

# Coordinating and adverbial conjunctions that split compound claims
_COMPOUND_SPLIT_REGEX = re.compile(
    r"\b(?:meanwhile|furthermore|additionally|moreover|on\s+the\s+other\s+hand|in\s+contrast|simultaneously)\b",
    re.IGNORECASE
)


class FactualClaim:
    """Represents a single extracted factual proposition."""

    def __init__(
        self,
        claim_id: int,
        statement: str,
        search_query: str,
        context: str = "",
        is_verifiable: bool = True,
    ):
        self.claim_id = claim_id
        self.statement = statement.strip()
        self.search_query = search_query.strip()
        self.context = context.strip()
        self.is_verifiable = is_verifiable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "claim": self.statement,
            "search_query": self.search_query,
            "targeted_query": self.search_query,
            "context": self.context,
            "original_context": self.context,
            "is_verifiable": self.is_verifiable,
        }


def extract_claims(text: str) -> List[Dict[str, Any]]:
    """
    Main extraction interface.
    Attempts Gemini structured claim extraction if GEMINI_API_KEY is configured;
    falls back seamlessly to local deterministic NLP decomposition.
    """
    if not text or not isinstance(text, str):
        return []

    clean_text = text.strip()
    if len(clean_text) < 10:
        return []

    # 1. Try Gemini API extraction if key is present
    if GEMINI_API_KEY:
        gemini_claims = _extract_claims_gemini(clean_text)
        if gemini_claims:
            return [c.to_dict() for c in gemini_claims]

    # 2. Local rule-based & syntactic claim extractor (zero-cost, offline fallback)
    local_claims = _extract_claims_local(clean_text)
    return [c.to_dict() for c in local_claims]


def _extract_claims_local(text: str) -> List[FactualClaim]:
    """
    Deterministic rule-based factual proposition extraction.
    Splits text by sentence boundaries and compound conjunctions,
    retaining original statement wording and context.
    """
    # Normalize whitespace and quotes
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Split by standard sentence delimiters (keeping abbreviations in mind)
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\b])', text)

    candidate_statements = []
    for s in raw_sentences:
        s_clean = s.strip()
        if not s_clean:
            continue

        # Check if sentence is compound statement linked by transition adverbs
        subparts = _COMPOUND_SPLIT_REGEX.split(s_clean)
        for part in subparts:
            part = part.strip()
            # Clean leading conjunctions or punctuation
            part = re.sub(r'^[,;:\s\-]+', '', part)
            if len(part) >= 15:
                candidate_statements.append((part, s_clean))

    claims: List[FactualClaim] = []
    claim_id = 1

    for stmt, context in candidate_statements:
        # Check if purely conversational or greeting
        if any(re.search(pat, stmt, re.IGNORECASE) for pat in _CONVERSATIONAL_MARKERS):
            continue

        # Check if question without factual assertion
        if stmt.endswith("?") and not any(k in stmt.lower() for k in ("did", "was", "were", "is it true")):
            continue

        # Filter out subjective opinions from testable factual claims
        if any(re.search(pat, stmt, re.IGNORECASE) for pat in _OPINION_MARKERS):
            continue

        # Generate targeted query: entity + action keywords
        search_query = _generate_claim_query(stmt)

        claims.append(FactualClaim(
            claim_id=claim_id,
            statement=stmt,
            search_query=search_query,
            context=context if context != stmt else "",
            is_verifiable=True,
        ))
        claim_id += 1

    # If splitting yielded nothing but input has length, use whole text as single claim if not opinion/greeting
    if not claims and len(text) >= 10:
        if not any(re.search(pat, text, re.IGNORECASE) for pat in _OPINION_MARKERS) and \
           not any(re.search(pat, text, re.IGNORECASE) for pat in _CONVERSATIONAL_MARKERS):
            claims.append(FactualClaim(
                claim_id=1,
                statement=text,
                search_query=_generate_claim_query(text),
                context="",
                is_verifiable=True,
            ))

    return claims


def _generate_claim_query(statement: str) -> str:
    """
    Generate targeted 4-7 keyword query for evidence retrieval.
    Removes quotes, punctuation, and selects high-signal entities/verbs.
    """
    return extract_search_keywords(statement, max_words=7)


# Public alias for query generation
generate_targeted_query = _generate_claim_query


def _extract_claims_gemini(text: str) -> Optional[List[FactualClaim]]:
    """
    Extract factual claims using Gemini REST API if GEMINI_API_KEY is configured.
    Falls back gracefully if network fails or quota is exhausted.
    """
    try:
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        prompt = (
            "You are a factual claim extraction system. Extract all distinct, verifiable factual claims from the text below.\n"
            "Return JSON in this exact format:\n"
            '{"claims": [{"statement": "exact claim with original wording", "search_query": "concise search query for fact checking", "is_verifiable": true}]}\n'
            "Rules:\n"
            "1. Preserve original wording and context.\n"
            "2. Separate compound sentences into individual factual propositions.\n"
            "3. Do not include subjective opinions as verifiable claims.\n\n"
            f"Text: {text[:4000]}"
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }

        resp = requests.post(url, json=payload, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                raw_json = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                import json
                parsed = json.loads(raw_json)
                claims_list = parsed.get("claims", [])
                results = []
                for i, item in enumerate(claims_list, 1):
                    stmt = item.get("statement", "").strip()
                    if stmt:
                        results.append(FactualClaim(
                            claim_id=i,
                            statement=stmt,
                            search_query=item.get("search_query") or _generate_claim_query(stmt),
                            context="",
                            is_verifiable=item.get("is_verifiable", True),
                        ))
                if results:
                    logger.info("Extracted %d claims via Gemini API.", len(results))
                    return results
    except Exception as e:
        logger.warning("Gemini claim extraction failed (%s); using local NLP extractor.", e)
    return None
