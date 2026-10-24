
import hashlib
import datetime
import logging
import urllib.parse
import concurrent.futures
from typing import List, Dict, Any, Optional

from config import (
    TIER_1_TLDS,
    TIER_1_DOMAINS,
    TIER_2_TLDS,
    TIER_2_DOMAINS,
    TIER_3_DOMAINS,
    TIER_4_DOMAINS,
    GEMINI_API_KEY,
)
from utils import (
    fetch_google_news_rss,
    fetch_factcheck_results,
    fetch_targeted_factcheck_rss,
    fetch_wikipedia_summary,
    compute_text_similarity,
)
from security import is_safe_url
from cache import evidence_cache

logger = logging.getLogger("FactGuard.EvidenceRetriever")


class EvidenceSource:
    """Represents a single retrieved evidence item."""

    def __init__(
        self,
        url: str,
        title: str,
        snippet: str,
        publisher: str = "",
        published_date: str = "",
        fact_check_rating: str = "",
        similarity_score: float = 0.0,
        claim: str = "",
        source_tier: Optional[int] = None,
        **kwargs,
    ):
        self.url = url.strip()
        self.title = title.strip()
        self.snippet = snippet.strip()
        self.claim = claim.strip()
        self.publisher = publisher.strip() or self._extract_publisher_from_url(self.url)
        self.published_date = published_date.strip()
        self.fact_check_rating = fact_check_rating.strip()
        self.similarity_score = float(similarity_score)
        self.retrieval_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

        # Compute tier
        tier_calc, tier_label_calc = self._classify_source_tier(self.url, self.publisher)
        self.tier = source_tier if source_tier is not None else tier_calc
        self.tier_label = tier_label_calc
        self.source_tier = self.tier

    @staticmethod
    def _extract_publisher_from_url(url: str) -> str:
        try:
            parsed = urllib.parse.urlparse(url)
            host = parsed.hostname or ""
            return host.replace("www.", "")
        except Exception:
            return "Web Source"

    @staticmethod
    def _classify_source_tier(url: str, publisher: str) -> tuple[int, str]:
        """Classify source into Quality Tiers 1 through 5."""
        try:
            parsed = urllib.parse.urlparse(url)
            host = (parsed.hostname or "").lower()
            pub_lower = publisher.lower()

            # Tier 1: Official government & intergovernmental bodies
            if any(host.endswith(tld) for tld in TIER_1_TLDS) or any(d in host for d in TIER_1_DOMAINS) or any(k in pub_lower for k in ("gov", "who", "cdc", "un ", "united nations", "nasa")):
                return 1, "Official / Government"

            # Tier 2: Primary research, academia & encyclopedic reference
            if "wikipedia.org" in host or "wikimedia.org" in host or any(host.endswith(tld) for tld in TIER_2_TLDS) or any(d in host for d in TIER_2_DOMAINS):
                return 2, "Academic / Reference"

            # Tier 3: Established fact-checking organizations
            if any(d in host for d in TIER_3_DOMAINS) or any(d in pub_lower for d in ("factcheck", "snopes", "politifact", "boomlive", "altnews", "leadstories", "reuters fact check", "full fact")):
                return 3, "Accredited Fact-Checker"

            # Tier 4: Reputable news publications
            if any(d in host for d in TIER_4_DOMAINS) or any(d in pub_lower for d in ("reuters", "associated press", "ap news", "bbc", "nytimes", "hindu", "the guardian", "wall street journal", "bloomberg")):
                return 4, "Reputable News Media"

        except Exception:
            pass

        return 5, "General Web"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "publisher": self.publisher,
            "published_date": self.published_date,
            "fact_check_rating": self.fact_check_rating,
            "similarity_score": round(self.similarity_score, 4),
            "tier": self.tier,
            "tier_label": self.tier_label,
            "retrieval_timestamp": self.retrieval_timestamp,
        }


# Public alias for EvidenceSource
EvidenceItem = EvidenceSource


class EvidenceRetriever:
    """Coordinates evidence search across fact-checking tools, RSS feeds, and web providers."""

    def __init__(self):
        pass

    def _classify_source_tier(self, url: str, publisher: str = "") -> int:
        tier, _ = EvidenceSource._classify_source_tier(url, publisher)
        return tier

    def _deduplicate_evidence(self, items: List[EvidenceSource]) -> List[EvidenceSource]:
        seen = set()
        deduped = []
        for item in items:
            canon = item.url.split("?")[0].rstrip("/").lower()
            if canon not in seen:
                seen.add(canon)
                deduped.append(item)
        return deduped


    def retrieve_evidence(self, query: str, statement: str, max_sources: int = 8) -> List[Dict[str, Any]]:
        """
        Retrieve and rank evidence for a specific factual claim using concurrent multi-source dispatch:
          1. Google Fact Check Tools API (accredited ratings & debunks)
          2. Targeted Fact-Checking RSS (Snopes, PolitiFact, FactCheck.org, Reuters Fact Check)
          3. Google News RSS (general real-time journalism corroboration)
          4. Wikipedia REST API (encyclopedic / academic primary ground-truth)
          5. Gemini Search Grounding (optional if key configured)
        Results are cached in memory (TTL 15 min) and ranked by Quality Tier and Semantic Relevance.
        """
        if not query and not statement:
            return []

        search_query = query or statement

        # Check in-memory cache first for sub-millisecond retrieval
        cache_key = f"ev:{hashlib.sha256((str(search_query) + '::' + str(statement)).encode('utf-8')).hexdigest()}"
        cached = evidence_cache.get(cache_key)
        if cached is not None:
            logger.debug("Evidence cache HIT for: '%s'", search_query[:40])
            return cached[:max_sources]

        evidence_list: List[EvidenceSource] = []
        seen_urls = set()

        # Concurrently dispatch queries across all providers
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_fc = executor.submit(fetch_factcheck_results, search_query)
            future_target_rss = executor.submit(fetch_targeted_factcheck_rss, search_query, max_results=6)
            future_news_rss = executor.submit(fetch_google_news_rss, search_query, max_results=max_sources * 2)
            future_wiki = executor.submit(fetch_wikipedia_summary, statement or search_query, max_results=2)
            future_gemini = (
                executor.submit(self._retrieve_gemini_grounding, statement, search_query)
                if GEMINI_API_KEY else None
            )

            # 1. Google Fact Check Tools API
            try:
                fc_results = future_fc.result(timeout=3.0) or []
                for fc in fc_results:
                    url = fc.get("url", "")
                    if url and url not in seen_urls:
                        is_safe, _ = is_safe_url(url)
                        if is_safe:
                            seen_urls.add(url)
                            evidence_list.append(EvidenceSource(
                                url=url,
                                title=fc.get("claim", ""),
                                snippet=f"Claim: '{fc.get('claim', '')}' evaluated by {fc.get('publisher', 'Fact-Checker')}. Rating: {fc.get('rating', '')}",
                                publisher=fc.get("publisher", ""),
                                fact_check_rating=fc.get("rating", ""),
                                similarity_score=0.92,
                                source_tier=3,
                            ))
            except Exception as e:
                logger.debug("Fact Check API retrieval error: %s", e)

            # 2. Targeted Fact-Checking RSS (Snopes, PolitiFact, FactCheck.org, etc.)
            try:
                target_articles = future_target_rss.result(timeout=3.0) or []
                if target_articles:
                    titles = [a.get("title", "") for a in target_articles]
                    sims = compute_text_similarity(statement, titles)
                    for article, sim in zip(target_articles, sims):
                        url = article.get("link", "")
                        if url and url not in seen_urls:
                            is_safe, _ = is_safe_url(url)
                            if is_safe:
                                seen_urls.add(url)
                                evidence_list.append(EvidenceSource(
                                    url=url,
                                    title=article.get("title", ""),
                                    snippet=article.get("summary", "") or article.get("title", ""),
                                    publisher=article.get("publisher", "") or "Accredited Fact-Checker",
                                    published_date=article.get("published", ""),
                                    similarity_score=max(0.60, float(sim) + 0.20),
                                    source_tier=3,
                                ))
            except Exception as e:
                logger.debug("Targeted fact-check RSS retrieval error: %s", e)

            # 3. Wikipedia REST Ground-Truth (Tier 2 Academic / Reference)
            try:
                wiki_results = future_wiki.result(timeout=2.5) or []
                for w in wiki_results:
                    url = w.get("link", "")
                    if url and url not in seen_urls:
                        is_safe, _ = is_safe_url(url)
                        if is_safe:
                            seen_urls.add(url)
                            summary_text = w.get("summary", "")
                            sim = compute_text_similarity(statement, f"{w.get('title', '')} {summary_text}")
                            evidence_list.append(EvidenceSource(
                                url=url,
                                title=w.get("title", ""),
                                snippet=summary_text[:350],
                                publisher=w.get("publisher", "Wikipedia"),
                                published_date=w.get("published", ""),
                                similarity_score=max(0.55, float(sim) + 0.15),
                                source_tier=2,
                            ))
            except Exception as e:
                logger.debug("Wikipedia retrieval error: %s", e)

            # 4. Google News RSS Corroboration
            try:
                articles = future_news_rss.result(timeout=3.0) or []
                if articles:
                    titles = [a.get("title", "") for a in articles]
                    sims = compute_text_similarity(statement, titles)
                    for article, sim in zip(articles, sims):
                        url = article.get("link", "")
                        if url and url not in seen_urls:
                            is_safe, _ = is_safe_url(url)
                            if is_safe:
                                seen_urls.add(url)
                                evidence_list.append(EvidenceSource(
                                    url=url,
                                    title=article.get("title", ""),
                                    snippet=article.get("summary", "") or article.get("title", ""),
                                    publisher=article.get("source", "") or article.get("publisher", ""),
                                    published_date=article.get("published", ""),
                                    similarity_score=float(sim),
                                ))
            except Exception as e:
                logger.debug("Google News RSS retrieval error: %s", e)

            # 5. Gemini Search Grounding (Optional)
            if future_gemini:
                try:
                    gemini_evidence = future_gemini.result(timeout=3.5) or []
                    for ge in gemini_evidence:
                        if ge.url and ge.url not in seen_urls:
                            is_safe, _ = is_safe_url(ge.url)
                            if is_safe:
                                seen_urls.add(ge.url)
                                evidence_list.append(ge)
                except Exception as e:
                    logger.debug("Gemini grounding retrieval error: %s", e)

        # Prioritize and rank: Lowest tier number first (Tier 1 > Tier 2 > Tier 3 > Tier 4 > Tier 5),
        # then highest similarity score within the tier.
        evidence_list.sort(key=lambda x: (x.tier, -x.similarity_score))

        result_dicts = [e.to_dict() for e in evidence_list[:max_sources]]

        # Cache valid evidence results for 15 minutes
        if result_dicts:
            evidence_cache.set(cache_key, result_dicts, ttl=900)

        return result_dicts

    def _retrieve_gemini_grounding(self, statement: str, query: str) -> List[EvidenceSource]:
        """Query Gemini with Google Search Grounding to discover authoritative evidence URLs."""
        sources: List[EvidenceSource] = []
        try:
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            prompt = f"Find recent web sources and evidence for the following claim: '{statement}'. Provide the source links and titles."
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "tools": [{"googleSearch": {}}],  # Enable Google Search grounding
            }
            resp = requests.post(url, json=payload, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    metadata = candidates[0].get("groundingMetadata", {})
                    chunks = metadata.get("groundingChunks", [])
                    for chunk in chunks:
                        web = chunk.get("web", {})
                        site_url = web.get("uri", "")
                        site_title = web.get("title", "")
                        if site_url:
                            sources.append(EvidenceSource(
                                url=site_url,
                                title=site_title,
                                snippet=f"Grounding evidence from {site_title}",
                                similarity_score=0.75,
                            ))
        except Exception as e:
            logger.warning("Gemini Grounding error: %s", e)
        return sources
