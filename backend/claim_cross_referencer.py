
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("FactGuard.CrossReferencer")

# Keywords in fact-checker ratings or article snippets indicating contradiction / debunk
_CONTRADICT_KEYWORDS = {
    "false", "fake", "incorrect", "pants on fire", "misleading", "untrue",
    "hoax", "satire", "debunk", "busted", "fabricated", "myth", "distorted",
    "wrong", "disproven", "no evidence", "refuted", "inaccurate", "unsupported",
    "baseless", "manipulated", "altered", "disinformation", "misinformation",
    "deepfake", "bogus", "unproven", "lacks context", "partly false", "mostly false",
    "half truth", "disputed"
}

# Keywords in fact-checker ratings indicating support / verification
_SUPPORT_KEYWORDS = {
    "true", "correct", "accurate", "verified", "mostly true", "confirmed",
    "legitimate", "proven", "authentic", "fact", "corroborated", "validated",
    "evidence supports", "found true", "official announcement"
}


def cross_reference_claim(
    claim_dict: Dict[str, Any],
    evidence_list: Optional[List[Any]] = None,
    evidence: Optional[List[Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Cross-reference a single factual claim against a list of retrieved evidence sources.
    Returns the standardized claim verdict dictionary.
    """
    from utils import compute_text_similarity

    if evidence_list is None:
        evidence_list = evidence if evidence is not None else kwargs.get("evidence", [])

    statement = (claim_dict.get("statement") or claim_dict.get("claim") or "").strip()
    is_verifiable = claim_dict.get("is_verifiable", True)
    claim_id = claim_dict.get("claim_id", 1)

    # Normalize evidence list to dictionaries
    clean_evidence: List[Dict[str, Any]] = []
    for ev in evidence_list:
        if hasattr(ev, "to_dict"):
            clean_evidence.append(ev.to_dict())
        elif isinstance(ev, dict):
            clean_evidence.append(ev)

    # 1. Check if claim was marked as non-verifiable (opinion, subjective feeling)
    if not is_verifiable:
        return {
            "claim_id": claim_id,
            "claim": statement,
            "status": "Not Verifiable",
            "explanation": "This statement represents a subjective opinion, personal sentiment, or untestable assertion rather than a verifiable factual proposition.",
            "confidence_score": 0.0,
            "confidence_level": "None",
            "confidence_tier": "Unverifiable",
            "confidence_summary": "Unverifiable: Subjective opinion or untestable value assertion.",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "source_quality_assessment": "None (Subjective / Opinion statement)",
            "source_quality": "None (Subjective / Opinion statement)",
            "uncertainty_limitations": "Opinions and values cannot be verified against objective empirical data.",
            "uncertainty_or_limitations": "Opinions and values cannot be verified against objective empirical data.",
            "sources_count": len(clean_evidence),
        }

    # 2. Check if no evidence could be retrieved
    if not clean_evidence:
        return {
            "claim_id": claim_id,
            "claim": statement,
            "status": "Insufficient Evidence",
            "explanation": "Insufficient evidence: No corroborating or refuting evidence could be retrieved from recognized government, academic, fact-checking, or news databases for this specific claim.",
            "confidence_score": 0.15,
            "confidence_level": "Low",
            "confidence_tier": "Low Confidence",
            "confidence_summary": "Low Confidence (15%): Zero corroborating or refuting records found.",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "source_quality_assessment": "Insufficient Data (0 sources retrieved)",
            "source_quality": "Insufficient Data (0 sources retrieved)",
            "uncertainty_limitations": "Absence of evidence does not inherently disprove the claim. Independent investigation or direct documentation is recommended.",
            "uncertainty_or_limitations": "Absence of evidence does not inherently disprove the claim. Independent investigation or direct documentation is recommended.",
            "sources_count": 0,
        }

    supporting: List[Dict[str, Any]] = []
    contradicting: List[Dict[str, Any]] = []

    # 3. Classify evidence into Supporting vs Contradicting
    for ev in clean_evidence:
        rating = ev.get("fact_check_rating", "").lower()
        snippet = ev.get("snippet", "").lower()
        title = ev.get("title", "").lower()
        sim = float(ev.get("similarity_score", 0.0))

        if sim == 0.0 and statement and (title or snippet):
            sim = compute_text_similarity(statement, f"{title} {snippet}")
            ev["similarity_score"] = sim

        # Check fact-checker rating first (most authoritative explicit verdict)
        if rating:
            if any(k in rating for k in _CONTRADICT_KEYWORDS):
                contradicting.append(ev)
                continue
            elif any(k in rating for k in _SUPPORT_KEYWORDS):
                supporting.append(ev)
                continue

        # Check title / snippet for active debunking keywords
        has_debunk_words = any(k in title or k in snippet for k in _CONTRADICT_KEYWORDS)
        if has_debunk_words and (sim >= 0.15 or rating):
            contradicting.append(ev)
        elif sim >= 0.20 or (sim > 0.0 and ev.get("tier", 5) <= 2):
            # High semantic similarity reporting or tier 1/2 corroboration indicates support
            supporting.append(ev)

    # Assess highest source tier among retrieved evidence
    best_tier = min((ev.get("tier", 5) for ev in clean_evidence), default=5)
    tier_labels = {
        1: "Tier 1 (Official Government / Intergovernmental)",
        2: "Tier 2 (Primary Research / Academic Journals)",
        3: "Tier 3 (Accredited Fact-Checking Organizations)",
        4: "Tier 4 (Reputable News Publications)",
        5: "Tier 5 (General Web / Aggregators)",
    }
    quality_assessment = tier_labels.get(best_tier, "General Web")

    # 4. Formulate 5-Status Verdict and Confidence Metrics
    if contradicting and not supporting:
        status = "Contradicted"
        top_debunk = contradicting[0]
        pub = top_debunk.get("publisher", "Independent Source")
        has_accredited = any(ev.get("tier", 5) <= 3 for ev in contradicting)
        confidence_score = 0.94 if has_accredited else (0.86 if len(contradicting) >= 2 else 0.78)
        confidence_level = "High" if confidence_score >= 0.85 else "Moderate"
        confidence_summary = f"{confidence_level} Confidence ({confidence_score:.0%}): Directly refuted by {len(contradicting)} source(s) including {pub}."
        explanation = f"Contradicted by {len(contradicting)} source(s). Verified reporting and fact-check records from {pub} refute this claim."
        uncertainty = "Contradictory evidence is established by accredited sources; further investigation may be required if new primary documentation emerges."

    elif supporting and not contradicting:
        status = "Supported"
        top_sup = supporting[0]
        pub = top_sup.get("publisher", "Recognized Source")
        has_authoritative = any(ev.get("tier", 5) <= 2 for ev in supporting)
        confidence_score = 0.92 if has_authoritative else (0.85 if len(supporting) >= 2 else 0.75)
        confidence_level = "High" if confidence_score >= 0.85 else "Moderate"
        confidence_summary = f"{confidence_level} Confidence ({confidence_score:.0%}): Corroborated by {len(supporting)} source(s) including {pub}."
        explanation = f"Supported by {len(supporting)} source(s). Corroborated by independent coverage and documented reporting from {pub}."
        uncertainty = "Evidence supports this claim based on currently available reporting; historical or statistical revisions may alter details."

    elif supporting and contradicting:
        status = "Mixed Evidence"
        confidence_score = 0.60
        confidence_level = "Moderate"
        confidence_summary = f"Moderate Confidence (60%): Disputed coverage ({len(supporting)} supporting vs {len(contradicting)} contradicting)."
        explanation = f"Conflicting evidence discovered: {len(supporting)} source(s) support while {len(contradicting)} source(s) dispute or contradict aspects of this claim."
        uncertainty = "Conflicting reports exist across reputable publications. The situation may be evolving, disputed, or subject to differing interpretations."

    else:
        status = "Insufficient Evidence"
        confidence_score = 0.35
        confidence_level = "Low"
        confidence_summary = "Low Confidence (35%): General mentions found but insufficient semantic precision."
        explanation = f"Found {len(clean_evidence)} loosely related mention(s), but none directly confirm or refute the specific details of this claim."
        uncertainty = "Retrieved sources do not address the core proposition with sufficient semantic precision. Not classified as false."

    return {
        "claim_id": claim_id,
        "claim": statement,
        "status": status,
        "explanation": explanation,
        "confidence_score": round(confidence_score, 2),
        "confidence_level": confidence_level,
        "confidence_tier": f"{confidence_level} Confidence",
        "confidence_summary": confidence_summary,
        "supporting_evidence": supporting,
        "contradicting_evidence": contradicting,
        "source_quality_assessment": quality_assessment,
        "source_quality": quality_assessment,
        "uncertainty_limitations": uncertainty,
        "uncertainty_or_limitations": uncertainty,
        "sources_count": len(clean_evidence),
    }


def cross_reference_all_claims(
    claims: List[Dict[str, Any]],
    evidence_by_claim: Dict[int, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Cross-reference all extracted claims and return individual verdicts."""
    verdicts = []
    for c in claims:
        cid = c.get("claim_id", 1)
        ev_list = evidence_by_claim.get(cid, [])
        verdict = cross_reference_claim(c, ev_list)
        verdicts.append(verdict)
    return verdicts
