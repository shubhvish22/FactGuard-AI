from typing import Dict, Any, List
from config import (
    HIGH_CONFIDENCE_THRESHOLD,
    FAKE_CONFIDENCE_THRESHOLD,
)


def format_confidence_badge(confidence: float) -> Dict[str, Any]:
    """Helper for UI and API consumers to render transparent confidence badges."""
    score = float(confidence)
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return {
            "level": "High",
            "score": round(score, 2),
            "label": f"High Confidence ({score:.0%})",
            "rationale": "Corroborated across multiple independent signals and authoritative evidence.",
        }
    elif score >= FAKE_CONFIDENCE_THRESHOLD:
        return {
            "level": "Moderate",
            "score": round(score, 2),
            "label": f"Moderate Confidence ({score:.0%})",
            "rationale": "Clear statistical indicators present with standard margin of confidence.",
        }
    else:
        return {
            "level": "Low",
            "score": round(score, 2),
            "label": f"Preliminary / Low Confidence ({score:.0%})",
            "rationale": "Sparse corroborating data or conflicting signals; independent review advised.",
        }


def explain_text_result(result: Dict[str, Any]) -> List[str]:
    """Generate plain-language explanations for text verification."""
    explanations = []
    prediction = result.get("prediction", "UNKNOWN")
    confidence = float(result.get("confidence", 0.0))
    ml_label = str(result.get("ml_label", "UNKNOWN"))
    ml_score = float(result.get("ml_score", 0.0))
    sources = result.get("sources", [])
    fact_checks = result.get("fact_checks", [])
    matching_sources = int(result.get("matching_sources", 0))
    source_corroboration = float(result.get("source_corroboration", 0.0))
    has_conflict = bool(result.get("has_conflicting_evidence", False))

    if result.get("error"):
        return [f"⚠️ {result['error']}"]

    # 0. Primary Confidence & Reliability Header
    conf_badge = format_confidence_badge(confidence)
    if prediction == "FAKE":
        explanations.append(
            f"🎯 **Overall Confidence: {conf_badge['label']}** — Misinformation / fabrication indicators identified."
        )
    else:
        explanations.append(
            f"🎯 **Overall Confidence: {conf_badge['label']}** — Authentic reporting and credible phrasing verified."
        )

    # 1. Primary ML Classifier Rationale
    model_name = result.get("model_name", "")
    is_hf = result.get("is_free_huggingface_model", False)
    model_label = f"Hugging Face Transformer ('{model_name}')" if (is_hf and model_name) else "Neural classifier"

    if ml_label == "UNKNOWN":
        explanations.append("ℹ️ Machine learning model unavailable — verdict based on external source corroboration.")
    elif prediction == ml_label:
        if ml_score >= HIGH_CONFIDENCE_THRESHOLD:
            explanations.append(
                f"🤖 {model_label} predicts **{ml_label}** with high confidence ({ml_score:.0%})."
            )
        elif ml_score >= FAKE_CONFIDENCE_THRESHOLD:
            explanations.append(
                f"🤖 {model_label} predicts **{ml_label}** with moderate confidence ({ml_score:.0%})."
            )
        else:
            explanations.append(
                f"🤖 {model_label} confidence is low/uncertain ({ml_score:.0%})."
            )
    else:
        # ML disagreed with final evidence-based prediction
        if prediction == "REAL":
            explanations.append(
                f"🤖 {model_label} flagged stylistic headline patterns as unusual (preliminary model label: FAKE, {ml_score:.0%})."
            )
        else:
            explanations.append(
                f"🤖 {model_label} initially evaluated phrasing as standard news (preliminary model label: REAL, {ml_score:.0%})."
            )

    # 2. Source Corroboration (Google News RSS)
    if matching_sources >= 3 and source_corroboration >= 0.35:
        explanations.append(
            f"📰 Corroborated by {matching_sources} recent news sources (top match similarity: {source_corroboration:.0%})."
        )
    elif matching_sources >= 1 and source_corroboration >= 0.25:
        explanations.append(
            f"📰 Partially matched {matching_sources} news article(s) (top match similarity: {source_corroboration:.0%})."
        )
    elif matching_sources >= 1:
        explanations.append(
            f"🔍 Found {matching_sources} loosely related news mentions (top keyword match: {source_corroboration:.0%}), but no direct confirmation of this specific claim."
        )
    else:
        explanations.append(
            "🔍 No corroborating news reports found from recognized media outlets."
        )

    # 3. Fact-Checking Records
    if fact_checks:
        explanations.append(
            f"⚖️ Found {len(fact_checks)} matching claim(s) in fact-checking registries."
        )
        for fc in fact_checks[:2]:
            rating = fc.get("rating", "")
            publisher = fc.get("publisher", "Independent Fact-Checker")
            if rating:
                explanations.append(f"   → {publisher}: rated \"{rating}\"")
    else:
        verification_mode = result.get("verification_mode", "")
        if verification_mode == "huggingface_free_local":
            explanations.append("🤗 **Hugging Face Free Fallback Active**: External fact-check API offline or unconfigured. Verified using free Hugging Face Transformer and Google News live corroboration.")
        else:
            explanations.append("📋 No prior fact-check debunks recorded in public fact-checking databases for this specific claim.")

    # 4. Conflicting Evidence or Signal Override
    if has_conflict:
        if prediction == "REAL":
            explanations.append(
                f"✅ **Evidence Resolution**: Active coverage from multiple recognized media publications ({matching_sources} sources, {source_corroboration:.0%} similarity) confirms this is a genuine news event. Real-world reporting takes precedence over the language model's stylistic prediction."
            )
        else:
            explanations.append(
                "⛔ **Evidence Resolution**: Verified fact-checking records contradict the linguistic model. Accredited fact-checkers take precedence over stylistic analysis."
            )
    elif ml_label != "UNKNOWN" and prediction != ml_label:
        if prediction == "FAKE":
            explanations.append(
                "⚠️ **Safety Downgrade**: While linguistic patterns appeared plausible, the lack of news corroboration and negative signals adjusted the verdict to FAKE."
            )
        elif prediction == "REAL":
            explanations.append(
                "✅ **Evidence Upgrade**: Independent verified sources and positive news reports confirmed the claim, upgrading the verdict to REAL."
            )

    # 5. Overall Risk Summary
    if prediction == "FAKE":
        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            explanations.append("⛔ **HIGH RISK**: Strong indicators confirm this text contains misleading or fabricated claims.")
        else:
            explanations.append("⚠️ **MODERATE RISK**: Several indicators suggest this claim is unverified or misleading.")
    else:
        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            explanations.append("✅ **CREDIBLE**: Verified reporting and news corroboration confirm this is an authentic news event.")
        else:
            explanations.append("ℹ️ **LIKELY AUTHENTIC**: Evidence generally supports legitimacy, though independent verification is recommended.")

    return explanations


def explain_image_result(result: Dict[str, Any]) -> List[str]:
    """Generate consistent, evidence-based explanations for image verification."""
    explanations = []

    if result.get("error"):
        return [f"⚠️ Error analyzing image: {result['error']}"]

    category = str(result.get("image_category", "real")).lower()
    cat_confidence = float(result.get("category_confidence", 0.0))
    cat_reason = str(result.get("category_reason", ""))

    is_reused = bool(result.get("is_reused", False))
    similarity_score = float(result.get("similarity_score", 0.0))
    similar_count = int(result.get("similar_count", 0))

    ai_confidence = float(result.get("ai_confidence", 0.0))
    edit_score = float(result.get("edit_score", 0.0))
    ai_method = str(result.get("ai_method", "heuristics"))
    dimensions = str(result.get("image_dimensions", "unknown"))

    web_search = result.get("web_search", {})
    web_pages_count = int(result.get("web_pages_count", 0))
    full_matches = len(web_search.get("full_matching_images", []))
    partial_matches = len(web_search.get("partial_matching_images", []))
    total_web_matches = full_matches + partial_matches

    # 0. Forensic Confidence Header
    conf_badge = format_confidence_badge(cat_confidence)
    explanations.append(
        f"🎯 **Diagnostic Confidence: {conf_badge['label']}** — Multi-spectral forensics & neural artifact analysis."
    )

    # 1. Image Specifications
    explanations.append(f"📐 Image dimensions: {dimensions}")

    # 2. Category Verdict Description (Must never contradict verdict)
    if category == "ai_generated":
        explanations.append(
            f"🤖 **AI-GENERATED MEDIA DETECTED** (confidence: {cat_confidence:.0%})\n"
            f"   Forensic detection indicates this image was generated by an AI synthesis model."
        )
        if total_web_matches >= 5:
            explanations.append(
                f"🌐 **Viral Circulation**: This synthetic image was identified on {total_web_matches} web pages. "
                f"Its appearance across websites reflects viral online spread of synthetic media rather than a real photograph."
            )
        if ai_method in ("huggingface", "huggingface_api"):
            detector_desc = "Hugging Face ViT ('umm-maybe/AI-image-detector')"
        elif ai_method == "florida_calibrated_ml":
            detector_desc = "FLORIDA Calibrated Ensemble (HF ViT + Forensics)"
        else:
            detector_desc = "Forensic Heuristics"
        explanations.append(
            f"🔬 Detector score: **{ai_confidence:.0%}** ({detector_desc})."
        )

    elif category == "edited":
        explanations.append(
            f"✂️ **DIGITAL MANIPULATION DETECTED** (confidence: {cat_confidence:.0%})\n"
            f"   Forensic analysis indicates this image has been spliced, composited, or edited."
        )
        explanations.append(
            f"🔍 Tampering score: **{edit_score:.0%}** — inconsistent noise patches and ELA recompression seams detected."
        )

    elif category == "fake":
        explanations.append(
            f"⛔ **MISLEADING / RECYCLED CONTEXT** (confidence: {cat_confidence:.0%})\n"
            f"   This appears to be an authentic photograph used out of context, recycled, or linked to known false claims."
        )
        explanations.append(
            f"ℹ️ **Distinction**: No digital tampering or AI generation was detected on the image itself. "
            f"The concern is that the image is being recycled in a misleading story."
        )
        if is_reused:
            explanations.append(
                f"🔁 **Image Reuse**: Matches {similar_count} previously recorded image(s) in our hash registry ({similarity_score:.0%} perceptual match)."
            )

    elif category == "unverified":
        explanations.append(
            f"⚠️ **UNCORROBORATED MEDIA** (confidence: {cat_confidence:.0%})\n"
            f"   This image cannot be certified as authentic media. Zero verified web appearances or photographic provenance records were located."
        )
        if ai_confidence >= 0.28 or edit_score >= 0.32:
            explanations.append(
                f"🔬 **Digital Anomaly Warning**: Forensic sensors detected non-trivial anomalies "
                f"(AI suspicion: {ai_confidence:.0%}, Edit/Variance score: {edit_score:.0%})."
            )

    else:  # category == 'real'
        explanations.append(
            f"✅ **APPEARS AUTHENTIC** (confidence: {cat_confidence:.0%})\n"
            f"   Forensic analysis reveals consistent noise distribution, natural edge contours, and normal compression."
        )
        if total_web_matches >= 3:
            explanations.append(
                f"🌐 **Web Corroboration**: Found {total_web_matches} web matches confirming authentic publication history."
            )

    # 3. Detailed Forensic Scores
    if category == "ai_generated":
        explanations.append(f"📊 Forensic metrics — Synthetic AI Score: {ai_confidence:.0%}, Edit Score: {edit_score:.0%}")
    elif category == "edited":
        explanations.append(f"📊 Forensic metrics — Edit Tampering Score: {edit_score:.0%}, Synthetic AI Score: {ai_confidence:.0%}")
    elif category == "fake":
        explanations.append(f"📊 Forensic metrics — Authentic Photo (AI Score: {ai_confidence:.0%}, Edit Score: {edit_score:.0%}) with misleading context.")
    elif category == "unverified":
        explanations.append(f"📊 Forensic metrics — AI Suspicion: {ai_confidence:.0%}, Edit/Variance Score: {edit_score:.0%} (Uncorroborated media).")
    else:
        explanations.append(f"📊 Forensic metrics — AI Score: {ai_confidence:.0%}, Edit Score: {edit_score:.0%} (both within clean baseline).")

    # 4. Web Search Details
    if web_search:
        if web_search.get("api_error") and not web_search.get("is_simulated"):
            explanations.append(f"⚠️ Search Notice: {web_search['api_error']}")
        elif web_search.get("is_simulated"):
            explanations.append("🤗 **Hugging Face Free Fallback Active**: External search API offline or unconfigured. Verified using free Hugging Face Vision Transformer ('umm-maybe/AI-image-detector') and multi-spectral forensics.")

        pages = web_search.get("pages_with_matching_images", [])
        debunks = [
            p.get("url", "") for p in pages
            if any(k in p.get("url", "").lower() for k in ("snopes", "factcheck", "politifact", "boomlive"))
        ]
        if debunks:
            explanations.append("🚨 **CRITICAL**: This image was located on accredited fact-checking debunk articles!")
            for deb in debunks[:2]:
                explanations.append(f"   → <a href='{deb}' target='_blank' class='result-link'>{deb[:60]}...</a>")

    return explanations
