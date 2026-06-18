"""
Briefing Layer — Gemini Adapter
Converts a flood risk score + key features into a plain-language
safety advisory using the Gemini API.

Interface: brief(score: float, features: dict) -> str
"""
from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            import google.generativeai as genai
            from app.config import GEMINI_API_KEY, GEMINI_MODEL
            if not GEMINI_API_KEY:
                return None
            genai.configure(api_key=GEMINI_API_KEY)
            _client = genai.GenerativeModel(GEMINI_MODEL)
        except Exception as e:
            logger.warning(f"Gemini client init failed: {e}")
            return None
    return _client


def _risk_label(score: float) -> str:
    if score < 0.25:   return "LOW"
    if score < 0.50:   return "MEDIUM"
    if score < 0.75:   return "HIGH"
    return "EXTREME"


async def brief(score: float, features: dict) -> str:
    """
    Generate a 3-4 sentence plain-language safety briefing asynchronously.
    Returns empty string if Gemini is unavailable.
    """
    client = _get_client()
    if client is None:
        return _fallback_briefing(score, features)

    district     = features.get("district", "this location")
    rainfall     = features.get("rainfall_7d_mm", 0)
    inundation   = features.get("inundation_area_sqm", 0)
    flood_active = features.get("flood_occurrence_current_event", "No")
    risk_label   = _risk_label(score)

    prompt = f"""You are a disaster risk communication officer for Sri Lanka's National Disaster Management Authority.
Generate a concise 3-sentence safety advisory (NO markdown, plain text only) for emergency responders based on:

- Location: {district} district, Sri Lanka
- Flood Risk Score: {score:.4f} out of 1.0 ({risk_label} RISK)
- 7-day accumulated rainfall: {rainfall:.1f} mm
- Current inundation area: {inundation:.0f} sqm
- Active flooding event: {flood_active}

The advisory must:
1. State the risk level and immediate threat clearly.
2. Give ONE specific action recommendation for emergency teams.
3. Mention any critical infrastructure or population concerns.

Keep it under 80 words total. Be direct and actionable."""

    try:
        response = await client.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Gemini generation failed: {e}")
        return _fallback_briefing(score, features)


def _fallback_briefing(score: float, features: dict) -> str:
    """Rule-based fallback when Gemini is unavailable."""
    district   = features.get("district", "this area")
    rainfall   = features.get("rainfall_7d_mm", 0)
    risk_label = _risk_label(score)

    if risk_label == "EXTREME":
        return (
            f"{district} district shows EXTREME flood risk (score: {score:.2f}). "
            f"Immediate evacuation of low-lying zones is recommended. "
            f"Emergency response teams should be deployed. Rainfall of {rainfall:.0f}mm "
            f"over 7 days significantly exceeds safe drainage capacity."
        )
    elif risk_label == "HIGH":
        return (
            f"{district} district is at HIGH flood risk (score: {score:.2f}). "
            f"Pre-emptive evacuation of vulnerable households is advised. "
            f"Authorities should monitor river levels closely given {rainfall:.0f}mm "
            f"7-day rainfall accumulation."
        )
    elif risk_label == "MEDIUM":
        return (
            f"{district} district shows MEDIUM flood risk (score: {score:.2f}). "
            f"Residents near waterways should be on alert. Continue monitoring rainfall and "
            f"drainage conditions. Current accumulation of {rainfall:.0f}mm requires attention."
        )
    else:
        return (
            f"{district} district is at LOW flood risk (score: {score:.2f}). "
            f"No immediate action required. Standard monitoring protocols apply. "
            f"Current 7-day rainfall of {rainfall:.0f}mm is within safe thresholds."
        )
