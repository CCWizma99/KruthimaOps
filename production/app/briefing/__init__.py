"""
Briefing Layer — Public Interface
Exposes: brief(score, features) -> str
Swap providers by changing BRIEFING_PROVIDER in config.py.
"""
from app.config import BRIEFING_PROVIDER

if BRIEFING_PROVIDER == "gemini":
    from app.briefing.gemini_adapter import brief
elif BRIEFING_PROVIDER == "disabled":
    async def brief(score: float, features: dict) -> str:
        return ""
else:
    raise ValueError(f"Unknown BRIEFING_PROVIDER: {BRIEFING_PROVIDER}")

__all__ = ["brief"]
