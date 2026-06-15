"""
Validation Layer — Public Interface
Exposes: validate(payload) -> (clean_data: dict, warnings: list[str])
"""
from app.validation.rules import validate

__all__ = ["validate"]
