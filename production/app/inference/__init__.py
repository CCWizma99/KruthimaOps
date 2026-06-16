"""
Inference Layer — Public Interface
Exposes: infer(features_dict) -> float
         load_artifacts() -> None
         get_model_metadata() -> dict
         get_district_reference() -> dict
The API orchestrator calls ONLY these functions.
"""
from app.inference.v1000_engine import infer, load_artifacts, get_model_metadata, get_district_reference

__all__ = ["infer", "load_artifacts", "get_model_metadata", "get_district_reference"]
