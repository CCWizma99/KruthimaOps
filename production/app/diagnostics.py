import os
import sqlite3
import logging
from typing import Dict, Any, List

from app import config

logger = logging.getLogger(__name__)

def run_diagnostics() -> Dict[str, Any]:
    results = {
        "overall_status": "Healthy",
        "checks": []
    }
    
    # 1. Artifact Integrity
    artifacts_status = "Healthy"
    missing_artifacts = []
    base_dir = os.path.join(config.MODELS_BASE_DIR, config.MODEL_VERSION)
    required_artifacts = [
        "cat1.cbm", "cat2.cbm", "catrmse.cbm",
        "xgb1.json", "xgb2.json", "lgb1.txt",
        "feature_lists.json", "model_metadata.json",
        "posthoc.json", "preprocessing.pkl",
        "stacker.json", "te_maps.pkl"
    ]
    
    for filename in required_artifacts:
        if not os.path.isfile(os.path.join(base_dir, filename)):
            missing_artifacts.append(filename)
            
    # Check district reference
    if not os.path.isfile(config.DISTRICT_REFERENCE_PATH):
        missing_artifacts.append("district_reference.json (data folder)")

    if missing_artifacts:
        artifacts_status = "Error"
        results["overall_status"] = "Error"
        artifact_msg = f"Missing {len(missing_artifacts)} files: {', '.join(missing_artifacts[:3])}..."
    else:
        artifact_msg = "All 12 ML artifacts and data references are present."

    results["checks"].append({
        "name": "ML Artifact Integrity",
        "status": artifacts_status,
        "message": artifact_msg
    })

    # 2. Database Connectivity
    db_status = "Healthy"
    try:
        # We rely on SQLite logic matching the adapter
        with sqlite3.connect(config.SQLITE_DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
        db_msg = f"Successfully connected to {config.SQLITE_DB_PATH} and ran query."
    except Exception as e:
        db_status = "Error"
        results["overall_status"] = "Error"
        db_msg = f"Database connection failed: {e}"
        
    results["checks"].append({
        "name": "Database Connectivity",
        "status": db_status,
        "message": db_msg
    })

    # 3. Environment variables / Tokens
    env_status = "Healthy"
    env_msg = []
    if not config.GEMINI_API_KEY:
        env_status = "Warning"
        env_msg.append("GEMINI_API_KEY missing.")
        if results["overall_status"] == "Healthy":
            results["overall_status"] = "Warning"
    else:
        env_msg.append("GEMINI_API_KEY is set.")
        
    if not os.getenv("CESIUM_ION_TOKEN"):
        env_status = "Warning"
        env_msg.append("CESIUM_ION_TOKEN missing.")
        if results["overall_status"] == "Healthy":
            results["overall_status"] = "Warning"
    else:
        env_msg.append("CESIUM_ION_TOKEN is set.")

    results["checks"].append({
        "name": "Environment Tokens",
        "status": env_status,
        "message": " | ".join(env_msg)
    })

    # 4. Gemini API Connectivity
    gemini_status = "Healthy"
    if config.GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.GEMINI_API_KEY)
            client = genai.GenerativeModel(config.GEMINI_MODEL)
            # A very lightweight prompt
            response = client.generate_content("Reply with the exact word 'OK'.")
            if "OK" in response.text.upper():
                gemini_msg = "Successfully pinged Gemini API and received valid response."
            else:
                gemini_status = "Warning"
                gemini_msg = f"Unexpected response from Gemini API: {response.text}"
                if results["overall_status"] == "Healthy":
                    results["overall_status"] = "Warning"
        except Exception as e:
            gemini_status = "Error"
            results["overall_status"] = "Error"
            gemini_msg = f"Failed to connect to Gemini API: {e}"
    else:
        gemini_status = "Skipped"
        gemini_msg = "Skipped Gemini connectivity check because key is missing."

    results["checks"].append({
        "name": "Gemini Connectivity",
        "status": gemini_status,
        "message": gemini_msg
    })

    return results
