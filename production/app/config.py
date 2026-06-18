"""
Flood Timeline — Application Configuration
Single source of truth for all layer switches.
Change values here to swap providers without touching any other file.
"""
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(BASE_DIR, '.env'))
load_dotenv() # Also try current directory just in case

# ── Model (Reload Triggered) ─────────────────────────────────────────
MODEL_VERSION   = "prod_v1k.2"
MODELS_BASE_DIR = os.path.join(BASE_DIR, "models")

# ── Monitoring ───────────────────────────────────────────────────────
MONITOR_BACKEND = "sqlite"          # options: "sqlite" (extend with "postgres")
SQLITE_DB_PATH  = os.path.join(BASE_DIR, "monitoring.db")

# ── AI Briefing ──────────────────────────────────────────────────────
BRIEFING_PROVIDER = "gemini"        # options: "gemini", "disabled"
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL      = "gemini-3.5-flash"

# ── Data Reference ───────────────────────────────────────────────────
DISTRICT_REFERENCE_PATH = os.path.join(BASE_DIR, "data", "district_reference.json")

# ── API ──────────────────────────────────────────────────────────────
API_TITLE       = "Flood Timeline — Flood Risk Prediction API"
API_DESCRIPTION = "Production ML inference API for flood risk scoring across Sri Lanka"
API_VERSION     = "1.0.0"

# ── Cesium / Map ─────────────────────────────────────────────────────
CESIUM_ION_TOKEN = os.getenv("CESIUM_ION_TOKEN", "")
