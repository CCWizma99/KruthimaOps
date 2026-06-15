@echo off
echo Starting FloodGuard SL Development Server (Self-Contained)...
set PYTHONPATH=.
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
