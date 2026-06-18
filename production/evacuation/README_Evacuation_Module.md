# FloodGuard SL — Evacuation Safe-Zone Module

This is a standalone evacuation/safe-zone module for FloodGuard SL. Keep it under `production/evacuation/` so it does not conflict with the main FastAPI flood prediction app.

## What it includes

- Leaflet-based safe-zone and evacuation-point map
- Citizen view and admin view
- Add/edit/delete evacuation points with an admin key
- SQLite storage in `data/evac_points.db`
- Seed data for demo points
- Safe-zone ranking API
- Simple evacuation route preview
- SVG export and offline package export
- Service worker tile caching via `sw.js`

## File placement

Place the folder like this:

```text
KruthimaOps/
└── production/
    └── evacuation/
        ├── server.py
        ├── evacuation_presentation.html
        ├── sw.js
        ├── seed_points.json
        ├── requirements.txt
        ├── .env.example
        └── README_Evacuation_Module.md
```

Do not place these files randomly in the main `production/app` folder unless you are converting them into FastAPI routes.

## Setup on Windows PowerShell

```powershell
cd "C:\Users\user\Downloads\KruthimaOps-Final\production\evacuation"
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
notepad .env
python server.py
```

Open:

```text
http://127.0.0.1:5000
```

## Admin key

The admin key is read from `.env`:

```text
FLOODGUARD_ADMIN_KEY=changeme-floodguard
```

Change this before demo. Do not commit `.env` to GitHub.

## API quick tests

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/health
Invoke-RestMethod http://127.0.0.1:5000/api/points
Invoke-RestMethod http://127.0.0.1:5000/api/routes
Invoke-RestMethod http://127.0.0.1:5000/api/rank_safe_zones?origin_lat=7.29\&origin_lng=80.63
```

Add a point:

```powershell
$body = @{
  type = "Safe Zone"
  label = "Test Shelter"
  description = "Demo shelter"
  status = "Open"
  capacity = 200
  lat = 7.2906
  lng = 80.6337
  x = 200
  y = 200
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:5000/api/points" `
  -Method Post `
  -Headers @{"X-Admin-Key"="changeme-floodguard"} `
  -Body $body `
  -ContentType "application/json"
```

## Important notes

- `evacuation.js` was renamed to `sw.js` because the HTML and backend expect `/sw.js`.
- This is a separate Flask server on port `5000`.
- The main FloodGuard prediction app still runs with FastAPI on port `8000`.
- For final integration, this can later be converted into FastAPI routes under `production/app/main.py`.
