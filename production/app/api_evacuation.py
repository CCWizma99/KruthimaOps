from fastapi import APIRouter, Request, HTTPException, Header, Depends
from fastapi.responses import JSONResponse, Response, HTMLResponse
import os
import uuid
from datetime import datetime, timezone
import io
import zipfile
import json
import math
from typing import Optional
from app.database import get_db_cursor

HERE = os.path.dirname(os.path.abspath(__file__))
# Note: we use the existing data directory as requested in the plan
DATA_DIR = os.path.abspath(os.path.join(HERE, '..', '..', 'data'))

# Original frontend files
HTML_FILE = os.path.join(HERE, '..', 'evacuation', 'evacuation_presentation.html')
SW_FILE = os.path.join(HERE, '..', 'evacuation', 'sw.js')

ADMIN_KEY = os.environ.get('FLOODGUARD_ADMIN_KEY', 'changeme-flood-timeline')

router = APIRouter(tags=["Evacuation"])
os.makedirs(DATA_DIR, exist_ok=True)

# ─── DATABASE ────────────────────────────────────────────────────
def init_db():
    import logging
    logger = logging.getLogger("app.api_evacuation")
    logger.info("[Database] Initializing PostgreSQL evacuation schema...")
    with get_db_cursor() as cur:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS points (
                id VARCHAR PRIMARY KEY,
                type VARCHAR,
                label VARCHAR,
                x DOUBLE PRECISION,
                y DOUBLE PRECISION,
                lat DOUBLE PRECISION,
                lng DOUBLE PRECISION,
                description TEXT,
                status VARCHAR,
                capacity INTEGER,
                created_at VARCHAR,
                updated_at VARCHAR
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS hazards (
                id VARCHAR PRIMARY KEY,
                lat DOUBLE PRECISION,
                lng DOUBLE PRECISION,
                radius_m DOUBLE PRECISION,
                note TEXT,
                created_at VARCHAR
            )
        ''')
    logger.info("[Database] Evacuation schema initialized.")

try:
    init_db()
except Exception as e:
    import logging
    logging.getLogger("app.api_evacuation").warning(f"Could not auto-initialize evacuation tables: {e}")

# ─── AUTH ───────────────────────────────────────────────────────
def require_admin(x_admin_key: Optional[str] = Header(None)):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="admin key required")
    return True

# ─── HELPERS ────────────────────────────────────────────────────
def row_to_point(r):
    return {
        'id': r['id'], 'type': r['type'], 'label': r['label'],
        'x': r['x'], 'y': r['y'], 'lat': r['lat'], 'lng': r['lng'],
        'description': r['description'], 'status': r['status'],
        'capacity': r['capacity'],
        'created_at': r['created_at'], 'updated_at': r['updated_at']
    }

def read_points():
    with get_db_cursor() as cur:
        cur.execute('SELECT id, type, label, x, y, lat, lng, description, status, capacity, created_at, updated_at FROM points ORDER BY created_at')
        rows = cur.fetchall()
    return [row_to_point(r) for r in rows]

def compute_xy_from_latlng(lat, lng):
    ref_lat, ref_lng = 7.8731, 80.7718
    R = 6371.0
    x = R * math.radians(lng - ref_lng) * math.cos(math.radians(ref_lat))
    y = R * math.radians(lat - ref_lat)
    return x, y

def append_point(data: dict):
    now = datetime.now(timezone.utc).isoformat()
    lat = data.get('lat')
    lng = data.get('lng')
    if lat is not None:
        try: lat = float(lat)
        except: lat = None
    if lng is not None:
        try: lng = float(lng)
        except: lng = None
    if lat is not None and lng is not None:
        x, y = compute_xy_from_latlng(lat, lng)
    else:
        x = float(data.get('x', 0) or 0)
        y = float(data.get('y', 0) or 0)
    row = {
        'id': uuid.uuid4().hex,
        'type': data.get('type', ''),
        'label': data.get('label', ''),
        'x': x,
        'y': y,
        'lat': lat,
        'lng': lng,
        'description': data.get('description', ''),
        'status': data.get('status', ''),
        'capacity': data.get('capacity'),
        'created_at': now,
        'updated_at': now
    }
    with get_db_cursor() as cur:
        cur.execute('''
            INSERT INTO points (id, type, label, x, y, lat, lng, description, status, capacity, created_at, updated_at)
            VALUES (%(id)s, %(type)s, %(label)s, %(x)s, %(y)s, %(lat)s, %(lng)s, %(description)s, %(status)s, %(capacity)s, %(created_at)s, %(updated_at)s)
        ''', row)
    return row

def update_point(point_id, data):
    with get_db_cursor() as cur:
        cur.execute('SELECT id, type, label, x, y, lat, lng, description, status, capacity, created_at, updated_at FROM points WHERE id = %s', (point_id,))
        existing = cur.fetchone()
        if existing is None:
            return None

        target = row_to_point(existing)
        if 'type' in data and data['type']: target['type'] = data['type']
        if 'label' in data and data['label']: target['label'] = data['label']
        if 'description' in data: target['description'] = data['description']
        if 'status' in data: target['status'] = data['status']
        if 'capacity' in data: target['capacity'] = data['capacity']

        lat = data.get('lat')
        lng = data.get('lng')
        if lat is not None or lng is not None:
            if lat is not None:
                try: lat = float(lat)
                except: 
                    return {'error': 'Invalid latitude'}
            if lng is not None:
                try: lng = float(lng)
                except: 
                    return {'error': 'Invalid longitude'}
            target['lat'] = lat
            target['lng'] = lng
            if lat is not None and lng is not None:
                target['x'], target['y'] = compute_xy_from_latlng(lat, lng)
        else:
            if 'x' in data and data['x'] is not None:
                try: target['x'] = float(data['x'])
                except: pass
            if 'y' in data and data['y'] is not None:
                try: target['y'] = float(data['y'])
                except: pass

        target['updated_at'] = datetime.now(timezone.utc).isoformat()
        cur.execute('''
            UPDATE points SET type=%(type)s, label=%(label)s, x=%(x)s, y=%(y)s, lat=%(lat)s, lng=%(lng)s,
                description=%(description)s, status=%(status)s, capacity=%(capacity)s, updated_at=%(updated_at)s
            WHERE id=%(id)s
        ''', target)
    return target

def delete_point(point_id):
    with get_db_cursor() as cur:
        cur.execute('DELETE FROM points WHERE id = %s', (point_id,))
        rowcount = cur.rowcount
    return rowcount > 0

def build_svg(points):
    lats = [p['lat'] for p in points if p['lat'] is not None]
    lngs = [p['lng'] for p in points if p['lng'] is not None]
    if not lats or not lngs:
        return '''<?xml version="1.0" encoding="utf-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 480">
        <rect width="720" height="480" fill="#0b1830"/>
        <text x="360" y="240" text-anchor="middle" fill="#8ea0c4" font-size="20">No points with coordinates</text>
        </svg>'''

    min_lat, max_lat = min(lats), max(lats)
    min_lng, max_lng = min(lngs), max(lngs)
    lat_pad = (max_lat - min_lat) * 0.1 or 0.01
    lng_pad = (max_lng - min_lng) * 0.1 or 0.01
    min_lat -= lat_pad
    max_lat += lat_pad
    min_lng -= lng_pad
    max_lng += lng_pad

    def project(lat, lng):
        x = (lng - min_lng) / (max_lng - min_lng) * 720
        y = (max_lat - lat) / (max_lat - min_lat) * 480
        return x, y

    parts = []
    parts.append('<?xml version="1.0" encoding="utf-8"?>')
    parts.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 480">')
    parts.append('<rect width="720" height="480" fill="#0b1830"/>')

    for p in points:
        if p['lat'] is None or p['lng'] is None: continue
        cx, cy = project(p['lat'], p['lng'])
        ptype = (p.get('type') or '').lower()
        label = p.get('label', '?')
        if ptype.startswith('safe'):
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="24" fill="#36d1c4" opacity="0.9"/>')
            parts.append(f'<text x="{cx}" y="{cy+7}" text-anchor="middle" font-size="16" fill="#04211e" font-weight="800">{label}</text>')
        else:
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="18" fill="#ff5d6c" opacity="0.95"/>')
            parts.append(f'<text x="{cx}" y="{cy+5}" text-anchor="middle" font-size="14" fill="white" font-weight="700">{label}</text>')
    parts.append('</svg>')
    return '\\n'.join(parts)


# ─── ROUTES ─────────────────────────────────────────────────────

@router.get('/evacuation')
async def evacuation_dashboard():
    with open(HTML_FILE, 'r', encoding='utf-8') as f:
        html = f.read()
    return HTMLResponse(content=html)



@router.get('/api/points')
def api_points_get():
    return read_points()

@router.post('/api/points', status_code=201)
async def api_points_post(request: Request, admin: bool = Depends(require_admin)):
    try:
        data = await request.json()
    except:
        raise HTTPException(status_code=400, detail="invalid json")
    
    if not data or 'type' not in data or 'label' not in data:
        raise HTTPException(status_code=400, detail="missing type or label")
    
    if data.get('lat') is not None:
        try: float(data['lat'])
        except: raise HTTPException(status_code=400, detail="invalid lat")
    if data.get('lng') is not None:
        try: float(data['lng'])
        except: raise HTTPException(status_code=400, detail="invalid lng")
        
    row = append_point(data)
    return row

@router.put('/api/points/{point_id}')
async def api_update_point_route(point_id: str, request: Request, admin: bool = Depends(require_admin)):
    try: data = await request.json()
    except: raise HTTPException(status_code=400, detail="invalid json")
    if not data:
        raise HTTPException(status_code=400, detail="missing payload")
    
    updated = update_point(point_id, data)
    if isinstance(updated, dict) and 'error' in updated:
        raise HTTPException(status_code=400, detail=updated['error'])
    if updated is None:
        raise HTTPException(status_code=404, detail="not found")
    return updated

@router.delete('/api/points/{point_id}')
def api_delete_point_route(point_id: str, admin: bool = Depends(require_admin)):
    ok = delete_point(point_id)
    if not ok:
        raise HTTPException(status_code=404, detail="not found")
    return {'status': 'deleted', 'id': point_id}

@router.get('/api/routes')
def api_routes():
    points = read_points()
    evac = [p for p in points if p['type'] and 'evac' in p['type'].lower()]
    safe = [p for p in points if p['type'] and 'safe' in p['type'].lower()]
    routes = []

    def haversine(a, b):
        if a['lat'] is None or a['lng'] is None or b['lat'] is None or b['lng'] is None:
            return float('inf')
        R = 6371.0
        lat1, lon1 = math.radians(a['lat']), math.radians(a['lng'])
        lat2, lon2 = math.radians(b['lat']), math.radians(b['lng'])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a_ = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a_), math.sqrt(1-a_))

    for e in evac:
        if not safe: continue
        nearest = min(safe, key=lambda s: haversine(e, s))
        path_latlng = []
        if e['lat'] is not None and e['lng'] is not None and nearest['lat'] is not None and nearest['lng'] is not None:
            path_latlng = [[e['lat'], e['lng']], [nearest['lat'], nearest['lng']]]
        status = 'safe'
        if 'closed' in (e.get('status', '') or '').lower() or 'closed' in (nearest.get('status', '') or '').lower():
            status = 'blocked'
        routes.append({
            'id': uuid.uuid4().hex, 'from_id': e['id'], 'to_id': nearest['id'],
            'path_svg': [], 'path_latlng': path_latlng, 'status': status
        })
    return routes

@router.get('/api/rank_safe_zones')
def api_rank_safe_zones(origin_lat: Optional[str] = None, origin_lng: Optional[str] = None, profile: str = 'default'):
    points = read_points()
    safe = [p for p in points if p['type'] and 'safe' in p['type'].lower()]
    if not safe: return []

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    scored = []
    for s in safe:
        score = 0.0
        dist = None
        if origin_lat and origin_lng and s['lat'] is not None and s['lng'] is not None:
            try: dist = haversine(float(origin_lat), float(origin_lng), float(s['lat']), float(s['lng']))
            except: dist = None
            
        dist_score = 0.0 if dist is None else max(0.0, 1.0 - min(dist / 50.0, 1.0))
        score += dist_score * 0.6
        st = (s.get('status') or '').lower()
        if 'open' in st or 'ready' in st or 'accessible' in st or 'active' in st: score += 0.3
        if 'overcrowded' in st: score -= 0.25
        if 'closed' in st or 'unavailable' in st: score -= 0.9
        
        cap = s.get('capacity')
        if cap is not None:
            try:
                cap = int(cap)
                if cap <= 0: score -= 0.2
                elif cap > 200: score += 0.05
            except: pass
            
        if profile == 'elderly': score += dist_score * 0.2
        scored.append({'point': s, 'score': round(score, 4), 'distance_km': dist})
        
    scored.sort(key=lambda x: x['score'], reverse=True)
    open_zones = [entry for entry in scored if 'closed' not in (entry['point']['status'] or '').lower()]
    if open_zones:
        scored = open_zones + [entry for entry in scored if 'closed' in (entry['point']['status'] or '').lower()]
    return scored

@router.get('/api/hazards')
def api_hazards_get():
    with get_db_cursor() as cur:
        cur.execute('SELECT id, lat, lng, radius_m, note, created_at FROM hazards ORDER BY created_at DESC')
        rows = cur.fetchall()
    return [dict(r) for r in rows]

@router.post('/api/hazards', status_code=201)
async def api_hazards_post(request: Request, admin: bool = Depends(require_admin)):
    try: data = await request.json()
    except: raise HTTPException(status_code=400, detail="invalid json")
    
    row = {
        'id': uuid.uuid4().hex,
        'lat': data.get('lat'), 'lng': data.get('lng'),
        'radius_m': data.get('radius_m', 100),
        'note': data.get('note', ''),
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    with get_db_cursor() as cur:
        cur.execute('INSERT INTO hazards (id, lat, lng, radius_m, note, created_at) VALUES (%(id)s,%(lat)s,%(lng)s,%(radius_m)s,%(note)s,%(created_at)s)', row)
    return row

@router.get('/api/export/svg')
def export_svg_route():
    points = read_points()
    svg = build_svg(points)
    return Response(content=svg, media_type='image/svg+xml', headers={'Content-Disposition': 'attachment; filename="evacuation_map.svg"'})

@router.get('/api/export/package')
def export_package_route():
    points = read_points()
    svg = build_svg(points)
    routes = []
    evac = [p for p in points if p['type'] and 'evac' in p['type'].lower()]
    safe_points = [p for p in points if p['type'] and 'safe' in p['type'].lower()]

    def haversine(a, b):
        if a['lat'] is None or a['lng'] is None or b['lat'] is None or b['lng'] is None:
            return float('inf')
        R = 6371.0
        lat1, lon1 = math.radians(a['lat']), math.radians(a['lng'])
        lat2, lon2 = math.radians(b['lat']), math.radians(b['lng'])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a_ = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a_), math.sqrt(1-a_))

    for e in evac:
        if not safe_points: continue
        nearest = min(safe_points, key=lambda s: haversine(e, s))
        routes.append({
            'from': e['id'], 'to': nearest['id'], 'path_svg': [],
            'path_latlng': [[e['lat'], e['lng']], [nearest['lat'], nearest['lng']]] if e['lat'] and nearest['lat'] else []
        })

    contacts = [
        {'name': 'National Emergency Hotline', 'phone': '117'},
        {'name': 'Local Municipality Office', 'phone': '+94 11 2 345678'}
    ]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('evacuation_map.svg', svg)
        zf.writestr('points.json', json.dumps(points, ensure_ascii=False, indent=2))
        zf.writestr('routes.json', json.dumps(routes, ensure_ascii=False, indent=2))
        zf.writestr('contacts.json', json.dumps(contacts, ensure_ascii=False, indent=2))
    buf.seek(0)
    return Response(content=buf.read(), media_type='application/zip', headers={'Content-Disposition': 'attachment; filename="evacuation_offline_package.zip"'})