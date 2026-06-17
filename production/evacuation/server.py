from flask import Flask, request, jsonify, Response
import os
import sqlite3
import uuid
from datetime import datetime, timezone
import io
import zipfile
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
DB_PATH = os.path.join(DATA_DIR, 'evac_points.db')
HTML_FILE = os.path.join(HERE, 'evacuation_presentation.html')
SW_FILE = os.path.join(HERE, 'sw.js')

ADMIN_KEY = os.environ.get('FLOODGUARD_ADMIN_KEY', 'changeme-floodguard')

app = Flask(__name__, static_folder=HERE)
os.makedirs(DATA_DIR, exist_ok=True)

# ─── DATABASE ────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS points (
            id TEXT PRIMARY KEY,
            type TEXT,
            label TEXT,
            x REAL,
            y REAL,
            lat REAL,
            lng REAL,
            description TEXT,
            status TEXT,
            capacity INTEGER,
            created_at TEXT,
            updated_at TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS hazards (
            id TEXT PRIMARY KEY,
            lat REAL,
            lng REAL,
            radius_m REAL,
            note TEXT,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ─── AUTH ───────────────────────────────────────────────────────
def require_admin(req):
    key = req.headers.get('X-Admin-Key', '')
    return key == ADMIN_KEY

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
    conn = get_db()
    rows = conn.execute('SELECT * FROM points ORDER BY created_at').fetchall()
    conn.close()
    return [row_to_point(r) for r in rows]

def compute_xy_from_latlng(lat, lng):
    """Simple equirectangular projection relative to a reference point."""
    # Use the center of Sri Lanka as reference for moderate accuracy
    ref_lat, ref_lng = 7.8731, 80.7718
    R = 6371.0
    x = R * math.radians(lng - ref_lng) * math.cos(math.radians(ref_lat))
    y = R * math.radians(lat - ref_lat)
    return x, y

def append_point(data):
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    # ensure lat/lng are floats
    lat = data.get('lat')
    lng = data.get('lng')
    if lat is not None:
        try: lat = float(lat)
        except: lat = None
    if lng is not None:
        try: lng = float(lng)
        except: lng = None
    # compute x/y if lat/lng present, else use provided or 0
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
    conn.execute('''
        INSERT INTO points (id, type, label, x, y, lat, lng, description, status, capacity, created_at, updated_at)
        VALUES (:id, :type, :label, :x, :y, :lat, :lng, :description, :status, :capacity, :created_at, :updated_at)
    ''', row)
    conn.commit()
    conn.close()
    return row

def update_point(point_id, data):
    conn = get_db()
    existing = conn.execute('SELECT * FROM points WHERE id = ?', (point_id,)).fetchone()
    if existing is None:
        conn.close()
        return None

    target = row_to_point(existing)
    # Update simple fields
    if 'type' in data and data['type']: target['type'] = data['type']
    if 'label' in data and data['label']: target['label'] = data['label']
    if 'description' in data: target['description'] = data['description']
    if 'status' in data: target['status'] = data['status']
    if 'capacity' in data: target['capacity'] = data['capacity']

    # Update lat/lng with validation
    lat = data.get('lat')
    lng = data.get('lng')
    if lat is not None or lng is not None:
        if lat is not None:
            try:
                lat = float(lat)
            except:
                conn.close()
                return {'error': 'Invalid latitude'}
        if lng is not None:
            try:
                lng = float(lng)
            except:
                conn.close()
                return {'error': 'Invalid longitude'}
        target['lat'] = lat
        target['lng'] = lng
        if lat is not None and lng is not None:
            target['x'], target['y'] = compute_xy_from_latlng(lat, lng)
        else:
            # if only one is provided, keep old x/y? Better to keep as is.
            pass
    else:
        # if x/y are provided directly (legacy)
        if 'x' in data and data['x'] is not None:
            try: target['x'] = float(data['x'])
            except: pass
        if 'y' in data and data['y'] is not None:
            try: target['y'] = float(data['y'])
            except: pass

    target['updated_at'] = datetime.now(timezone.utc).isoformat()

    conn.execute('''
        UPDATE points SET type=:type, label=:label, x=:x, y=:y, lat=:lat, lng=:lng,
            description=:description, status=:status, capacity=:capacity, updated_at=:updated_at
        WHERE id=:id
    ''', target)
    conn.commit()
    conn.close()
    return target

def delete_point(point_id):
    conn = get_db()
    cur = conn.execute('DELETE FROM points WHERE id = ?', (point_id,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0

def build_svg(points):
    # Compute bounding box from points with lat/lng
    lats = [p['lat'] for p in points if p['lat'] is not None]
    lngs = [p['lng'] for p in points if p['lng'] is not None]
    if not lats or not lngs:
        # fallback to empty map
        return '''<?xml version="1.0" encoding="utf-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 480">
        <rect width="720" height="480" fill="#0b1830"/>
        <text x="360" y="240" text-anchor="middle" fill="#8ea0c4" font-size="20">No points with coordinates</text>
        </svg>'''

    min_lat, max_lat = min(lats), max(lats)
    min_lng, max_lng = min(lngs), max(lngs)
    # Add padding
    lat_pad = (max_lat - min_lat) * 0.1 or 0.01
    lng_pad = (max_lng - min_lng) * 0.1 or 0.01
    min_lat -= lat_pad
    max_lat += lat_pad
    min_lng -= lng_pad
    max_lng += lng_pad

    # Map to SVG coordinates (720x480)
    def project(lat, lng):
        x = (lng - min_lng) / (max_lng - min_lng) * 720
        y = (max_lat - lat) / (max_lat - min_lat) * 480  # flip y
        return x, y

    parts = []
    parts.append('<?xml version="1.0" encoding="utf-8"?>')
    parts.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 480">')
    parts.append('<rect width="720" height="480" fill="#0b1830"/>')

    for p in points:
        if p['lat'] is None or p['lng'] is None:
            continue
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
    return '\n'.join(parts)

# ─── ROUTES ─────────────────────────────────────────────────────
@app.route('/')
def index():
    with open(HTML_FILE, 'r', encoding='utf-8') as f:
        html = f.read()
    return Response(html, mimetype='text/html')

@app.route('/sw.js')
def service_worker():
    with open(SW_FILE, 'r', encoding='utf-8') as f:
        js = f.read()
    return Response(js, mimetype='application/javascript')

@app.route('/api/points', methods=['GET', 'POST'])
def api_points():
    if request.method == 'GET':
        return jsonify(read_points())
    if not require_admin(request):
        return jsonify({'error': 'admin key required'}), 401
    data = request.get_json(force=True)
    if not data or 'type' not in data or 'label' not in data:
        return jsonify({'error': 'missing type or label'}), 400
    # validate lat/lng if provided
    if data.get('lat') is not None:
        try: float(data['lat'])
        except: return jsonify({'error': 'invalid lat'}), 400
    if data.get('lng') is not None:
        try: float(data['lng'])
        except: return jsonify({'error': 'invalid lng'}), 400
    row = append_point(data)
    print(f"[ADMIN] Added point {row['id']} - {row['label']}")
    return jsonify(row), 201

@app.route('/api/points/<point_id>', methods=['PUT'])
def api_update_point(point_id):
    if not require_admin(request):
        return jsonify({'error': 'admin key required'}), 401
    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'missing payload'}), 400
    updated = update_point(point_id, data)
    if isinstance(updated, dict) and 'error' in updated:
        return jsonify(updated), 400
    if updated is None:
        return jsonify({'error': 'not found'}), 404
    print(f"[ADMIN] Updated point {point_id}")
    return jsonify(updated)

@app.route('/api/points/<point_id>', methods=['DELETE'])
def api_delete_point(point_id):
    if not require_admin(request):
        return jsonify({'error': 'admin key required'}), 401
    ok = delete_point(point_id)
    if not ok:
        return jsonify({'error': 'not found'}), 404
    print(f"[ADMIN] Deleted point {point_id}")
    return jsonify({'status': 'deleted', 'id': point_id})

@app.route('/api/routes', methods=['GET'])
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
        if not safe:
            continue
        nearest = min(safe, key=lambda s: haversine(e, s))
        path_latlng = []
        if e['lat'] is not None and e['lng'] is not None and nearest['lat'] is not None and nearest['lng'] is not None:
            path_latlng = [[e['lat'], e['lng']], [nearest['lat'], nearest['lng']]]
        status = 'safe'
        if 'closed' in (e.get('status', '') or '').lower() or 'closed' in (nearest.get('status', '') or '').lower():
            status = 'blocked'
        routes.append({
            'id': uuid.uuid4().hex, 'from_id': e['id'], 'to_id': nearest['id'],
            'path_svg': [],  # not used by frontend
            'path_latlng': path_latlng,
            'status': status
        })
    return jsonify(routes)

@app.route('/api/rank_safe_zones', methods=['GET'])
def api_rank_safe_zones():
    points = read_points()
    safe = [p for p in points if p['type'] and 'safe' in p['type'].lower()]
    if not safe:
        return jsonify([])
    origin_lat = request.args.get('origin_lat')
    origin_lng = request.args.get('origin_lng')
    profile = request.args.get('profile', 'default')

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
            try:
                dist = haversine(float(origin_lat), float(origin_lng), float(s['lat']), float(s['lng']))
            except Exception:
                dist = None
        dist_score = 0.0 if dist is None else max(0.0, 1.0 - min(dist / 50.0, 1.0))
        score += dist_score * 0.6
        st = (s.get('status') or '').lower()
        if 'open' in st or 'ready' in st or 'accessible' in st or 'active' in st:
            score += 0.3
        if 'overcrowded' in st:
            score -= 0.25
        if 'closed' in st or 'unavailable' in st:
            score -= 0.9
        cap = s.get('capacity')
        if cap is not None:
            try:
                cap = int(cap)
                if cap <= 0:
                    score -= 0.2
                elif cap > 200:
                    score += 0.05
            except Exception:
                pass
        if profile == 'elderly':
            score += dist_score * 0.2
        scored.append({'point': s, 'score': round(score, 4), 'distance_km': dist})
    # Sort by score descending
    scored.sort(key=lambda x: x['score'], reverse=True)
    # Hard-filter closed: move them to the end unless no open zones exist
    open_zones = [entry for entry in scored if 'closed' not in (entry['point']['status'] or '').lower()]
    if open_zones:
        scored = open_zones + [entry for entry in scored if 'closed' in (entry['point']['status'] or '').lower()]
    return jsonify(scored)

@app.route('/api/hazards', methods=['GET', 'POST'])
def api_hazards():
    conn = get_db()
    if request.method == 'GET':
        rows = conn.execute('SELECT * FROM hazards ORDER BY created_at DESC').fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    if not require_admin(request):
        conn.close()
        return jsonify({'error': 'admin key required'}), 401
    data = request.get_json(force=True)
    row = {
        'id': uuid.uuid4().hex,
        'lat': data.get('lat'), 'lng': data.get('lng'),
        'radius_m': data.get('radius_m', 100),
        'note': data.get('note', ''),
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    conn.execute('INSERT INTO hazards (id, lat, lng, radius_m, note, created_at) VALUES (:id,:lat,:lng,:radius_m,:note,:created_at)', row)
    conn.commit()
    conn.close()
    print(f"[ADMIN] Added hazard {row['id']}")
    return jsonify(row), 201

@app.route('/api/export/svg')
def export_svg():
    points = read_points()
    svg = build_svg(points)
    return Response(svg, mimetype='image/svg+xml', headers={'Content-Disposition': 'attachment; filename="evacuation_map.svg"'})

@app.route('/api/export/package')
def export_package():
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
        if not safe_points:
            continue
        nearest = min(safe_points, key=lambda s: haversine(e, s))
        routes.append({
            'from': e['id'], 'to': nearest['id'],
            'path_svg': [],  # not used
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
    return Response(buf.read(), mimetype='application/zip', headers={'Content-Disposition': 'attachment; filename="evacuation_offline_package.zip"'})

if __name__ == '__main__':
    # Use environment variable to control debug mode; default to False for safety
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)