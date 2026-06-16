from flask import Flask, request, jsonify, Response
import os
import csv
import uuid
from datetime import datetime
import io
import zipfile
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'evac_points.csv')
HTML_FILE = os.path.join(HERE, 'evacuation_presentation.html')

app = Flask(__name__, static_folder=HERE)

os.makedirs(DATA_DIR, exist_ok=True)

CSV_FIELDS = ['id', 'type', 'label', 'x', 'y', 'lat', 'lng', 'description', 'status', 'created_at']

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def read_points():
    points = []
    try:
        with open(CSV_PATH, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                if not r.get('id'):
                    continue
                try:
                    x = float(r.get('x', 0)) if r.get('x') not in (None, '') else 0.0
                except Exception:
                    x = 0.0
                try:
                    y = float(r.get('y', 0)) if r.get('y') not in (None, '') else 0.0
                except Exception:
                    y = 0.0
                try:
                    lat = float(r.get('lat')) if r.get('lat') not in (None, '') else None
                except Exception:
                    lat = None
                try:
                    lng = float(r.get('lng')) if r.get('lng') not in (None, '') else None
                except Exception:
                    lng = None
                points.append({
                    'id': r.get('id'),
                    'type': r.get('type'),
                    'label': r.get('label'),
                    'x': x,
                    'y': y,
                    'lat': lat,
                    'lng': lng,
                    'description': r.get('description', ''),
                    'status': r.get('status', ''),
                    'created_at': r.get('created_at', '')
                })
    except FileNotFoundError:
        return []
    return points


def write_all_points(points):
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for p in points:
            writer.writerow({
                'id': p['id'], 'type': p['type'], 'label': p['label'],
                'x': p['x'], 'y': p['y'],
                'lat': p['lat'] if p['lat'] is not None else '',
                'lng': p['lng'] if p['lng'] is not None else '',
                'description': p['description'], 'status': p['status'],
                'created_at': p['created_at']
            })


def append_point(point):
    row = {
        'id': point.get('id') or uuid.uuid4().hex,
        'type': point.get('type', ''),
        'label': point.get('label', ''),
        'x': point.get('x', 0),
        'y': point.get('y', 0),
        'lat': point.get('lat', ''),
        'lng': point.get('lng', ''),
        'description': point.get('description', ''),
        'status': point.get('status', ''),
        'created_at': datetime.utcnow().isoformat()
    }
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)
    return row


def update_point(point_id, data):
    points = read_points()
    target = None
    for p in points:
        if p['id'] == point_id:
            target = p
            break
    if target is None:
        return None

    if 'type' in data and data['type'] not in (None, ''):
        target['type'] = data['type']
    if 'label' in data and data['label'] not in (None, ''):
        target['label'] = data['label']
    if 'description' in data:
        target['description'] = data['description']
    if 'status' in data:
        target['status'] = data['status']
    if 'x' in data and data['x'] not in (None, ''):
        try:
            target['x'] = float(data['x'])
        except Exception:
            pass
    if 'y' in data and data['y'] not in (None, ''):
        try:
            target['y'] = float(data['y'])
        except Exception:
            pass
    if 'lat' in data:
        try:
            target['lat'] = float(data['lat']) if data['lat'] not in (None, '') else None
        except Exception:
            target['lat'] = None
    if 'lng' in data:
        try:
            target['lng'] = float(data['lng']) if data['lng'] not in (None, '') else None
        except Exception:
            target['lng'] = None

    write_all_points(points)
    return target


def delete_point(point_id):
    points = read_points()
    remaining = [p for p in points if p['id'] != point_id]
    if len(remaining) == len(points):
        return False
    write_all_points(remaining)
    return True


def build_svg(points):
    parts = []
    parts.append('<?xml version="1.0" encoding="utf-8"?>')
    parts.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 480">')
    parts.append('<rect width="720" height="480" fill="#edf3fb"/>')
    parts.append('<rect x="120" y="100" width="480" height="220" rx="22" fill="#ffffff" stroke="#c8d7e8" stroke-width="2"/>')
    for p in points:
        ptype = (p.get('type') or '').lower()
        if ptype.startswith('safe'):
            parts.append(f'<circle cx="{p["x"]}" cy="{p["y"]}" r="24" fill="#00b386" opacity="0.9"/>')
            parts.append(f'<text x="{p["x"]}" y="{p["y"]+7}" text-anchor="middle" font-size="16" fill="white" font-weight="800">{p["label"]}</text>')
        else:
            parts.append(f'<circle cx="{p["x"]}" cy="{p["y"]}" r="18" fill="#ff6b6b" opacity="0.95"/>')
            parts.append(f'<text x="{p["x"]}" y="{p["y"]+5}" text-anchor="middle" font-size="14" fill="white" font-weight="700">{p["label"]}</text>')
    parts.append('</svg>')
    return '\n'.join(parts)


@app.route('/')
def index():
    with open(HTML_FILE, 'r', encoding='utf-8') as f:
        html = f.read()
    return Response(html, mimetype='text/html')


@app.route('/api/points', methods=['GET', 'POST'])
def api_points():
    if request.method == 'GET':
        return jsonify(read_points())
    data = request.get_json(force=True)
    if not data or 'type' not in data or 'label' not in data:
        return jsonify({'error': 'missing type or label'}), 400
    try:
        data['x'] = float(data.get('x', 0) or 0)
        data['y'] = float(data.get('y', 0) or 0)
    except Exception:
        return jsonify({'error': 'invalid coordinates'}), 400
    if data.get('lat') in (None, ''):
        data['lat'] = None
    if data.get('lng') in (None, ''):
        data['lng'] = None
    row = append_point(data)
    return jsonify(row), 201


@app.route('/api/points/<point_id>', methods=['PUT'])
def api_update_point(point_id):
    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'missing payload'}), 400
    updated = update_point(point_id, data)
    if updated is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(updated)


@app.route('/api/points/<point_id>', methods=['DELETE'])
def api_delete_point(point_id):
    ok = delete_point(point_id)
    if not ok:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'status': 'deleted', 'id': point_id})


@app.route('/api/routes', methods=['GET'])
def api_routes():
    points = read_points()
    evac = [p for p in points if p['type'] and 'evac' in p['type'].lower()]
    safe = [p for p in points if p['type'] and 'safe' in p['type'].lower()]
    routes = []

    def euclid(a, b):
        return math.hypot(a['x'] - b['x'], a['y'] - b['y'])

    for e in evac:
        if not safe:
            continue
        nearest = min(safe, key=lambda s: euclid(e, s))
        path_svg = [[e['x'], e['y']], [nearest['x'], nearest['y']]]
        path_latlng = []
        if e['lat'] is not None and e['lng'] is not None and nearest['lat'] is not None and nearest['lng'] is not None:
            path_latlng = [[e['lat'], e['lng']], [nearest['lat'], nearest['lng']]]
        status = 'safe'
        if 'closed' in (e.get('status', '') or '').lower() or 'closed' in (nearest.get('status', '') or '').lower():
            status = 'blocked'
        routes.append({
            'id': uuid.uuid4().hex,
            'from_id': e['id'],
            'to_id': nearest['id'],
            'path_svg': path_svg,
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
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
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
        if 'closed' in st or 'unavailable' in st:
            score -= 0.5
        if profile == 'elderly':
            score += dist_score * 0.2
        scored.append({'point': s, 'score': round(score, 4), 'distance_km': dist})
    scored.sort(key=lambda x: x['score'], reverse=True)
    return jsonify(scored)


@app.route('/api/export/svg')
def export_svg():
    points = read_points()
    svg = build_svg(points)
    return Response(svg, mimetype='image/svg+xml', headers={
        'Content-Disposition': 'attachment; filename="evacuation_map.svg"'
    })


@app.route('/api/export/package')
def export_package():
    points = read_points()
    svg = build_svg(points)

    routes = []
    evac = [p for p in points if p['type'] and 'evac' in p['type'].lower()]
    safe_points = [p for p in points if p['type'] and 'safe' in p['type'].lower()]

    def euclid(a, b):
        return math.hypot(a['x'] - b['x'], a['y'] - b['y'])

    for e in evac:
        if not safe_points:
            continue
        nearest = min(safe_points, key=lambda s: euclid(e, s))
        routes.append({
            'from': e['id'],
            'to': nearest['id'],
            'path_svg': [[e['x'], e['y']], [nearest['x'], nearest['y']]],
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
    return Response(buf.read(), mimetype='application/zip', headers={
        'Content-Disposition': 'attachment; filename="evacuation_offline_package.zip"'
    })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)