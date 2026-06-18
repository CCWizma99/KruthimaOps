import json
import time
import requests

stations = [
    'Nagalagam Street', 'Hanwella', 'Glencourse', 'Kithulgala', 'Holombuwa', 'Deraniyagala', 'Norwood',
    'Putupaula', 'Ellagawa', 'Rathnapura', 'Magura', 'Kalawellawa',
    'Baddegama', 'Thawalama',
    'Thalgahagoda', 'Panadugama', 'Pitabeddara', 'Urawa',
    'Moraketiya',
    'Thanamalwila', 'Wellawaya', 'Kuda Oya',
    'Katharagama',
    'Nakkala',
    'Siyambalanduwa',
    'Padiyathalawa',
    'Manampitiya', 'Weraganthota', 'Peradeniya', 'Nawalapitiya', 'Thaldena',
    'Horowpothana',
    'Yaka Wewa',
    'Thanthirimale',
    'Galgamuwa',
    'Moragaswewa',
    'Badalgama', 'Giriulla',
    'Dunamale'
]

# Approximate fallbacks if OSM fails
FALLBACK_COORDS = {
    'Nagalagam Street': {'lat': 6.9538, 'lon': 79.8770},
    'Hanwella': {'lat': 6.8978, 'lon': 80.0811},
    'Baddegama': {'lat': 6.1869, 'lon': 80.1906},
    'Rathnapura': {'lat': 6.7056, 'lon': 80.3847},
    'Katharagama': {'lat': 6.4144, 'lon': 81.3340},
    'Manampitiya': {'lat': 7.9142, 'lon': 81.0967},
    'Peradeniya': {'lat': 7.2660, 'lon': 80.5954},
    'Nawalapitiya': {'lat': 7.0543, 'lon': 80.5350},
    'Galgamuwa': {'lat': 8.0336, 'lon': 80.2741},
    'Wellawaya': {'lat': 6.7350, 'lon': 81.1042},
    'Thanamalwila': {'lat': 6.4385, 'lon': 81.1350},
    'Padiyathalawa': {'lat': 7.3916, 'lon': 81.1610},
    'Siyambalanduwa': {'lat': 6.9097, 'lon': 81.5658},
    'Giriulla': {'lat': 7.3298, 'lon': 80.1171},
    'Thanthirimale': {'lat': 8.5833, 'lon': 80.2667},
    'Kithulgala': {'lat': 6.9936, 'lon': 80.4124},
    'Deraniyagala': {'lat': 6.9248, 'lon': 80.3391},
    'Norwood': {'lat': 6.8406, 'lon': 80.6063},
    'Ellagawa': {'lat': 6.7570, 'lon': 80.1583},
    'Thawalama': {'lat': 6.3402, 'lon': 80.3348},
    'Panadugama': {'lat': 6.1081, 'lon': 80.5050},
    'Pitabeddara': {'lat': 6.2201, 'lon': 80.4800},
    'Moraketiya': {'lat': 6.3113, 'lon': 80.8932},
    'Nakkala': {'lat': 6.8524, 'lon': 81.3218},
    'Weraganthota': {'lat': 7.3195, 'lon': 80.9922},
    'Horowpothana': {'lat': 8.5146, 'lon': 80.8715},
    'Badalgama': {'lat': 7.2885, 'lon': 79.9723},
}

coords = {}
headers = {'User-Agent': 'FloodRiskResearchBot/1.0'}

for st in stations:
    try:
        r = requests.get(f'https://nominatim.openstreetmap.org/search?q={st}, Sri Lanka&format=json', headers=headers, timeout=5)
        if r.ok:
            data = r.json()
            if data:
                coords[st] = {'lat': float(data[0]['lat']), 'lon': float(data[0]['lon'])}
                print(f'Found: {st} -> {data[0]["lat"]}, {data[0]["lon"]}')
            else:
                print(f'Not found: {st}, using fallback if available')
                if st in FALLBACK_COORDS:
                    coords[st] = FALLBACK_COORDS[st]
        else:
            print(f'HTTP Error for {st}: {r.status_code}')
            if st in FALLBACK_COORDS:
                coords[st] = FALLBACK_COORDS[st]
    except Exception as e:
        print(f'Timeout/Error for {st}, using fallback. {e}')
        if st in FALLBACK_COORDS:
            coords[st] = FALLBACK_COORDS[st]
    time.sleep(1.2) # Friendly delay

# Some default values for completely missing ones to avoid breaking Haversine
for st in stations:
    if st not in coords:
        coords[st] = {'lat': 7.0, 'lon': 80.0} # Center of SL fallback

with open('c:/KruthimaOps/data/station_coords.json', 'w') as f:
    json.dump(coords, f, indent=4)
print("Finished saving station_coords.json")
