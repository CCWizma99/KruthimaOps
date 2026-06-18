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

coords = {}
headers = {'User-Agent': 'FloodRiskResearchBot/1.0'}

for st in stations:
    try:
        r = requests.get(f'https://nominatim.openstreetmap.org/search?q={st}, Sri Lanka&format=json', headers=headers)
        if r.ok:
            data = r.json()
            if data:
                coords[st] = {'lat': float(data[0]['lat']), 'lon': float(data[0]['lon'])}
                print(f'Found: {st} -> {data[0]["lat"]}, {data[0]["lon"]}')
            else:
                print(f'Not found: {st}')
        else:
            print(f'HTTP Error for {st}: {r.status_code}')
        time.sleep(1)
    except Exception as e:
        print(f'Error for {st}: {e}')

with open('c:/KruthimaOps/data/station_coords.json', 'w') as f:
    json.dump(coords, f, indent=4)
print("Finished saving station_coords.json")
