import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371.0 # Earth radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def find_closest_gauge(row_lat, row_lon, gauge_data, station_coords):
    """
    Given a lat/lon and the day's gauge_data, find the closest active gauge
    using the static station_coords map.
    Returns: (station_name, distance_km, gauge_dict)
    """
    if not gauge_data:
        return None, None, None
        
    closest_station = None
    min_distance = float('inf')
    
    for station_name, g_info in gauge_data.items():
        # Match station name to coordinates
        # Exact match or substring match (e.g. "Kalawellawa" in "Kalawellawa (Millakanda)")
        matched_coords = None
        for st_name, coords in station_coords.items():
            if st_name.lower() in station_name.lower() or station_name.lower() in st_name.lower():
                matched_coords = coords
                break
                
        if matched_coords:
            dist = haversine_distance(row_lat, row_lon, matched_coords['lat'], matched_coords['lon'])
            if dist < min_distance:
                min_distance = dist
                closest_station = station_name
                
    if closest_station:
        return closest_station, min_distance, gauge_data[closest_station]
        
    return None, None, None
