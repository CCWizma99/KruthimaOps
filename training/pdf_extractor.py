import os
import requests
import pdfplumber
import json
import re

HEADERS = {
    "User-Agent": "FloodRiskResearchBot/1.0 (Collecting ~600 historical PDFs for ML flood prediction research; friendly scraper)",
    "X-Purpose": "Academic Research"
}

CACHE_DIR = "c:/KruthimaOps/data/pdf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def download_pdf(url, date_str):
    """Download PDF to cache if it doesn't exist."""
    filename = f"dmc_{date_str}.pdf"
    filepath = os.path.join(CACHE_DIR, filename)
    
    if os.path.exists(filepath):
        return filepath
        
    print(f"Downloading PDF for {date_str} from {url}...")
    try:
        r = requests.get(url, headers=HEADERS, verify=False, timeout=15)
        if r.ok:
            with open(filepath, 'wb') as f:
                f.write(r.content)
            return filepath
        else:
            print(f"Failed to download {url}: {r.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return None

def extract_gauge_data(pdf_path):
    """Extract table from DMC PDF and return a dictionary of gauge data."""
    if not os.path.exists(pdf_path):
        return None
        
    gauge_data = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Usually the data is on the first page
            page = pdf.pages[0]
            # Extract table
            table = page.extract_table()
            
            if not table:
                return gauge_data
                
            # Scan through rows to find the data.
            # Table structure varies, but usually it's:
            # Basin | River | Station | Unit | Alert | Minor | Major | 11am | 12pm | Remarks ...
            for row in table:
                if not row or len(row) < 8: continue
                
                # Check if it's a data row by seeing if unit is 'm' or 'ft' or there are numbers
                # Column index can shift if cells are merged. Let's find the station name.
                # Station is typically the 3rd or 4th column depending on merged cells.
                
                # Clean up None values
                clean_row = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
                
                # Find the "Unit" column to anchor our indexing (usually contains "m" or "ft")
                unit_idx = -1
                for i, cell in enumerate(clean_row):
                    if cell in ['m', 'ft']:
                        unit_idx = i
                        break
                        
                if unit_idx > 0 and unit_idx + 4 < len(clean_row):
                    station_name = clean_row[unit_idx - 1]
                    # Sometimes station name gets merged with other things, but usually it's clean
                    if "Station" in station_name or station_name == "": continue
                    
                    try:
                        minor_flood = float(re.sub(r'[^\d.]', '', clean_row[unit_idx + 2]))
                        major_flood = float(re.sub(r'[^\d.]', '', clean_row[unit_idx + 3]))
                        
                        # 11:00 am or 12:00 noon reading (take noon if available, else 11am)
                        level_11am_str = clean_row[unit_idx + 4]
                        level_12pm_str = clean_row[unit_idx + 5] if unit_idx + 5 < len(clean_row) else ""
                        
                        water_level = None
                        if level_12pm_str and re.search(r'\d', level_12pm_str):
                            water_level = float(re.sub(r'[^\d.-]', '', level_12pm_str))
                        elif level_11am_str and re.search(r'\d', level_11am_str):
                            water_level = float(re.sub(r'[^\d.-]', '', level_11am_str))
                            
                        if water_level is not None:
                            gauge_data[station_name] = {
                                "unit": clean_row[unit_idx],
                                "minor_flood": minor_flood,
                                "major_flood": major_flood,
                                "water_level": water_level
                            }
                    except ValueError:
                        pass # Ignore rows that fail to parse
    except Exception as e:
        print(f"Error parsing {pdf_path}: {e}")
        
    return gauge_data

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings()
    
    # Test on a known PDF
    test_url = "https://www.dmc.gov.lk/images/dmcreports/Water_level_&_Rainfall_2026__1781420151.pdf"
    path = download_pdf(test_url, "test_2026_06_14")
    if path:
        print(f"Downloaded to {path}")
        data = extract_gauge_data(path)
        print(json.dumps(data, indent=2))
        print(f"Extracted {len(data)} stations.")
