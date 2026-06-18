import json
import requests
from bs4 import BeautifulSoup
import os

DMC_INDEX_URL = "https://www.dmc.gov.lk/index.php?option=com_dmcreports&view=reports&Itemid=277&limit=0&search=&report_type_id=6&fromdate=&todate=&lang=en"
HEADERS = {
    "User-Agent": "FloodRiskResearchBot/1.0 (Collecting ~600 historical PDFs for ML flood prediction research; friendly scraper)",
    "X-Purpose": "Academic Research"
}

def build_index(output_path="c:/KruthimaOps/data/dmc_pdf_index.json"):
    print(f"Fetching DMC index from {DMC_INDEX_URL}...")
    r = requests.get(DMC_INDEX_URL, headers=HEADERS, verify=False)
    
    if not r.ok:
        print(f"Failed to fetch. Status code: {r.status_code}")
        return
        
    soup = BeautifulSoup(r.text, 'html.parser')
    rows = soup.find_all('tr')
    
    pdf_map = {}
    
    for row in rows:
        cols = row.find_all('td')
        if len(cols) > 3:
            date_str = cols[1].text.strip()
            link = cols[3].find('a')
            if link and link.get('href'):
                href = link.get('href')
                if href.lower().endswith('.pdf'):
                    # The DMC website formats dates as YYYY-MM-DD
                    if date_str not in pdf_map:
                        pdf_map[date_str] = "https://www.dmc.gov.lk" + href

    print(f"Indexed {len(pdf_map)} unique dates with PDFs.")
    
    with open(output_path, 'w') as f:
        json.dump(pdf_map, f, indent=4)
    print(f"Saved index to {output_path}")

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings()
    build_index()
