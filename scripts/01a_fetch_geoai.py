import os
import json
import time
import requests
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_OUTPUT_FILE = OUTPUT_DIR / "raw_geoai_london.json"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

AMENITIES = [
    {"category": "hospital", "query": "hospital in London, UK"},
    {"category": "park", "query": "park in London, UK"},
    {"category": "bank", "query": "bank in London, UK"},
    {"category": "pharmacy", "query": "pharmacy in London, UK"},
    {"category": "restaurant", "query": "restaurant in London, UK"},
    {"category": "fire_station", "query": "fire station in London, UK"},
    {"category": "museum", "query": "museum in London, UK"},
    {"category": "university", "query": "university in London, UK"}
]

HEADERS = {
    "User-Agent": "MEI-GeoAI-Benchmark/1.0 (academic geospatial research)"
}

def fetch_category(category_info: dict, limit: int = 15) -> list:
    cat = category_info["category"]
    query = category_info["query"]
    print(f"[*] Fetching strictly London UK entities for: '{cat}'...")
    
    params = {
        "q": query,
        "format": "json",
        "countrycodes": "gb",
        "viewbox": "-0.5,51.2,0.3,51.7",
        "bounded": 1,
        "limit": limit * 2,
        "addressdetails": 1,
        "extratags": 1
    }
    
    try:
        response = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=15)
        response.raise_for_status()
        results = response.json()
        
        valid_records = []
        seen_names = set()
        
        for item in results:
            name = item.get("name")
            display_name = item.get("display_name", "")
            
            if not name:
                parts = [p.strip() for p in display_name.split(",") if p.strip()]
                if parts and not parts[0].isdigit():
                    name = parts[0]
            
            if not name or name.lower() in seen_names or len(name) < 3:
                continue
                
            lat = float(item.get("lat", 0))
            lon = float(item.get("lon", 0))
            
            # Strict London bounding box validation (Lat: 51.2-51.7, Lon: -0.5 to 0.3)
            if not (51.2 <= lat <= 51.7 and -0.5 <= lon <= 0.3):
                continue
                
            address = item.get("address", {})
            extratags = item.get("extratags", {})
            
            record = {
                "osm_id": item.get("osm_id"),
                "osm_type": item.get("osm_type"),
                "category": cat,
                "name": name,
                "coordinates": {
                    "latitude": lat,
                    "longitude": lon
                },
                "address": {
                    "road": address.get("road"),
                    "suburb": address.get("suburb") or address.get("neighbourhood"),
                    "borough": address.get("borough") or address.get("city_district") or address.get("county"),
                    "postcode": address.get("postcode")
                },
                "raw_tags": extratags
            }
            
            valid_records.append(record)
            seen_names.add(name.lower())
            
            if len(valid_records) >= limit:
                break
                
        print(f"    -> [SUCCESS] Collected {len(valid_records)} verified London UK '{cat}' entities.")
        return valid_records
        
    except Exception as e:
        print(f"    [!] Error fetching '{cat}': {e}")
        return []

def main():
    print("=" * 70)
    print("STARTING STRICT GEOAI DATA HARVESTING (London, United Kingdom ONLY)")
    print("Strict Bounding: Lat [51.2, 51.7], Lon [-0.5, 0.3] | Country: GB")
    print("=" * 70)
    
    all_data = {}
    
    for item in AMENITIES:
        cat_records = fetch_category(item, limit=12)
        all_data[item["category"]] = cat_records
        time.sleep(1.0)
        
    with open(RAW_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
        
    total_entities = sum(len(v) for v in all_data.values())
    print("\n" + "=" * 70)
    print(f"[COMPLETED] Successfully Harvested {total_entities} Verified London UK Entities!")
    print(f"File Saved: {RAW_OUTPUT_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    main()
