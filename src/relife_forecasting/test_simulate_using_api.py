import requests
from pathlib import Path

url = "http://127.0.0.1:9090/simulate"

params = {
    "archetype": "true",
    "category": "Single Family House",
    "country": "Greece",
    "name": "SFH_Greece_1946_1969",
    "weather_source": "pvgis",
}

headers = {
    "accept": "application/json",
}

epw_path = Path("/Users/dantonucci/Downloads/2020_Athens.epw")

files = {
    "epw_file": (epw_path.name, epw_path.open("rb"), "application/octet-stream"),
}

data = {
    "bui_json": "string",
    "system_json": "string",
}

resp = requests.post(url, params=params, headers=headers, files=files, data=data)
resp.raise_for_status()
print(resp.json())
