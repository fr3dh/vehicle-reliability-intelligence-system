import os
import json
import time
import requests

RAW_DIR = "data/raw"
YEAR_START = 2019
YEAR_END = 2025

os.makedirs(RAW_DIR, exist_ok=True)

# Real automobile manufacturers only
VALID_AUTO_MAKES = {
    "ACURA", "ALFA ROMEO", "AUDI", "BMW", "BUICK", "CADILLAC",
    "CHEVROLET", "CHRYSLER", "DODGE", "FORD", "GENESIS", "GMC",
    "HONDA", "HYUNDAI", "INFINITI", "JAGUAR", "JEEP", "KIA",
    "LAND ROVER", "LEXUS", "LINCOLN", "MAZDA", "MERCEDES-BENZ",
    "MINI", "MITSUBISHI", "NISSAN", "PORSCHE", "RAM",
    "SUBARU", "TESLA", "TOYOTA", "VOLKSWAGEN", "VOLVO"
}

def cached_get(url, params=None, cache_path=None):
    """Return cached response if exists; otherwise call API and store."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    r = requests.get(url, params=params)
    data = r.json()
    
    # Rate limiting: wait 1 second after API call
    time.sleep(1)

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)

    return data


def get_years():
    url = "https://api.nhtsa.gov/products/vehicle/modelYears?issueType=c"
    cache = f"{RAW_DIR}/years.json"
    data = cached_get(url, cache_path=cache)

    years = []
    for x in data.get("results", []):
        y = x.get("modelYear")
        if y.isdigit():
            y = int(y)
            if YEAR_START <= y <= YEAR_END:
                years.append(y)

    return sorted(years)


def get_makes(year):
    cache = f"{RAW_DIR}/{year}_makes.json"
    url = f"https://api.nhtsa.gov/products/vehicle/makes?modelYear={year}&issueType=c"
    data = cached_get(url, cache_path=cache)

    makes = []
    for x in data.get("results", []):
        make = x.get("make", "").upper()
        if make in VALID_AUTO_MAKES:
            makes.append(make)

    return makes


def get_models(year, make):
    safe_make = make.replace("/", "_")
    cache = f"{RAW_DIR}/{year}_{safe_make}_models.json"
    url = f"https://api.nhtsa.gov/products/vehicle/models?modelYear={year}&make={make}&issueType=c"

    data = cached_get(url, cache_path=cache)
    return [x["model"] for x in data.get("results", [])]


def fetch_complaints(year, make, model):
    safe_make = make.replace("/", "_")
    safe_model = model.replace("/", "_")

    cache = f"{RAW_DIR}/{year}_{safe_make}_{safe_model}.json"
    url = "https://api.nhtsa.gov/complaints/complaintsByVehicle"

    data = cached_get(
        url,
        params={"make": make, "model": model, "modelYear": year},
        cache_path=cache
    )
    return data


def main():
    years = get_years()
    print("Years to fetch:", years)

    for year in years:
        makes = get_makes(year)
        print(f"[{year}] Auto makes: {makes}")

        for make in makes:
            models = get_models(year, make)
            for model in models:
                print(f"Fetching complaints for {year} {make} {model}...")
                fetch_complaints(year, make, model)

    print("\n🎉 DONE: All raw complaints stored in data/raw/")


if __name__ == "__main__":
    main()
