import os
import json
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_all_raw():
    rows = []

    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(RAW_DIR, filename), "r") as f:
                data = json.load(f)
                rows.extend(data.get("results", []))

    print(f"Loaded {len(rows)} raw rows.")
    return pd.json_normalize(rows)


def clean_df(df):
    records = []

    for _, row in df.iterrows():
        summary = row.get("summary", "")
        if not isinstance(summary, str) or len(summary.strip()) < 20:
            continue

        products = row.get("products", [])
        prod = products[0] if products and isinstance(products[0], dict) else {}

        try:
            year = int(prod.get("productYear"))
        except:
            year = None

        records.append({
            "odi": row.get("odiNumber"),
            "summary": summary.strip(),
            "make": prod.get("productMake"),
            "model": prod.get("productModel"),
            "year": year,
            "manufacturer": row.get("manufacturer"),
            "components": row.get("components"),
            "dateIncident": row.get("dateOfIncident"),
            "crash": bool(row.get("crash", False)),
            "fire": bool(row.get("fire", False)),
        })

    clean = pd.DataFrame(records)
    print(f"Cleaned complaints: {len(clean)}")
    return clean


def main():
    raw_df = load_all_raw()
    clean = clean_df(raw_df)

    out_path = os.path.join(PROCESSED_DIR, "complaints_clean.csv")
    clean.to_csv(out_path, index=False)

    print(f"\n🎉 Clean dataset saved at: {out_path}")
    print(f"Final chunk count: {len(clean)}")


if __name__ == "__main__":
    main()
