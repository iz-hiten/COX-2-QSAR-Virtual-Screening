"""Download measured COX-2 (CHEMBL230) and COX-1 (CHEMBL221) IC50 data from the ChEMBL web API.

Writes data/raw_cox2.csv and data/raw_cox1.csv. Nothing here is simulated: every row is a
published experimental measurement deposited in ChEMBL.
"""
import sys, time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
TARGETS = {"cox2": "CHEMBL230", "cox1": "CHEMBL221"}
FIELDS = ["molecule_chembl_id", "canonical_smiles", "standard_type", "standard_relation",
          "standard_value", "standard_units", "pchembl_value", "assay_chembl_id",
          "assay_description", "document_chembl_id", "target_chembl_id", "data_validity_comment"]

out_dir = Path(__file__).resolve().parent.parent / "data"
out_dir.mkdir(exist_ok=True)

for name, tid in TARGETS.items():
    rows, offset = [], 0
    while True:
        params = {"target_chembl_id": tid, "standard_type": "IC50", "limit": 1000, "offset": offset}
        for attempt in range(4):
            try:
                r = requests.get(BASE, params=params, timeout=90)
                r.raise_for_status()
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print("  retry", e, flush=True)
                time.sleep(3)
        js = r.json()
        batch = js["activities"]
        if not batch:
            break
        rows += [{k: a.get(k) for k in FIELDS} for a in batch]
        offset += len(batch)
        print(f"  {name}: {offset}/{js['page_meta']['total_count']}", flush=True)
        if offset >= js["page_meta"]["total_count"]:
            break
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"raw_{name}.csv", index=False)
    print(name, df.shape, flush=True)
