"""Download the virtual screening library from ChEMBL.

Two sets are pulled:
  * a diverse drug-like library (MW 200-500, clogP <= 5) used as the screening deck;
  * all compounds that have reached clinical phase 3 or 4, so that any hit which is already an
    approved or late-stage drug can be flagged as a repurposing candidate.

Compounds already present in the COX-2 or COX-1 training data are removed later (script 05),
so the screen is genuinely prospective with respect to the model's training set.
"""
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TARGET_N = 20000


def fetch(params, cap):
    rows, offset = [], 0
    while len(rows) < cap:
        p = dict(params, limit=1000, offset=offset)
        for attempt in range(4):
            try:
                r = requests.get(BASE, params=p, timeout=120)
                r.raise_for_status()
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print("  retry", e, flush=True)
                time.sleep(3)
        js = r.json()
        batch = js["molecules"]
        if not batch:
            break
        for m in batch:
            s = (m.get("molecule_structures") or {}).get("canonical_smiles")
            if not s:
                continue
            props = m.get("molecule_properties") or {}
            rows.append({"chembl_id": m["molecule_chembl_id"], "smiles": s,
                         "pref_name": m.get("pref_name"), "max_phase": m.get("max_phase"),
                         "mw": props.get("full_mwt"), "alogp": props.get("alogp")})
        offset += len(batch)
        print(f"  {len(rows)} compounds", flush=True)
        if offset >= js["page_meta"]["total_count"]:
            break
    return pd.DataFrame(rows)


print("screening deck...", flush=True)
deck = fetch({"molecule_properties__full_mwt__gte": 200,
              "molecule_properties__full_mwt__lte": 500,
              "molecule_properties__alogp__lte": 5,
              "molecule_type": "Small molecule"}, TARGET_N)
deck.to_csv(DATA / "library_raw.csv", index=False)
print("deck", deck.shape, flush=True)

print("clinical / approved drugs...", flush=True)
drugs = fetch({"max_phase__gte": 3, "molecule_type": "Small molecule"}, 6000)
drugs.to_csv(DATA / "drugs_raw.csv", index=False)
print("drugs", drugs.shape, flush=True)
