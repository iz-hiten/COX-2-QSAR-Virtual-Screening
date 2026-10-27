"""Post-process the scored library: tiered hit lists, known-drug annotation and the
independent-recovery check for clinically used COX-2 inhibitors.

Reads results/library_scored.csv (already produced by 05_screen.py) so no re-prediction is
needed. Appends the extra fields to results/screening_summary.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"

summary = json.loads((RES / "screening_summary.json").read_text())
lib = pd.read_csv(RES / "library_scored.csv")
ind = lib[lib["in_domain"]].copy()

TIERS = {"tier1_strict": (7.0, 1.0), "tier2_prioritised": (6.5, 1.0), "tier3_broad": (6.0, 1.0)}
tiers = {}
for name, (p, s) in TIERS.items():
    sel = ind[(ind["pred_pIC50_cox2"] >= p) & (ind["pred_selectivity_index"] >= s)]
    tiers[name] = {"criteria": f"pred pIC50 >= {p} and SI >= {s}", "n": int(len(sel)),
                   "n_scaffolds": int(sel["murcko_scaffold"].nunique()),
                   "n_known_drugs": int(sel["is_known_drug"].sum()),
                   "median_MolWt": round(float(sel["MolWt"].median()), 1),
                   "median_MolLogP": round(float(sel["MolLogP"].median()), 2),
                   "median_TPSA": round(float(sel["TPSA"].median()), 1),
                   "median_qed": round(float(sel["qed"].median()), 3)}
    sel.sort_values("pred_pIC50_cox2", ascending=False).to_csv(RES / f"hits_{name}.csv", index=False)
summary["tiers"] = tiers

# prediction compression: how far the model's dynamic range falls short of the measured range
summary["prediction_range"] = {
    "library_pred_min": round(float(ind["pred_pIC50_cox2"].min()), 2),
    "library_pred_max": round(float(ind["pred_pIC50_cox2"].max()), 2),
    "library_pred_median": round(float(ind["pred_pIC50_cox2"].median()), 2),
    "training_observed_max": round(float(pd.read_csv(DATA / "cox2_curated.csv")["pIC50"].max()), 2),
}

# Pre-specified panel of clinically used cyclooxygenase inhibitors. Membership and the
# expected selectivity class are fixed from the pharmacology literature before looking at
# any prediction, so this is a blind check rather than a post-hoc selection.
COX_PANEL = {
    "CELECOXIB": "selective", "ROFECOXIB": "selective", "VALDECOXIB": "selective",
    "ETORICOXIB": "selective", "PARECOXIB": "selective", "LUMIRACOXIB": "selective",
    "POLMACOXIB": "selective", "NIMESULIDE": "preferential", "MELOXICAM": "preferential",
    "ETODOLAC": "preferential", "DICLOFENAC": "preferential", "IBUPROFEN": "non-selective",
    "NAPROXEN": "non-selective", "KETOPROFEN": "non-selective", "INDOMETHACIN": "non-selective",
    "PIROXICAM": "non-selective", "FLURBIPROFEN": "non-selective", "ASPIRIN": "non-selective",
    "SULINDAC": "non-selective", "KETOROLAC": "non-selective", "MEFENAMIC ACID": "non-selective",
}
ind["_name"] = ind["pref_name"].astype(str).str.upper()
n_lib = len(ind)
panel_rows = []
for r in ind[ind["_name"].isin(COX_PANEL)].itertuples(index=False):
    panel_rows.append({"name": r.pref_name, "chembl_id": r.chembl_id,
                       "expected_class": COX_PANEL[str(r.pref_name).upper()],
                       "pred_pIC50_cox2": round(r.pred_pIC50_cox2, 2),
                       "pred_selectivity_index": round(r.pred_selectivity_index, 2),
                       "rank_in_library": int((ind["pred_pIC50_cox2"] > r.pred_pIC50_cox2).sum() + 1)})
panel_rows.sort(key=lambda d: -d["pred_selectivity_index"])
summary["cox_panel_recovery"] = {
    "panel_size_defined": len(COX_PANEL),
    "panel_found_in_screened_library": len(panel_rows),
    "library_size_ranked": n_lib,
    "members": panel_rows,
}
# does predicted selectivity separate the selective from the non-selective members?
by_class = {}
for cls in ("selective", "preferential", "non-selective"):
    v = [d["pred_selectivity_index"] for d in panel_rows if d["expected_class"] == cls]
    if v:
        by_class[cls] = {"n": len(v), "mean_SI": round(float(np.mean(v)), 2),
                         "min_SI": round(float(np.min(v)), 2), "max_SI": round(float(np.max(v)), 2)}
summary["cox_panel_recovery"]["by_expected_class"] = by_class
means = [by_class[c]["mean_SI"] for c in ("selective", "preferential", "non-selective") if c in by_class]
summary["cox_panel_recovery"]["class_ordering_monotonic"] = bool(
    len(means) == 3 and means[0] > means[1] > means[2])

# High-scoring named drugs with no established COX activity: candidate false positives,
# reported explicitly rather than quietly omitted.
other = ind[ind["pref_name"].notna() & ~ind["_name"].isin(COX_PANEL)]
other = other.sort_values("pred_pIC50_cox2", ascending=False).head(6)
summary["high_scoring_non_cox_drugs"] = [
    {"name": r.pref_name, "chembl_id": r.chembl_id,
     "pred_pIC50_cox2": round(r.pred_pIC50_cox2, 2),
     "pred_selectivity_index": round(r.pred_selectivity_index, 2)}
    for r in other.itertuples(index=False)]

# top 20 overall, for the supplementary table
top20 = ind.sort_values("pred_pIC50_cox2", ascending=False).head(20)
top20.to_csv(RES / "top20_overall.csv", index=False)
summary["top_20"] = [{"chembl_id": r.chembl_id, "pref_name": (None if pd.isna(r.pref_name) else r.pref_name),
                      "pred_pIC50_cox2": round(r.pred_pIC50_cox2, 2),
                      "pred_selectivity_index": round(r.pred_selectivity_index, 2),
                      "MolWt": round(r.MolWt, 1), "MolLogP": round(r.MolLogP, 2),
                      "TPSA": round(r.TPSA, 1), "qed": round(r.qed, 3),
                      "known_drug": bool(r.is_known_drug)} for r in top20.itertuples(index=False)]

(RES / "screening_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({"tiers": tiers, "named_drug_recovery": summary["named_drug_recovery"][:5],
                  "prediction_range": summary["prediction_range"]}, indent=2))
