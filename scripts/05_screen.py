"""QSAR-guided virtual screening and in silico characterisation of the hits.

Pipeline:
  1. Standardise the library with the same routine used for the training data.
  2. Remove every compound already present in the COX-2 or COX-1 training sets, so the screen is
     prospective with respect to the model.
  3. Remove PAINS (pan-assay interference) substructures and compounds failing Lipinski/Veber
     drug-likeness.
  4. Predict COX-2 pIC50 with the selected QSAR model retrained on all curated COX-2 data, and
     predict COX-1 pIC50 with the same architecture trained on the COX-1 set, giving a predicted
     selectivity index.
  5. Keep only compounds inside the applicability domain.
  6. Characterise the ranked hits (physicochemical profile, drug-likeness, scaffold).

A retrospective enrichment test on the held-out COX-2 test set is also reported, to show the
ranking is meaningful before it is applied to unseen compounds.

Writes results/screening_summary.json, results/top_hits.csv and results/library_scored.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
SEED = 42

DESCRIPTORS = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
               "NumAromaticRings", "NumAliphaticRings", "FractionCSP3", "HeavyAtomCount",
               "RingCount", "NHOHCount", "NOCount", "MolMR", "BalabanJ", "BertzCT",
               "HallKierAlpha", "Kappa2", "Chi1v", "qed"]
_desc_fn = {n: f for n, f in Descriptors.descList if n in DESCRIPTORS}
_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
_chooser = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()
_params = FilterCatalogParams()
_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_pains = FilterCatalog(_params)
ALLOWED = set("C N O S P F Cl Br I H B Si Se".split())


def standardise(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _chooser.choose(mol)
        mol = _uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    syms = {a.GetSymbol() for a in mol.GetAtoms()}
    if not syms <= ALLOWED or "C" not in syms:
        return None
    if not 8 <= mol.GetNumHeavyAtoms() <= 70:
        return None
    return mol


def featurise(mols):
    d = np.array([[_desc_fn[n](m) for n in DESCRIPTORS] for m in mols], float)
    f = np.array([np.array(_morgan.GetFingerprintAsNumPy(m), dtype=np.int8) for m in mols], np.int8)
    return d, f


def build(name):
    """Fit the selected architecture on all curated data for one isoform."""
    df = pd.read_csv(DATA / f"{name}_curated.csv")
    mols, keep = [], []
    for s in df["clean_smiles"]:
        m = Chem.MolFromSmiles(s)
        keep.append(m is not None)
        if m is not None:
            mols.append(m)
    df = df[keep].reset_index(drop=True)
    d, f = featurise(mols)
    X, y = np.hstack([d, f]), df["pIC50"].to_numpy()
    mdl = make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale", epsilon=0.1)).fit(X, y)
    return mdl, d, df


summary = {}

# ---------------- models for both isoforms ----------------
cox2_model, cox2_desc, cox2_df = build("cox2")
cox1_model, _, cox1_df = build("cox1")
summary["training"] = {"cox2_compounds": len(cox2_df), "cox1_compounds": len(cox1_df),
                       "architecture": "SVR (RBF) on 20 descriptors + 1024-bit ECFP4"}

known = set(cox2_df["inchikey"]) | set(cox1_df["inchikey"])

# applicability domain built on the COX-2 training descriptors
sc = StandardScaler().fit(cox2_desc)
nn = NearestNeighbors(n_neighbors=6).fit(sc.transform(cox2_desc))
d_tr = nn.kneighbors(sc.transform(cox2_desc))[0][:, 1:].mean(1)
ad_threshold = float(np.percentile(d_tr, 95))

# ---------------- retrospective enrichment on the held-out test set ----------------
test = pd.read_csv(RES / "predictions_test.csv")
order = test.sort_values("predicted", ascending=False)
actives = (order["observed"] >= 7.0).to_numpy()
rate = actives.mean()
enr = {}
for pct in (1, 5, 10):
    k = max(1, int(len(order) * pct / 100))
    enr[f"EF_{pct}%"] = round(float(actives[:k].mean() / rate), 2)
    enr[f"hit_rate_top_{pct}%"] = round(float(actives[:k].mean()), 3)
enr["baseline_active_rate"] = round(float(rate), 3)
enr["active_threshold_pIC50"] = 7.0
summary["retrospective_enrichment"] = enr
print("enrichment", enr, flush=True)

# ---------------- library preparation ----------------
lib = pd.read_csv(DATA / "library_raw.csv").drop_duplicates("chembl_id")
drugs = pd.read_csv(DATA / "drugs_raw.csv").drop_duplicates("chembl_id")
drug_ids = set(drugs["chembl_id"])
lib = pd.concat([lib, drugs], ignore_index=True).drop_duplicates("chembl_id")
steps = {"library_downloaded": len(lib)}

mols, rows = [], []
for rec in lib.itertuples(index=False):
    m = standardise(rec.smiles)
    if m is None:
        continue
    key = Chem.MolToInchiKey(m)
    if key in known:
        continue
    mols.append(m)
    rows.append({"chembl_id": rec.chembl_id, "pref_name": rec.pref_name,
                 "max_phase": rec.max_phase, "inchikey": key,
                 "smiles": Chem.MolToSmiles(m)})
lib = pd.DataFrame(rows).drop_duplicates("inchikey").reset_index(drop=True)
mols = [m for m, keep in zip(mols, ~pd.DataFrame(rows).duplicated("inchikey"))  if keep]
steps["after_standardisation_and_novelty"] = len(lib)
print("library after novelty filter:", len(lib), flush=True)

# PAINS + drug-likeness
flags = []
for m in mols:
    mw, logp = Descriptors.MolWt(m), Crippen.MolLogP(m)
    hbd, hba = Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m)
    rotb, tpsa = Lipinski.NumRotatableBonds(m), Descriptors.TPSA(m)
    lip = sum([mw <= 500, logp <= 5, hbd <= 5, hba <= 10])
    flags.append({"pains": _pains.HasMatch(m), "lipinski_violations": 4 - lip,
                  "veber_pass": rotb <= 10 and tpsa <= 140,
                  "MolWt": mw, "MolLogP": logp, "TPSA": tpsa, "NumHDonors": hbd,
                  "NumHAcceptors": hba, "NumRotatableBonds": rotb,
                  "qed": Descriptors.qed(m)})
lib = pd.concat([lib, pd.DataFrame(flags)], axis=1)
mask = (~lib["pains"]) & (lib["lipinski_violations"] <= 1) & (lib["veber_pass"])
steps["pains_removed"] = int(lib["pains"].sum())
lib, mols = lib[mask].reset_index(drop=True), [m for m, k in zip(mols, mask) if k]
steps["after_pains_and_druglikeness"] = len(lib)
print("after PAINS/drug-likeness:", len(lib), flush=True)

# ---------------- prediction ----------------
d, f = featurise(mols)
X = np.hstack([d, f])
lib["pred_pIC50_cox2"] = cox2_model.predict(X)
lib["pred_pIC50_cox1"] = cox1_model.predict(X)
lib["pred_selectivity_index"] = lib["pred_pIC50_cox2"] - lib["pred_pIC50_cox1"]

dist = nn.kneighbors(sc.transform(d))[0][:, :5].mean(1)
lib["ad_distance"] = dist
lib["in_domain"] = dist <= ad_threshold
steps["inside_applicability_domain"] = int(lib["in_domain"].sum())
print("inside AD:", int(lib["in_domain"].sum()), flush=True)

lib["murcko_scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
lib["is_known_drug"] = lib["chembl_id"].isin(drug_ids)
lib.to_csv(RES / "library_scored.csv", index=False)

hits = lib[lib["in_domain"] & (lib["pred_pIC50_cox2"] >= 7.0) &
           (lib["pred_selectivity_index"] >= 1.0)].copy()
hits = hits.sort_values("pred_pIC50_cox2", ascending=False).reset_index(drop=True)
steps["hits_pIC50_ge_7_and_selective"] = len(hits)
hits.head(50).to_csv(RES / "top_hits.csv", index=False)

summary["screening_funnel"] = steps
summary["ad_threshold"] = round(ad_threshold, 4)
summary["hit_criteria"] = {"pred_pIC50_cox2": ">= 7.0", "pred_selectivity_index": ">= 1.0",
                           "applicability_domain": "inside", "pains": "none",
                           "lipinski_violations": "<= 1", "veber": "pass"}
top = hits.head(10)
summary["top_10"] = [{"chembl_id": r.chembl_id, "pref_name": r.pref_name,
                      "pred_pIC50_cox2": round(r.pred_pIC50_cox2, 2),
                      "pred_selectivity_index": round(r.pred_selectivity_index, 2),
                      "MolWt": round(r.MolWt, 1), "MolLogP": round(r.MolLogP, 2),
                      "TPSA": round(r.TPSA, 1), "qed": round(r.qed, 3),
                      "known_drug": bool(r.is_known_drug)} for r in top.itertuples(index=False)]
summary["n_unique_hit_scaffolds"] = int(hits["murcko_scaffold"].nunique())
summary["hits_that_are_known_drugs"] = int(hits["is_known_drug"].sum())

(RES / "screening_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: summary[k] for k in ("screening_funnel", "n_unique_hit_scaffolds")}, indent=2))
