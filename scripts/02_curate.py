"""Curate the raw ChEMBL activity data into a modelling-ready QSAR dataset.

Steps follow accepted practice for QSAR data curation: keep only exact IC50 values in nM,
drop ChEMBL-flagged records, standardise structures (largest fragment, neutralise, canonical
tautomer-insensitive InChIKey), remove inorganics and extreme sizes, then aggregate replicate
measurements per compound by the median pIC50 and discard compounds whose replicates disagree
by more than one log unit.

Writes data/cox2_curated.csv, data/cox1_curated.csv, data/selectivity.csv and
results/curation_log.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
RES.mkdir(exist_ok=True)

MAX_SPREAD_LOG = 1.0          # discard a compound if replicate pIC50 range exceeds this
HEAVY_ATOM_RANGE = (8, 70)
ALLOWED = set("C N O S P F Cl Br I H B Si Se".split())

_chooser = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def standardise(smiles):
    """Return (canonical SMILES, InChIKey) for a cleaned single-component neutral molecule."""
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
    n = mol.GetNumHeavyAtoms()
    if not HEAVY_ATOM_RANGE[0] <= n <= HEAVY_ATOM_RANGE[1]:
        return None
    return Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol)


def curate(name):
    log = {}
    df = pd.read_csv(DATA / f"raw_{name}.csv")
    log["raw_records"] = len(df)

    df = df[df["data_validity_comment"].isna()]
    log["after_validity_filter"] = len(df)

    df = df[(df["standard_units"] == "nM") & (df["standard_relation"] == "=")]
    log["after_exact_nM"] = len(df)

    df = df[pd.to_numeric(df["standard_value"], errors="coerce").notna()]
    df["standard_value"] = df["standard_value"].astype(float)
    df = df[(df["standard_value"] > 0) & (df["standard_value"] < 1e7)]
    log["after_value_range"] = len(df)

    std = df["canonical_smiles"].map(lambda s: standardise(s))
    df = df.assign(clean_smiles=[x[0] if x else None for x in std],
                   inchikey=[x[1] if x else None for x in std])
    df = df[df["clean_smiles"].notna()]
    log["after_structure_standardisation"] = len(df)

    df["pIC50"] = 9.0 - np.log10(df["standard_value"])

    grp = df.groupby("inchikey")
    agg = grp.agg(clean_smiles=("clean_smiles", "first"),
                  molecule_chembl_id=("molecule_chembl_id", "first"),
                  pIC50=("pIC50", "median"),
                  n_measurements=("pIC50", "size"),
                  spread=("pIC50", lambda s: s.max() - s.min()),
                  n_documents=("document_chembl_id", "nunique")).reset_index()
    log["unique_compounds"] = len(agg)

    agg = agg[agg["spread"] <= MAX_SPREAD_LOG]
    log["after_replicate_consistency"] = len(agg)
    log["median_measurements_per_compound"] = float(agg["n_measurements"].median())
    log["pIC50_min"] = round(float(agg["pIC50"].min()), 3)
    log["pIC50_max"] = round(float(agg["pIC50"].max()), 3)
    log["pIC50_mean"] = round(float(agg["pIC50"].mean()), 3)
    log["pIC50_std"] = round(float(agg["pIC50"].std()), 3)

    agg = agg.sort_values("pIC50", ascending=False).reset_index(drop=True)
    agg.to_csv(DATA / f"{name}_curated.csv", index=False)
    print(name, json.dumps(log, indent=2), flush=True)
    return agg, log


cox2, log2 = curate("cox2")
cox1, log1 = curate("cox1")

# selectivity: compounds measured against both isoforms
sel = cox2[["inchikey", "clean_smiles", "molecule_chembl_id", "pIC50"]].merge(
    cox1[["inchikey", "pIC50"]], on="inchikey", suffixes=("_cox2", "_cox1"))
sel["selectivity_index"] = sel["pIC50_cox2"] - sel["pIC50_cox1"]   # log10(IC50_COX1 / IC50_COX2)
sel = sel.sort_values("selectivity_index", ascending=False).reset_index(drop=True)
sel.to_csv(DATA / "selectivity.csv", index=False)

logs = {"cox2": log2, "cox1": log1,
        "paired_both_isoforms": len(sel),
        "selectivity_index_median": round(float(sel["selectivity_index"].median()), 3),
        "max_spread_log_threshold": MAX_SPREAD_LOG,
        "heavy_atom_range": list(HEAVY_ATOM_RANGE)}
(RES / "curation_log.json").write_text(json.dumps(logs, indent=2))
print("paired compounds:", len(sel), flush=True)
