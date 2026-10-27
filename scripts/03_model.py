"""Build and validate QSAR regression models for COX-2 pIC50.

Descriptors: 20 interpretable RDKit physicochemical descriptors + 1024-bit Morgan (ECFP4)
fingerprints. Models: Ridge, PLS, Random Forest, Gradient Boosting, SVR.

Validation follows OECD-style QSAR principles:
  * random 80/20 hold-out test set,
  * 5-fold cross-validation on the training set (Q2),
  * an additional Murcko-scaffold split so that no scaffold is shared between train and test,
  * y-randomisation (20 permutations) to rule out chance correlation,
  * a k-nearest-neighbour applicability domain.

Writes results/model_metrics.json, results/predictions_test.csv and data/descriptors.csv.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
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


def featurise(smiles_list):
    desc, fps, ok = [], [], []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            ok.append(False)
            continue
        row = [_desc_fn[n](m) for n in DESCRIPTORS]
        if not all(np.isfinite(row)):
            ok.append(False)
            continue
        desc.append(row)
        fps.append(np.array(_morgan.GetFingerprintAsNumPy(m), dtype=np.int8))
        ok.append(True)
    return np.array(desc, float), np.array(fps, np.int8), np.array(ok)


def scaffold_split(smiles, test_frac=0.2, seed=SEED):
    """Group compounds by Bemis-Murcko scaffold; whole scaffolds go to one side only."""
    buckets = defaultdict(list)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        core = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""
        buckets[core].append(i)
    groups = sorted(buckets.values(), key=len, reverse=True)
    rng = np.random.default_rng(seed)
    big, small = [g for g in groups if len(g) > 1], [g for g in groups if len(g) == 1]
    rng.shuffle(small)
    test, n_target = [], int(len(smiles) * test_frac)
    for g in big[::-1] + small:          # fill the test set from rarer scaffolds first
        if len(test) >= n_target:
            break
        test += g
    test = set(test)
    return np.array([i not in test for i in range(len(smiles))]), np.array([i in test for i in range(len(smiles))])


def metrics(y, p):
    return {"R2": round(float(r2_score(y, p)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 4),
            "MAE": round(float(mean_absolute_error(y, p)), 4)}


def models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "PLS": make_pipeline(StandardScaler(), PLSRegression(n_components=10)),
        "SVR (RBF)": make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale", epsilon=0.1)),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=1,
                                               n_jobs=-1, random_state=SEED),
    }


df = pd.read_csv(DATA / "cox2_curated.csv")
X_desc, X_fp, ok = featurise(df["clean_smiles"].tolist())
df = df[ok].reset_index(drop=True)
y = df["pIC50"].to_numpy()
X = np.hstack([X_desc, X_fp])
print("dataset", X.shape, flush=True)

pd.DataFrame(X_desc, columns=DESCRIPTORS).assign(pIC50=y, smiles=df["clean_smiles"]).to_csv(
    DATA / "descriptors.csv", index=False)

out = {"n_compounds": int(len(df)), "n_descriptors": len(DESCRIPTORS), "n_fp_bits": 1024,
       "seed": SEED, "splits": {}}

# ---------------- random split ----------------
idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.2, random_state=SEED)
cv = KFold(5, shuffle=True, random_state=SEED)
rand = {}
for name, mdl in models().items():
    mdl.fit(X[tr], y[tr])
    pred_te = np.asarray(mdl.predict(X[te])).ravel()
    q2 = np.asarray(cross_val_predict(mdl, X[tr], y[tr], cv=cv, n_jobs=1)).ravel()
    rand[name] = {"test": metrics(y[te], pred_te),
                  "cv_train": metrics(y[tr], q2) | {"Q2": round(float(r2_score(y[tr], q2)), 4)}}
    print(f"{name:20s} test R2={rand[name]['test']['R2']:.3f} RMSE={rand[name]['test']['RMSE']:.3f} "
          f"Q2={rand[name]['cv_train']['Q2']:.3f}", flush=True)
out["splits"]["random"] = {"n_train": len(tr), "n_test": len(te), "models": rand}

best = max(rand, key=lambda k: rand[k]["test"]["R2"])
out["best_model"] = best
print("best:", best, flush=True)

# predictions of the best model on the hold-out set, for plotting
mdl = models()[best]
mdl.fit(X[tr], y[tr])
pred_te = np.asarray(mdl.predict(X[te])).ravel()

# ---------------- applicability domain (kNN distance in descriptor space) ----------------
from sklearn.neighbors import NearestNeighbors
sc = StandardScaler().fit(X_desc[tr])
nn = NearestNeighbors(n_neighbors=6).fit(sc.transform(X_desc[tr]))
d_tr = nn.kneighbors(sc.transform(X_desc[tr]))[0][:, 1:].mean(1)
threshold = float(np.percentile(d_tr, 95))
d_te = nn.kneighbors(sc.transform(X_desc[te]))[0][:, :5].mean(1)
inside = d_te <= threshold
out["applicability_domain"] = {
    "method": "mean distance to 5 nearest training neighbours in standardised descriptor space",
    "threshold_95th_percentile": round(threshold, 4),
    "test_inside_fraction": round(float(inside.mean()), 4),
    "test_R2_inside": round(float(r2_score(y[te][inside], pred_te[inside])), 4),
    "test_RMSE_inside": round(float(np.sqrt(mean_squared_error(y[te][inside], pred_te[inside]))), 4),
    "test_RMSE_outside": round(float(np.sqrt(mean_squared_error(y[te][~inside], pred_te[~inside]))), 4)
    if (~inside).sum() > 2 else None,
}

pd.DataFrame({"smiles": df["clean_smiles"].to_numpy()[te], "chembl_id": df["molecule_chembl_id"].to_numpy()[te],
              "observed": y[te], "predicted": pred_te, "in_domain": inside}).to_csv(
    RES / "predictions_test.csv", index=False)

# ---------------- scaffold split ----------------
m_tr, m_te = scaffold_split(df["clean_smiles"].tolist())
scaf = {}
for name, mdl2 in models().items():
    mdl2.fit(X[m_tr], y[m_tr])
    p = np.asarray(mdl2.predict(X[m_te])).ravel()
    scaf[name] = metrics(y[m_te], p)
    print(f"scaffold {name:20s} R2={scaf[name]['R2']:.3f}", flush=True)
out["splits"]["scaffold"] = {"n_train": int(m_tr.sum()), "n_test": int(m_te.sum()),
                             "n_unique_scaffolds": len({MurckoScaffold.MurckoScaffoldSmiles(s) for s in df["clean_smiles"] if Chem.MolFromSmiles(s)}),
                             "models": scaf}

# ---------------- y-randomisation ----------------
rng = np.random.default_rng(SEED)
r2s = []
for _ in range(20):
    yp = rng.permutation(y[tr])
    m2 = models()[best]
    m2.fit(X[tr], yp)
    r2s.append(float(r2_score(y[te], np.asarray(m2.predict(X[te])).ravel())))
out["y_randomisation"] = {"n_runs": 20, "mean_R2": round(float(np.mean(r2s)), 4),
                          "max_R2": round(float(np.max(r2s)), 4),
                          "real_R2": rand[best]["test"]["R2"]}
print("y-randomisation mean R2:", round(np.mean(r2s), 4), flush=True)

# ---------------- descriptor importance (RF on descriptors only) ----------------
rf = RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=SEED).fit(X_desc[tr], y[tr])
imp = sorted(zip(DESCRIPTORS, rf.feature_importances_), key=lambda t: -t[1])
out["descriptor_importance"] = [{"descriptor": n, "importance": round(float(v), 4)} for n, v in imp]
out["descriptor_only_test_R2"] = round(float(r2_score(y[te], rf.predict(X_desc[te]))), 4)

(RES / "model_metrics.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ("best_model", "y_randomisation", "applicability_domain")}, indent=2))
