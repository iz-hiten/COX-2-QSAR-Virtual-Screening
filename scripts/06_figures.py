"""Generate all paper figures from the real results produced by scripts 02-05."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8,
                     "axes.titlesize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "legend.fontsize": 7, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5})
TEAL, AMBER, GREY, RED = "#1f6f6b", "#b5821f", "#8d8d86", "#b3433a"
W = 3.4

metrics = json.loads((RES / "model_metrics.json").read_text())
screen = json.loads((RES / "screening_summary.json").read_text())
curation = json.loads((RES / "curation_log.json").read_text())


def save(fig, name):
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ---------- Fig 1: workflow ----------
fig, ax = plt.subplots(figsize=(W, 4.1))
ax.set_xlim(0, 10); ax.set_ylim(0, 12.4); ax.axis("off")
steps = [("Data acquisition", f"ChEMBL IC50 records\nCOX-2 {curation['cox2']['raw_records']:,} | COX-1 {curation['cox1']['raw_records']:,}", "#e6efee"),
         ("Curation", f"exact nM values, structure\nstandardisation, replicate check\n{curation['cox2']['after_replicate_consistency']:,} COX-2 compounds", "#e6efee"),
         ("Descriptors", "20 physicochemical descriptors\n+ 1024-bit ECFP4", "#e6efee"),
         ("QSAR modelling", "5 algorithms, random and\nscaffold splits, y-randomisation", "#f6eddd"),
         ("Virtual screening", f"{screen['screening_funnel']['library_downloaded']:,} compounds, novelty,\nPAINS and drug-likeness filters", "#f6eddd"),
         ("Characterisation", "selectivity, applicability\ndomain, physicochemical profile", "#f6eddd")]
y, h, gap = 10.6, 1.55, 0.52
for title, sub, fc in steps:
    ax.add_patch(FancyBboxPatch((0.5, y), 9, h, boxstyle="round,pad=0.03,rounding_size=0.2",
                                fc=fc, ec="#333", lw=0.7))
    ax.text(5, y + h * 0.70, title, ha="center", va="center", fontsize=7.8, weight="bold")
    ax.text(5, y + h * 0.29, sub, ha="center", va="center", fontsize=6.3, linespacing=1.2)
    if y - gap > 0:
        ax.annotate("", (5, y - gap + 0.03), (5, y - 0.03),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333"))
    y -= h + gap
save(fig, "f1_workflow.png")

# ---------- Fig 2: activity distributions ----------
cox2 = pd.read_csv(DATA / "cox2_curated.csv")
cox1 = pd.read_csv(DATA / "cox1_curated.csv")
fig, axes = plt.subplots(1, 2, figsize=(W, 1.8))
bins = np.arange(3.75, 11.25, 0.25)
axes[0].hist(cox2["pIC50"], bins, color=TEAL)
axes[0].set_title(f"COX-2 (n = {len(cox2):,})"); axes[0].set_xlabel("pIC$_{50}$"); axes[0].set_ylabel("compounds")
axes[1].hist(cox1["pIC50"], bins, color=AMBER)
axes[1].set_title(f"COX-1 (n = {len(cox1):,})"); axes[1].set_xlabel("pIC$_{50}$")
for a in axes:
    a.axvline(7.0, color=RED, ls="--", lw=0.8)
fig.tight_layout(w_pad=1.2); save(fig, "f2_activity_distribution.png")

# ---------- Fig 3: chemical space (PCA of descriptors) ----------
desc = pd.read_csv(DATA / "descriptors.csv")
cols = [c for c in desc.columns if c not in ("pIC50", "smiles")]
Z = PCA(2, random_state=42).fit(StandardScaler().fit_transform(desc[cols]))
P = Z.transform(StandardScaler().fit_transform(desc[cols]))
fig, ax = plt.subplots(figsize=(W, 2.5))
sc_ = ax.scatter(P[:, 0], P[:, 1], c=desc["pIC50"], s=3.5, cmap="viridis", alpha=0.75, linewidths=0)
cb = fig.colorbar(sc_, ax=ax, pad=0.02); cb.set_label("pIC$_{50}$", fontsize=7.5); cb.ax.tick_params(labelsize=7)
ax.set_xlabel(f"PC1 ({Z.explained_variance_ratio_[0]*100:.0f}% variance)")
ax.set_ylabel(f"PC2 ({Z.explained_variance_ratio_[1]*100:.0f}%)")
save(fig, "f3_chemical_space.png")

# ---------- Fig 4: model comparison ----------
rand = metrics["splits"]["random"]["models"]
scaf = metrics["splits"]["scaffold"]["models"]
names = list(rand)
fig, ax = plt.subplots(figsize=(W, 2.0))
x = np.arange(len(names)); w = 0.27
ax.bar(x - w, [rand[n]["cv_train"]["Q2"] for n in names], w, color=GREY, label="CV $Q^2$ (train)")
ax.bar(x, [rand[n]["test"]["R2"] for n in names], w, color=TEAL, label="random split $R^2$")
ax.bar(x + w, [scaf[n]["R2"] for n in names], w, color=AMBER, label="scaffold split $R^2$")
ax.set_xticks(x, names, rotation=25, ha="right"); ax.set_ylabel("$R^2$ / $Q^2$"); ax.set_ylim(0, 0.75)
ax.legend(frameon=False, ncol=3, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, 1.22),
          columnspacing=1.0, handlelength=1.2)
save(fig, "f4_model_comparison.png")

# ---------- Fig 5: observed vs predicted ----------
pred = pd.read_csv(RES / "predictions_test.csv")
fig, ax = plt.subplots(figsize=(W, 2.6))
ins, outs = pred[pred["in_domain"]], pred[~pred["in_domain"]]
ax.scatter(ins["observed"], ins["predicted"], s=7, color=TEAL, alpha=0.6, linewidths=0, label="inside AD")
ax.scatter(outs["observed"], outs["predicted"], s=12, facecolors="none", edgecolors=RED, lw=0.6, label="outside AD")
lim = [3.6, 11.0]
ax.plot(lim, lim, color="#555", lw=0.8)
for d in (1, -1):
    ax.plot(lim, [v + d for v in lim], color="#bbb", lw=0.6, ls="--")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("observed pIC$_{50}$"); ax.set_ylabel("predicted pIC$_{50}$")
b = metrics["best_model"]; t = metrics["splits"]["random"]["models"][b]["test"]
ax.set_title(f"{b}: $R^2$ = {t['R2']:.3f}, RMSE = {t['RMSE']:.3f}")
ax.legend(frameon=False, loc="upper left", fontsize=6.8)
save(fig, "f5_observed_predicted.png")

# ---------- Fig 6: y-randomisation ----------
yr = metrics["y_randomisation"]
fig, ax = plt.subplots(figsize=(W, 1.7))
ax.bar(["y-randomised\n(mean of 20)", "y-randomised\n(best of 20)", "real model"],
       [yr["mean_R2"], yr["max_R2"], yr["real_R2"]], color=[GREY, GREY, TEAL], width=0.55)
ax.axhline(0, color="#333", lw=0.6)
ax.set_ylabel("test $R^2$")
for i, v in enumerate([yr["mean_R2"], yr["max_R2"], yr["real_R2"]]):
    ax.text(i, v + (0.02 if v >= 0 else -0.05), f"{v:.3f}", ha="center", fontsize=7)
save(fig, "f6_y_randomisation.png")

# ---------- Fig 7: descriptor importance ----------
imp = metrics["descriptor_importance"][:12][::-1]
fig, ax = plt.subplots(figsize=(W, 2.2))
ax.barh([d["descriptor"] for d in imp], [d["importance"] for d in imp], color=TEAL, height=0.68)
ax.set_xlabel("Random Forest importance")
save(fig, "f7_descriptor_importance.png")

# ---------- Fig 8: COX-2 vs COX-1 selectivity ----------
sel = pd.read_csv(DATA / "selectivity.csv")
fig, ax = plt.subplots(figsize=(W, 2.5))
ax.scatter(sel["pIC50_cox1"], sel["pIC50_cox2"], s=6, color=TEAL, alpha=0.55, linewidths=0)
lim = [3.6, 11.0]
ax.plot(lim, lim, color="#555", lw=0.8, label="equipotent")
ax.plot(lim, [v + 1 for v in lim], color=RED, lw=0.7, ls="--", label="10-fold COX-2 selective")
ax.plot(lim, [v + 2 for v in lim], color=AMBER, lw=0.7, ls=":", label="100-fold")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("COX-1 pIC$_{50}$"); ax.set_ylabel("COX-2 pIC$_{50}$")
ax.set_title(f"{len(sel):,} compounds measured on both isoforms")
ax.legend(frameon=False, loc="lower right", fontsize=6.5)
save(fig, "f8_selectivity.png")

# ---------- Fig 9: screening funnel and score distribution ----------
fn = screen["screening_funnel"]
labels = ["downloaded", "novel +\nstandardised", "PAINS +\ndrug-like", "inside AD", "final hits"]
vals = [fn["library_downloaded"], fn["after_standardisation_and_novelty"],
        fn["after_pains_and_druglikeness"], fn["inside_applicability_domain"],
        fn["hits_pIC50_ge_7_and_selective"]]
fig, axes = plt.subplots(1, 2, figsize=(W, 1.95))
axes[0].barh(range(len(vals))[::-1], vals, color=[TEAL, TEAL, TEAL, AMBER, RED], height=0.66)
axes[0].set_yticks(range(len(vals))[::-1], labels, fontsize=6.3)
axes[0].set_xscale("log"); axes[0].set_xlabel("compounds (log)")
for i, v in enumerate(vals):
    axes[0].text(v * 1.15, len(vals) - 1 - i, f"{v:,}", va="center", fontsize=6.2)
lib = pd.read_csv(RES / "library_scored.csv")
axes[1].hist(lib["pred_pIC50_cox2"], np.arange(3.5, 10.5, 0.2), color=GREY, label="library")
axes[1].axvline(7.0, color=RED, ls="--", lw=0.9)
axes[1].set_xlabel("predicted COX-2 pIC$_{50}$"); axes[1].set_ylabel("compounds")
fig.tight_layout(w_pad=1.0); save(fig, "f9_screening_funnel.png")

# ---------- Fig 10: enrichment curve ----------
t = pd.read_csv(RES / "predictions_test.csv").sort_values("predicted", ascending=False)
act = (t["observed"] >= 7.0).to_numpy().astype(float)
frac = np.arange(1, len(act) + 1) / len(act)
fig, ax = plt.subplots(figsize=(W, 1.9))
ax.plot(frac * 100, np.cumsum(act) / act.sum() * 100, color=TEAL, lw=1.2, label="QSAR ranking")
ax.plot([0, 100], [0, 100], color="#999", lw=0.8, ls="--", label="random")
ax.set_xlabel("% of ranked test set screened"); ax.set_ylabel("% of actives found")
ax.legend(frameon=False, loc="lower right")
save(fig, "f10_enrichment.png")

# ---------- Fig 10b: pre-specified COX drug panel, grouped by expected class ----------
rec = pd.DataFrame(screen["cox_panel_recovery"]["members"])
CLS_COL = {"selective": TEAL, "preferential": AMBER, "non-selective": RED}
rec["order"] = rec["expected_class"].map({"selective": 0, "preferential": 1, "non-selective": 2})
rec = rec.sort_values(["order", "pred_selectivity_index"], ascending=[False, True])
fig, ax = plt.subplots(figsize=(W, 2.9))
ax.barh(range(len(rec)), rec["pred_selectivity_index"],
        color=[CLS_COL[c] for c in rec["expected_class"]], height=0.68)
ax.set_yticks(range(len(rec)), [n.title() for n in rec["name"]], fontsize=6.4)
ax.axvline(0, color="#333", lw=0.7)
ax.axvline(1.0, color="#555", ls="--", lw=0.7)
ax.set_xlabel("predicted selectivity index  (COX-2 − COX-1 pIC$_{50}$)")
ax.set_xlim(-1.9, 3.0)
handles = [plt.Rectangle((0, 0), 1, 1, color=CLS_COL[c]) for c in CLS_COL]
ax.legend(handles, [c + " (known)" for c in CLS_COL], frameon=False, fontsize=6.2,
          loc="lower right", handlelength=1.0)
save(fig, "f10b_drug_recovery.png")

# ---------- Fig 11: structures of the strict (tier 1) hits ----------
hits = pd.read_csv(RES / "hits_tier1_strict.csv")
ms = [Chem.MolFromSmiles(s) for s in hits["smiles"]]
legends = []
for r in hits.itertuples(index=False):
    name = r.pref_name if isinstance(r.pref_name, str) else r.chembl_id
    legends.append(f"{name}\npIC50 {r.pred_pIC50_cox2:.2f} | SI {r.pred_selectivity_index:.2f}")
img = Draw.MolsToGridImage(ms, molsPerRow=2, subImgSize=(360, 270), legends=legends, returnPNG=False)
img.save(FIG / "f11_top_hits.png")

# ---------- Fig 12: hit physicochemical profile (tier 2, n = 83) ----------
hits_all = pd.read_csv(RES / "hits_tier2_prioritised.csv")
props = ["MolWt", "MolLogP", "TPSA", "qed"]
bounds = {"MolWt": (500, "Lipinski"), "MolLogP": (5, "Lipinski"), "TPSA": (140, "Veber"), "qed": (None, None)}
fig, axes = plt.subplots(1, 4, figsize=(W, 1.6))
for ax, p in zip(axes, props):
    ax.boxplot([lib[p].dropna(), hits_all[p].dropna()], widths=0.55, patch_artist=True,
               boxprops=dict(facecolor="#dfe9e8", lw=0.6), medianprops=dict(color=TEAL, lw=1.1),
               flierprops=dict(marker=".", ms=1.5, alpha=0.3), whiskerprops=dict(lw=0.6),
               capprops=dict(lw=0.6))
    ax.set_xticks([1, 2], ["lib.", "hits"], fontsize=6.5)
    ax.set_title(p, fontsize=7.2)
    lim = bounds[p][0]
    if lim is not None:
        ax.axhline(lim, color=RED, ls="--", lw=0.7)
fig.tight_layout(w_pad=0.6); save(fig, "f12_hit_properties.png")

print("figures written to", FIG)
