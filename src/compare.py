"""Comparison of the forest, the baseline tree and the k-NN classifier.

    python compare.py
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

import config as cfg
import data
import metrics
import sweeps

MEASURES = ["macro_f1", "weighted_f1", "accuracy", "balanced_acc"]
PRETTY = {"macro_f1": "Macro $F_1$", "weighted_f1": "Weighted $F_1$",
          "accuracy": "Accuracy", "balanced_acc": "Balanced accuracy"}
MODELS = ["knn", "tree", "forest"]

KNN_DROPPED = ["sloss", "dloss", "spkts", "dpkts", "ct_srv_dst",
               "ct_dst_src_ltm", "ct_src_dport_ltm", "ct_src_ltm",
               "tcprtt", "dwin"]


def conflict_rates(X, y):
    """Share of each class whose records share features with another class."""
    row_hash = pd.util.hash_pandas_object(X, index=False).to_numpy()
    labels = pd.Series(y)
    n_classes = labels.groupby(row_hash).nunique()
    in_conflict = pd.Series(row_hash).map(n_classes).to_numpy() > 1
    return np.array([in_conflict[y == c].mean() * 100
                     for c in range(cfg.N_CLASSES)])


def fold_scores(final, y, folds):
    """Scores of every model on every fold; the forest averaged over seeds."""
    models = json.loads((final / "models.json").read_text())
    df = pd.read_csv(final / "scores.csv")
    valid = df[df["split"] == "valid"]
    cols = [c for c in valid.columns if c in MEASURES
            or c.startswith(("f1_", "precision_", "recall_"))]

    tree = valid[valid["config"] == "tree"].set_index("fold")[cols]
    forest_rows = valid[(valid["config"] == "forest")
                        & (valid["n_trees"] == models["forest"]["n_trees"])]
    forest = forest_rows.groupby("fold")[cols].mean()

    knn_pred = np.load(cfg.BASELINE / "knn_y_pred.npy",
                       allow_pickle=True).astype(np.int64)
    knn = pd.DataFrame([metrics.scores(y[folds == k], knn_pred[folds == k],
                                       cfg.N_CLASSES)
                        for k in range(cfg.N_SPLITS)])[cols]
    knn.index.name = "fold"

    by_seed = forest_rows.groupby("seed")[MEASURES].mean()
    return {"knn": knn, "tree": tree.sort_index(), "forest": forest}, by_seed


def paired_tests(scores):
    """Paired t-tests of the forest against each baseline, Holm corrected."""
    out = {}
    for other in ["tree", "knn"]:
        diff = (scores["forest"]["macro_f1"].to_numpy()
                - scores[other]["macro_f1"].to_numpy())
        t, p = stats.ttest_rel(scores["forest"]["macro_f1"],
                               scores[other]["macro_f1"])
        ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(),
                              scale=stats.sem(diff))
        out[f"forest_vs_{other}"] = {
            "mean_difference": float(diff.mean()),
            "std_difference": float(diff.std(ddof=1)),
            "t": float(t), "df": len(diff) - 1, "p": float(p),
            "ci_95": [float(ci[0]), float(ci[1])],
            "folds_favouring_forest": int((diff > 0).sum()),
        }
    first, second = sorted(out, key=lambda k: out[k]["p"])
    out[first]["p_holm"] = min(1.0, 2 * out[first]["p"])
    out[second]["p_holm"] = max(out[first]["p_holm"], out[second]["p"])
    return out


def measure_choice(out):
    """Selection of every sweep by macro F1 and by accuracy."""
    rows = []
    for sweep in sweeps.ORDER:
        path = sweeps.selection_path(sweep, out)
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        rows.append({"sweep": sweep,
                     "by_macro_f1": s["by_macro_f1"]["selected"],
                     "by_accuracy": s["by_accuracy"]["selected"],
                     "agree": s["measures_agree"]})
    return pd.DataFrame(rows)


def run(out=cfg.OUT):
    final = out / "final"
    tables = out / "tables"
    tables.mkdir(exist_ok=True)

    X, y, _, _ = data.load_clean()
    folds = data.load_folds(y)
    names = data.class_names()
    scores, by_seed = fold_scores(final, y, folds)

    overall = pd.DataFrame({
        f"{m}_{stat}": [getattr(scores[m][c], stat)(**kw) for c in MEASURES]
        for m in MODELS
        for stat, kw in [("mean", {}), ("std", {"ddof": 1})]}, index=MEASURES)
    overall.to_csv(final / "overall.csv")
    print("overall, mean (standard deviation over folds)")
    print(overall.round(4).to_string(), "\n")
    (tables / "overall.tex").write_text("".join(
        f"{PRETTY[c]} & " + " & ".join(
            f"{overall.loc[c, m + '_mean']:.4f} ({overall.loc[c, m + '_std']:.4f})"
            for m in MODELS) + " \\\\\n" for c in MEASURES))

    per_fold = pd.DataFrame({m: scores[m]["macro_f1"].to_numpy()
                             for m in MODELS})
    per_fold.index = pd.RangeIndex(1, cfg.N_SPLITS + 1, name="fold")
    per_fold.to_csv(final / "folds.csv")
    tests = paired_tests(scores)
    (final / "tests.json").write_text(json.dumps(tests, indent=2))
    print("macro F1 per fold")
    print(per_fold.round(4).to_string())
    print(json.dumps(tests, indent=2), "\n")
    (tables / "folds.tex").write_text("".join(
        f"{fold} & " + " & ".join(f"{v:.4f}" for v in row) + " \\\\\n"
        for fold, row in zip(per_fold.index, per_fold.to_numpy())))

    per_class = pd.DataFrame({"class": names, "conflict_pct": conflict_rates(X, y)})
    for m in MODELS:
        for part in ["f1", "precision", "recall"]:
            per_class[f"{m}_{part}"] = [scores[m][f"{part}_{c}"].mean()
                                        for c in range(cfg.N_CLASSES)]
    per_class["forest_minus_tree_f1"] = (per_class["forest_f1"]
                                         - per_class["tree_f1"])

    cm = sum(metrics.confusion_matrix(y, p, cfg.N_CLASSES)
             for p in np.load(final / "forest_y_pred.npy"))
    per_class["forest_to_exploits_pct"] = (
        100 * cm[:, list(names).index("Exploits")] / cm.sum(axis=1))
    per_class = per_class.sort_values("conflict_pct")
    per_class.to_csv(final / "per_class.csv", index=False)
    print("per-class F1 against the conflict rate")
    print(per_class[["class", "conflict_pct", "knn_f1", "tree_f1", "forest_f1",
                     "forest_minus_tree_f1"]].round(3).to_string(index=False))
    rho, p_rho = stats.spearmanr(per_class["conflict_pct"],
                                 per_class["forest_minus_tree_f1"])
    print(f"Spearman correlation of conflict rate and gain: {rho:.3f} "
          f"(p = {p_rho:.3f})\n")
    (tables / "per_class.tex").write_text("".join(
        f"{r['class']} & {r['conflict_pct']:.1f} & {r['knn_f1']:.3f} & "
        f"{r['tree_f1']:.3f} & {r['forest_f1']:.3f} & "
        f"{r['forest_to_exploits_pct']:.0f} \\\\\n"
        for _, r in per_class.iterrows()))

    imp = pd.read_csv(final / "importances.csv").set_index("feature")
    importance = {"gain_rank_correlation": {"rho": float(rho),
                                            "p": float(p_rho)}}
    for m in ["tree", "forest"]:
        ranked = imp[m].sort_values(ascending=False)
        importance[m] = {
            "top3": ranked.index[:3].tolist(),
            "top3_share": float(ranked.iloc[:3].sum()),
            "features_for_90_percent": int((ranked.cumsum() < 0.9).sum() + 1),
            "share_of_knn_dropped": float(imp.loc[KNN_DROPPED, m].sum()),
        }
    print("feature importance")
    print(json.dumps({m: importance[m] for m in ["tree", "forest"]}, indent=2))

    summary = {
        "importance": importance,
        "forest_seed_spread": {c: float(by_seed[c].std(ddof=1))
                               for c in MEASURES},
        "forest_seed_range_macro_f1": [float(by_seed["macro_f1"].min()),
                                       float(by_seed["macro_f1"].max())],
    }
    timing = json.loads((final / "timing.json").read_text())
    knn_timing = json.loads((cfg.BASELINE / "knn_timing.json").read_text())
    summary["rows_per_second"] = {
        "knn": knn_timing["rows_per_second"],
        "tree": timing["tree"]["rows_per_second"],
        "forest": timing["forest"]["rows_per_second"]}
    summary["fit_seconds"] = {"tree": timing["tree"]["fit_s"],
                              "forest": timing["forest"]["fit_s"]}
    choice = measure_choice(out)
    choice.to_csv(final / "measure_choice.csv", index=False)
    (final / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nforest spread across seeds:", summary["forest_seed_spread"])
    print("rows classified per second:", summary["rows_per_second"])
    print("\nselection by macro F1 and by accuracy")
    print(choice.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run(cfg.OUT if args.out is None else cfg.SRC / args.out)
