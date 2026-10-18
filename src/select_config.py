"""Selects the simplest configuration within one standard error of the best.

    python select_config.py 2
"""
import argparse
import json

import numpy as np
import pandas as pd

import config as cfg
import sweeps
from run_experiment import scores_path

MEASURES = {"by_macro_f1": "macro_f1", "by_accuracy": "accuracy"}


def summarise(sweep, out=cfg.OUT):
    """Mean and standard error of every configuration on the validation folds."""
    df = pd.read_csv(scores_path(sweep, out))
    key = "n_trees" if sweep == "1" else "config"
    valid = df[df["split"] == "valid"]
    train = df[df["split"] == "train"]

    table = {}
    for name, group in valid.groupby(key):
        row = {"fits": len(group), "mean_leaves": group["mean_leaves"].mean()}
        for measure in MEASURES.values():
            per_fold = group.groupby("fold")[measure].mean()
            row[measure] = per_fold.mean()
            row[f"{measure}_se"] = per_fold.std(ddof=1) / np.sqrt(len(per_fold))
        table[name] = row
    table = pd.DataFrame(table).T
    table["train_macro_f1"] = train.groupby(key)["macro_f1"].mean()
    return table


def _simplest(table, measure, candidates, order):
    sub = table.loc[candidates]
    best = sub[measure].idxmax()
    threshold = sub.loc[best, measure] - sub.loc[best, f"{measure}_se"]
    within = [c for c in candidates if sub.loc[c, measure] >= threshold]
    return min(within, key=order), best, threshold


def _apply_rule(sweep, table, measure, configs):
    """Returns (selected name, best name, threshold, parameters, edge flag)."""
    names = list(table.index)
    inf = float("inf")

    if sweep == "1":
        chosen, best, thr = _simplest(table, measure, names, lambda n: n)
        return (chosen, best, thr, {"n_trees": int(chosen)},
                chosen == max(sweeps.TREE_COUNTS))

    params = {c["name"]: c["params"] for c in configs}

    if sweep == "0":
        chosen, best, thr = _simplest(
            table, measure, names, lambda n: table.loc[n, "mean_leaves"])
        p = params[chosen]
        edge = (p["max_depth"] == min(d for d in sweeps.DEPTHS if d)
                or p["min_samples_leaf"] == max(sweeps.LEAVES))
        return chosen, best, thr, p, edge

    if sweep == "2":        
        standard = [n for n in names if params[n]["bootstrap"] == "standard"]
        none = [n for n in names if params[n]["bootstrap"] == "none"]
        best_std = table.loc[standard, measure].idxmax()
        best_none = table.loc[none, measure].idxmax()
        lead = table.loc[best_none, measure] - table.loc[best_std, measure]
        group = none if lead > table.loc[best_none, f"{measure}_se"] else standard
        chosen, best, thr = _simplest(
            table, measure, group, lambda n: params[n]["max_features"])
        p = params[chosen]
        return (chosen, best, thr, {"max_features": p["max_features"],
                                    "bootstrap": p["bootstrap"]}, False)

    if sweep == "3a":
        chosen, best, thr = _simplest(
            table, measure, names, lambda n: params[n]["max_depth"] or inf)
        depth = params[chosen]["max_depth"]
        return (chosen, best, thr, {"max_depth": depth},
                depth == min(d for d in sweeps.DEPTHS if d))

    if sweep == "3b":
        chosen, best, thr = _simplest(
            table, measure, names,
            lambda n: -params[n]["min_samples_leaf"])
        leaf = params[chosen]["min_samples_leaf"]
        return (chosen, best, thr, {"min_samples_leaf": leaf},
                leaf == max(sweeps.LEAVES))

    if sweep == "4":
        best = table[measure].idxmax()
        thr = table.loc[best, measure] - table.loc[best, f"{measure}_se"]
        chosen = "none" if table.loc["none", measure] >= thr else best
        resample = {c["name"]: c["resample"] for c in configs}[chosen]
        p = params[chosen]
        return (chosen, best, thr,
                {"bootstrap": p["bootstrap"],
                 "class_weight": p["class_weight"],
                 "class_sample_size": p["class_sample_size"],
                 "resample": resample}, False)

    raise SystemExit(f"no selection rule for sweep {sweep}")


def select(sweep, out=cfg.OUT):
    spec = sweeps.get_sweep(sweep, out)
    table = summarise(sweep, out)

    expected = len(spec["seeds"]) * cfg.N_SPLITS
    if sweep != "1" and len(table) != len(spec["configs"]):
        raise SystemExit(f"sweep {sweep} is incomplete: {len(table)} of "
                         f"{len(spec['configs'])} configurations have scores")
    if (table["fits"] != expected).any():
        raise SystemExit(f"sweep {sweep} is incomplete: some configurations "
                         f"have fewer than {expected} fits")

    result = {"sweep": sweep}
    for label, measure in MEASURES.items():
        chosen, best, thr, params, edge = _apply_rule(
            sweep, table, measure, spec["configs"])
        result[label] = {
            "selected": str(chosen), "best": str(best),
            "best_mean": float(table.loc[best, measure]),
            "best_se": float(table.loc[best, f"{measure}_se"]),
            "threshold": float(thr),
            "selected_mean": float(table.loc[chosen, measure]),
            "params": params, "edge": bool(edge),
        }
    result["selected"] = result["by_macro_f1"]["params"]
    result["measures_agree"] = (result["by_macro_f1"]["selected"]
                                == result["by_accuracy"]["selected"])

    sweeps.selection_path(sweep, out).write_text(json.dumps(result, indent=2))

    show = table[["macro_f1", "macro_f1_se", "accuracy", "accuracy_se",
                  "train_macro_f1", "mean_leaves"]].astype(float)
    if sweep == "0":
        show = show.sort_values("macro_f1", ascending=False).head(15)
    print(f"\nsweep {sweep}" + (" (best 15 by macro F1)" if sweep == "0" else ""))
    print(show.round(4).to_string())
    for label in MEASURES:
        r = result[label]
        print(f"{label:<12}: selected {r['selected']} "
              f"({r['selected_mean']:.4f}); best {r['best']} "
              f"({r['best_mean']:.4f}, se {r['best_se']:.4f})"
              + ("  ** EDGE VALUE **" if r["edge"] else ""))
    print(f"measures agree: {result['measures_agree']}\n", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sweep", help="0, 1, 2, 3a, 3b or 4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    select(args.sweep, cfg.OUT if args.out is None else cfg.SRC / args.out)


if __name__ == "__main__":
    main()
