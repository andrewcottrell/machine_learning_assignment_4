"""Final evaluation of the selected forest and the tuned baseline tree.

    python final_eval.py
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import config as cfg
import metrics
import sweeps
from forest import RandomForest
from run_experiment import FoldData, ScoreFile, _rows
from tree import DecisionTree


def selected_models(out=cfg.OUT):
    """Parameters of the baseline tree and of the forest, and its resampling."""
    def load(sweep):
        path = sweeps.selection_path(sweep, out)
        if not path.exists():
            raise SystemExit(f"selection of sweep {sweep} is missing")
        return json.loads(path.read_text())["selected"]

    tree = load("0")
    forest = dict(sweeps.FOREST_DEFAULTS)
    for sweep in ["1", "2", "3a", "3b", "4"]:
        forest.update(load(sweep))
    resample = forest.pop("resample")
    return tree, forest, tuple(resample) if resample else None


def run(out=cfg.OUT, n_jobs=None):
    final = out / "final"
    fits = final / "fits"
    fits.mkdir(parents=True, exist_ok=True)

    tree_params, forest_params, resample = selected_models(out)
    n_selected = forest_params["n_trees"]
    grown = max(max(sweeps.TREE_COUNTS), n_selected)
    checkpoints = sorted({n for n in sweeps.TREE_COUNTS} | {n_selected})
    (final / "models.json").write_text(json.dumps(
        {"tree": tree_params, "forest": forest_params, "resample": resample,
         "trees_grown": grown}, indent=2))
    print("tree  :", tree_params)
    print("forest:", forest_params, "resample:", resample, flush=True)

    scores = ScoreFile(final / "scores.csv")
    data = FoldData()
    names = data.num_cols + list(cfg.FLAG_COLS) + data.nom_cols

    for fold in range(cfg.N_SPLITS):
        # train on original records
        Xt, yt, Xv, yv, nominal = data.get(fold)
        path = fits / f"tree_s{cfg.RANDOM_STATE}_f{fold}.npz"
        if not path.exists():
            tree = DecisionTree(**tree_params, seed=cfg.RANDOM_STATE)
            t0 = time.perf_counter()
            tree.fit(Xt, yt, nominal, cfg.N_CLASSES)
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            pred = tree.predict(Xv)
            predict_s = time.perf_counter() - t0
            extra = {"fit_s": fit_s, "predict_s": predict_s,
                     "mean_depth": tree.depth_, "mean_leaves": tree.n_leaves_,
                     "n_train": len(yt)}
            config = {"name": "tree", "params": tree_params, "resample": None}
            scores.append(_rows("final", "tree", config, cfg.RANDOM_STATE,
                                fold, 1, extra,
                                [("train", yt, tree.predict(Xt)),
                                 ("valid", yv, pred)]))
            np.savez(path, pred=pred, importances=tree.importances_)
        print(f"fold {fold}: tree done", flush=True)

    for fold in range(cfg.N_SPLITS):
        Xt, yt, Xv, yv, nominal = data.get(fold, resample)
        for seed in cfg.SEEDS:
            path = fits / f"forest_s{seed}_f{fold}.npz"
            if path.exists():
                continue
            forest = RandomForest(**{**forest_params, "n_trees": grown},
                                  seed=seed, n_jobs=n_jobs)
            t0 = time.perf_counter()
            forest.fit(Xt, yt, nominal, cfg.N_CLASSES)
            fit_s = time.perf_counter() - t0
            staged_v = forest.staged_predict(Xv, checkpoints)
            staged_t = forest.staged_predict(Xt, checkpoints)
            extra = {"fit_s": fit_s, "predict_s": "none",
                     "mean_depth": forest.depths_.mean(),
                     "mean_leaves": ((forest.n_nodes_ + 1) / 2).mean(),
                     "n_train": len(yt)}
            config = {"name": "forest", "params": forest_params,
                      "resample": resample}
            rows = []
            for n, pred_t, pred_v in zip(checkpoints, staged_t, staged_v):
                rows += _rows("final", "forest", config, seed, fold, n, extra,
                              [("train", yt, pred_t), ("valid", yv, pred_v)])
            scores.append(rows)
            np.savez(path, pred=staged_v[checkpoints.index(n_selected)],
                     importances=forest.importances_)
            print(f"fold {fold}: forest seed {seed} done "
                  f"({fit_s:.0f} s for {grown} trees)", flush=True)

    assemble(final, fits, data.folds, names)
    timing(final, data, tree_params, forest_params, n_jobs)


def assemble(final, fits, folds, names):
    def collect(model, seed):
        pred = np.empty(len(folds), dtype=np.int16)
        imps = []
        for fold in range(cfg.N_SPLITS):
            fit = np.load(fits / f"{model}_s{seed}_f{fold}.npz")
            pred[folds == fold] = fit["pred"]
            imps.append(fit["importances"])
        return pred, np.mean(imps, axis=0)

    pred, tree_imp = collect("tree", cfg.RANDOM_STATE)
    np.save(final / "tree_y_pred.npy", pred)
    forest_preds, forest_imps = zip(*[collect("forest", s) for s in cfg.SEEDS])
    np.save(final / "forest_y_pred.npy", np.array(forest_preds))

    pd.DataFrame({"feature": names, "tree": tree_imp,
                  "forest": np.mean(forest_imps, axis=0)}
                 ).to_csv(final / "importances.csv", index=False)


def timing(final, data, tree_params, forest_params, n_jobs):
    path = final / "timing.json"
    if path.exists():
        return
    Xt, yt, Xv, yv, nominal = data.get(0, None)
    out = {"n_train": len(yt), "n_valid": len(yv), "threads": n_jobs or "all"}
    models = {
        "tree": DecisionTree(**tree_params, seed=cfg.RANDOM_STATE),
        "forest": RandomForest(**forest_params, seed=cfg.SEEDS[0],
                               n_jobs=n_jobs),
    }
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(Xt, yt, nominal, cfg.N_CLASSES)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        model.predict(Xv)
        predict_s = time.perf_counter() - t0
        out[name] = {"fit_s": fit_s, "predict_s": predict_s,
                     "rows_per_second": len(yv) / predict_s}
    out["note"] = "fold 0, original training records"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run(cfg.OUT if args.out is None else cfg.SRC / args.out, args.n_jobs)
