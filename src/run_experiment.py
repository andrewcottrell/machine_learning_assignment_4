"""Runs one sweep over the folds and seeds and appends to its score file.

    python run_experiment.py 1
    python run_experiment.py all      every sweep, selecting after each

One row is written per configuration, seed, fold, split and tree count. A fit
whose rows are already in the file is skipped, so an interrupted run resumes.
"""
import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config as cfg
import data
import metrics
import sweeps
from forest import RandomForest
from tree import DecisionTree

PARAM_COLS = ["max_features", "bootstrap", "max_depth", "min_samples_leaf",
              "criterion", "class_weight", "class_sample_size"]
KEY_COLS = ["sweep", "config", "model", "seed", "fold", "split", "n_trees"]
INFO_COLS = ["resample", "resample_target", "fit_s", "predict_s",
             "mean_depth", "mean_leaves", "n_train"]


def columns():
    measure_cols = list(metrics.scores([0], [0], cfg.N_CLASSES))
    return KEY_COLS + PARAM_COLS + INFO_COLS + measure_cols


def scores_path(sweep, out=cfg.OUT):
    return out / f"sweep_{sweep}" / "scores.csv"


class ScoreFile:
    """Append-only score file that knows which fits it already holds."""

    def __init__(self, path):
        self.path = path
        self.cols = columns()
        self.done = set()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            text = path.read_text()
            if text and not text.endswith("\n"):
                path.write_text(text[:text.rfind("\n") + 1])
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    self.done.add((row["config"], int(row["seed"]),
                                   int(row["fold"])))
        else:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.cols)

    def append(self, rows):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.cols)
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        for row in rows:
            self.done.add((row["config"], row["seed"], row["fold"]))


class FoldData:

    def __init__(self):
        self.X, self.y, self.num_cols, self.nom_cols = data.load_clean()
        self.folds = data.load_folds(self.y)
        self.cache = {}
        self.cached_resample = None

    def get(self, fold, resample=None):
        if resample != self.cached_resample:
            self.cache.clear()
            self.cached_resample = resample
        if fold not in self.cache:
            trn, val = self.folds != fold, self.folds == fold
            X_trn, y_trn = self.X[trn], self.y[trn]
            if resample is not None:
                import resample as rs
                kind, target = resample
                X_trn, y_trn = rs.load_or_make(fold, kind, target, X_trn,
                                               y_trn, self.num_cols,
                                               self.nom_cols)
            enc = data.BinEncoder(self.num_cols, self.nom_cols).fit(X_trn)
            self.cache[fold] = (enc.transform(X_trn), y_trn,
                                enc.transform(self.X[val]), self.y[val],
                                enc.is_nominal)
        return self.cache[fold]


def _rows(sweep, model, config, seed, fold, n_trees, extra, splits):
    params = {c: config["params"].get(c) for c in PARAM_COLS}
    kind, target = config["resample"] or (None, None)
    rows = []
    for split, y_true, y_pred in splits:
        row = {"sweep": sweep, "config": config["name"], "model": model,
               "seed": seed, "fold": fold, "split": split, "n_trees": n_trees,
               **params, "resample": kind, "resample_target": target, **extra,
               **metrics.scores(y_true, y_pred, cfg.N_CLASSES)}
        rows.append({k: ("none" if v is None else v) for k, v in row.items()})
    return rows


def run_tree(sweep, config, seed, fold, fold_data):
    Xt, yt, Xv, yv, nominal = fold_data
    tree = DecisionTree(**config["params"], seed=seed)
    t0 = time.perf_counter()
    tree.fit(Xt, yt, nominal, cfg.N_CLASSES)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_v = tree.predict(Xv)
    predict_s = time.perf_counter() - t0
    extra = {"fit_s": fit_s, "predict_s": predict_s,
             "mean_depth": tree.depth_, "mean_leaves": tree.n_leaves_,
             "n_train": len(yt)}
    return _rows(sweep, "tree", config, seed, fold, 1, extra,
                 [("train", yt, tree.predict(Xt)), ("valid", yv, pred_v)])


def run_forest(sweep, config, seed, fold, fold_data, checkpoints, n_jobs):
    Xt, yt, Xv, yv, nominal = fold_data
    forest = RandomForest(**config["params"], seed=seed, n_jobs=n_jobs)
    t0 = time.perf_counter()
    forest.fit(Xt, yt, nominal, cfg.N_CLASSES)
    fit_s = time.perf_counter() - t0

    checkpoints = checkpoints or [forest.n_trees]
    t0 = time.perf_counter()
    staged_v = forest.staged_predict(Xv, checkpoints)
    predict_s = time.perf_counter() - t0
    staged_t = forest.staged_predict(Xt, checkpoints)

    extra = {"fit_s": fit_s, "predict_s": predict_s,
             "mean_depth": forest.depths_.mean(),
             "mean_leaves": ((forest.n_nodes_ + 1) / 2).mean(),
             "n_train": len(yt)}
    rows = []
    for n_trees, pred_t, pred_v in zip(sorted(checkpoints), staged_t,
                                       staged_v):
        rows += _rows(sweep, "forest", config, seed, fold, n_trees, extra,
                      [("train", yt, pred_t), ("valid", yv, pred_v)])
    return rows


def run_sweep(sweep, out=cfg.OUT, n_jobs=None, max_fits=None, folds=None):
    n_jobs = n_jobs or os.cpu_count()
    spec = sweeps.get_sweep(sweep, out)
    scores = ScoreFile(scores_path(sweep, out))
    fold_data = FoldData() if folds is None else folds

    todo = [(config, seed, fold)
            for config in spec["configs"]
            for seed in spec["seeds"]
            for fold in range(cfg.N_SPLITS)
            if (config["name"], seed, fold) not in scores.done]
    total = (len(spec["configs"]) * len(spec["seeds"]) * cfg.N_SPLITS)
    print(f"sweep {sweep}: {len(todo)} of {total} fits to run, "
          f"{n_jobs} threads -> {scores.path}", flush=True)
    todo = todo[:max_fits]

    start = time.perf_counter()

    def report(i):
        elapsed = time.perf_counter() - start
        left = elapsed / i * (len(todo) - i)
        if i in (1, len(todo)) or i % max(1, len(todo) // 20) == 0:
            print(f"  {i}/{len(todo)} fits, {elapsed / 60:.1f} min elapsed, "
                  f"about {left / 60:.0f} min left", flush=True)

    if spec["model"] == "tree":
        for fold in range(cfg.N_SPLITS):
            fold_data.get(fold)

        def one(job):
            config, seed, fold = job
            return run_tree(sweep, config, seed, fold, fold_data.get(fold))

        with ThreadPoolExecutor(n_jobs) as pool:
            for i, rows in enumerate(pool.map(one, todo), start=1):
                scores.append(rows)
                report(i)
    else:
        for i, (config, seed, fold) in enumerate(todo, start=1):
            rows = run_forest(sweep, config, seed, fold,
                              fold_data.get(fold, config["resample"]),
                              spec["checkpoints"], n_jobs)
            scores.append(rows)
            report(i)

    return fold_data


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sweep", help="0, 1, 2, 3a, 3b, 4 or all")
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--max-fits", type=int, default=None,
                        help="stop after this many fits (for a trial run)")
    parser.add_argument("--out", default=None,
                        help="result directory, default results/")
    args = parser.parse_args()

    out = cfg.OUT if args.out is None else cfg.SRC / args.out

    if args.sweep != "all":
        run_sweep(args.sweep, out, args.n_jobs, args.max_fits)
        return

    import select_config
    fold_data = None
    for sweep in sweeps.ORDER:
        fold_data = run_sweep(sweep, out, args.n_jobs, folds=fold_data)
        selection = select_config.select(sweep, out)
        if selection["by_macro_f1"]["edge"] and sweep != "0":
            raise SystemExit(
                f"sweep {sweep} selected an edge value. Extend the values in "
                "sweeps.py, log it in PLAN.md and run again.")


if __name__ == "__main__":
    main()
