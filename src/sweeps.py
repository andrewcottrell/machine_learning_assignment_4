"""Definitions of the parameter and experiment sweeps."""
import json
from itertools import product

import config as cfg

ORDER = ["0", "1", "2", "3a", "3b", "4"]

TREE_COUNTS = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]
FEATURES = [1, 3, 6, 10, 14, 21, 42]
DEPTHS = [5, 8, 10, 12, 15, 20, 25, 30, 40, None]
LEAVES = [1, 2, 5, 10, 20, 50]
CRITERIA = ["gini", "entropy"]
BALANCED_SIZES = [None, 500, 1000, 2500, 5000]
SMOTE_TARGETS = [5000, 20000]

FOREST_DEFAULTS = dict(n_trees=max(TREE_COUNTS), max_features=6,
                       bootstrap="standard", max_depth=None,
                       min_samples_leaf=1, criterion="entropy",
                       class_weight=None, class_sample_size=None)

REQUIRES = {"0": [], "1": [], "2": ["1"], "3a": ["1", "2"],
            "3b": ["1", "2", "3a"], "4": ["1", "2", "3a", "3b"]}


def selection_path(sweep, out=cfg.OUT):
    return out / f"sweep_{sweep}" / "selection.json"


def selected_values(sweep, out=cfg.OUT):
    params = dict(FOREST_DEFAULTS)
    for earlier in REQUIRES[sweep]:
        path = selection_path(earlier, out)
        if not path.exists():
            raise SystemExit(f"sweep {sweep} needs the selection of sweep "
                             f"{earlier}: run select_config.py {earlier}")
        params.update(json.loads(path.read_text())["selected"])
    return params


def _config(name, params, resample=None):
    return {"name": name, "params": params, "resample": resample}


def get_sweep(sweep, out=cfg.OUT):
    if sweep == "0":
        configs = []
        for crit, depth, leaf, cw in product(CRITERIA, DEPTHS, LEAVES,
                                             [None, "balanced"]):
            name = f"{crit}_d{depth or 'none'}_l{leaf}_{cw or 'unweighted'}"
            configs.append(_config(name, dict(
                criterion=crit, max_depth=depth, min_samples_leaf=leaf,
                class_weight=cw)))
        return {"model": "tree", "configs": configs,
                "seeds": [cfg.RANDOM_STATE], "checkpoints": None}

    base = selected_values(sweep, out)
    checkpoints = None

    if sweep == "1":
        configs = [_config(f"trees{base['n_trees']}", base)]
        checkpoints = TREE_COUNTS
    elif sweep == "2":
        configs = [_config(f"f{m}_{boot}",
                           {**base, "max_features": m, "bootstrap": boot})
                   for boot, m in product(["standard", "none"], FEATURES)]
    elif sweep == "3a":
        configs = [_config(f"d{d or 'none'}", {**base, "max_depth": d})
                   for d in DEPTHS]
    elif sweep == "3b":
        configs = [_config(f"l{leaf}", {**base, "min_samples_leaf": leaf})
                   for leaf in LEAVES]
    elif sweep == "4":
        configs = [_config("none", base),
                   _config("class_weights",
                           {**base, "class_weight": "balanced"})]
        for size in BALANCED_SIZES:
            configs.append(_config(
                f"balanced_{size or 'min'}",
                {**base, "bootstrap": "balanced", "class_sample_size": size}))
        for kind, target in product(["smote", "smote_tomek"], SMOTE_TARGETS):
            configs.append(_config(f"{kind}_{target}", base,
                                   resample=(kind, target)))
    else:
        raise SystemExit(f"unknown sweep {sweep}; choose from {ORDER}")

    return {"model": "forest", "configs": configs, "seeds": cfg.SEEDS,
            "checkpoints": checkpoints}
