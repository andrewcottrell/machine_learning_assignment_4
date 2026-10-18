"""Checks of the forest.

Run from src/ with: python -m pytest tests/test_forest.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
import data
from forest import RandomForest, bootstrap_counts, tree_seeds
from tree import DecisionTree


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    Xc = rng.integers(0, 20, size=(9, 4000)).astype(np.int32)
    y = ((Xc[0] > 9).astype(int) + (Xc[1] > 4) + (Xc[2] == 3)) % 3
    y = np.where(rng.random(4000) < 0.15, rng.integers(0, 3, 4000), y)
    nominal = np.zeros(9, dtype=bool)
    nominal[2] = True
    return Xc, y.astype(np.int64), nominal


def test_one_unbagged_tree_is_the_tree(toy):
    Xc, y, nominal = toy
    forest = RandomForest(n_trees=1, max_features=None, bootstrap="none",
                          seed=4).fit(Xc, y, nominal, 3)
    tree = DecisionTree(seed=tree_seeds(4, 0)[1]).fit(Xc, y, nominal, 3)
    assert (forest.predict(Xc) == tree.predict(Xc)).all()


def test_result_does_not_depend_on_thread_count(toy):
    Xc, y, nominal = toy
    a = RandomForest(n_trees=24, seed=7, n_jobs=1).fit(Xc, y, nominal, 3)
    b = RandomForest(n_trees=24, seed=7, n_jobs=4).fit(Xc, y, nominal, 3)
    c = RandomForest(n_trees=24, seed=8, n_jobs=4).fit(Xc, y, nominal, 3)
    assert (a.predict(Xc) == b.predict(Xc)).all()
    assert np.allclose(a.importances_, b.importances_)
    assert not (a.predict(Xc) == c.predict(Xc)).all()


def test_staged_predictions_are_prefixes_of_the_forest(toy):
    Xc, y, nominal = toy
    big = RandomForest(n_trees=30, seed=1).fit(Xc, y, nominal, 3)
    small = RandomForest(n_trees=10, seed=1).fit(Xc, y, nominal, 3)
    staged = big.staged_predict(Xc, [1, 10, 30])
    assert (staged[1] == small.predict(Xc)).all()
    assert (staged[2] == big.predict(Xc)).all()


def test_bootstrap_counts(toy):
    _, y, _ = toy
    rng = np.random.default_rng(0)
    assert bootstrap_counts(y, "none", 3, rng).tolist() == [1] * len(y)
    assert bootstrap_counts(y, "standard", 3, rng).sum() == len(y)

    counts = bootstrap_counts(y, "balanced", 3, rng)
    per_class = np.bincount(y, weights=counts)
    assert (per_class == np.bincount(y).min()).all()
    counts = bootstrap_counts(y, "balanced", 3, rng, class_sample_size=500)
    assert (np.bincount(y, weights=counts) == 500).all()


def test_forest_scores_above_single_tree_on_subsample():
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    if not cfg.RAW_PATH.exists():
        pytest.skip("dataset not available")

    X, y, num_cols, nom_cols = data.load_clean()
    keep, _ = train_test_split(np.arange(len(y)), train_size=40_000,
                               stratify=y, random_state=cfg.RANDOM_STATE)
    trn, val = train_test_split(keep, test_size=0.25, stratify=y[keep],
                                random_state=cfg.RANDOM_STATE)
    enc = data.BinEncoder(num_cols, nom_cols).fit(X.iloc[trn])
    Xt, Xv = enc.transform(X.iloc[trn]), enc.transform(X.iloc[val])

    tree = DecisionTree(seed=0).fit(Xt, y[trn], enc.is_nominal, cfg.N_CLASSES)
    forest = RandomForest(n_trees=100, seed=0)
    forest.fit(Xt, y[trn], enc.is_nominal, cfg.N_CLASSES)

    f1_tree = f1_score(y[val], tree.predict(Xv), average="macro")
    f1_forest = f1_score(y[val], forest.predict(Xv), average="macro")
    assert f1_forest > f1_tree
