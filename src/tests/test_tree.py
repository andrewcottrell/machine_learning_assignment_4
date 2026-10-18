"""Checks of the tree against known answers and against scikit-learn.

Run from src/ with: python -m pytest tests/test_tree.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
import data
from tree import DecisionTree, balanced_class_weight


def leaf_sizes(tree, Xc, counts=None):
    """Number of training records reaching each leaf."""
    leaves = np.empty(Xc.shape[1], dtype=np.int64)
    for i in range(Xc.shape[1]):
        node = 0
        while tree.feature_[node] >= 0:
            f = tree.feature_[node]
            b = Xc[f, i]
            left = (b == tree.threshold_[node] if tree.is_nominal_[f]
                    else b <= tree.threshold_[node])
            node = tree.left_[node] if left else tree.right_[node]
        leaves[i] = node
    return np.bincount(leaves, weights=counts)[np.unique(leaves)]


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    Xc = rng.integers(0, 20, size=(6, 3000)).astype(np.int32)
    y = ((Xc[0] > 9).astype(int) + (Xc[1] > 4)) % 3
    y = np.where(rng.random(3000) < 0.1, rng.integers(0, 3, 3000), y)
    return Xc, y.astype(np.int64), np.zeros(6, dtype=bool)


def test_separable_data_is_fitted_exactly():
    rng = np.random.default_rng(1)
    Xc = rng.integers(0, 50, size=(4, 2000)).astype(np.int32)
    y = ((Xc[0] > 20) ^ (Xc[2] <= 7)).astype(np.int64)
    tree = DecisionTree().fit(Xc, y, np.zeros(4, dtype=bool), 2)
    assert (tree.predict(Xc) == y).all()


def test_nominal_split_isolates_one_level():
    rng = np.random.default_rng(2)
    Xc = rng.integers(0, 9, size=(2, 1000)).astype(np.int32)
    y = (Xc[1] == 4).astype(np.int64)
    tree = DecisionTree().fit(Xc, y, np.array([False, True]), 2)
    assert tree.n_nodes_ == 3
    assert tree.feature_[0] == 1 and tree.threshold_[0] == 4
    # an unseen level falls on the "rest" side
    assert tree.predict(np.array([[0], [99]], dtype=np.int32))[0] == 0


def test_depth_and_leaf_limits_hold(toy):
    Xc, y, nominal = toy
    tree = DecisionTree(max_depth=3).fit(Xc, y, nominal, 3)
    assert tree.depth_ <= 3
    tree = DecisionTree(min_samples_leaf=25).fit(Xc, y, nominal, 3)
    assert leaf_sizes(tree, Xc).min() >= 25


def test_counts_equal_repeated_records(toy):
    Xc, y, nominal = toy
    rng = np.random.default_rng(3)
    counts = rng.integers(0, 4, size=Xc.shape[1]).astype(np.int32)
    rep = np.repeat(np.arange(Xc.shape[1]), counts)
    a = DecisionTree(seed=5).fit(Xc, y, nominal, 3, counts=counts)
    b = DecisionTree(seed=5).fit(np.ascontiguousarray(Xc[:, rep]), y[rep],
                                 nominal, 3)
    assert (a.predict(Xc) == b.predict(Xc)).all()


def test_same_seed_same_tree(toy):
    Xc, y, nominal = toy
    a = DecisionTree(max_features=2, seed=9).fit(Xc, y, nominal, 3)
    b = DecisionTree(max_features=2, seed=9).fit(Xc, y, nominal, 3)
    c = DecisionTree(max_features=2, seed=10).fit(Xc, y, nominal, 3)
    assert np.array_equal(a.feature_, b.feature_)
    assert not np.array_equal(a.feature_, c.feature_)


def test_balanced_weights_equalise_classes():
    y = np.array([0] * 90 + [1] * 10)
    cw = balanced_class_weight(y, np.ones(100), 3)
    assert cw[0] * 90 == pytest.approx(cw[1] * 10)
    assert cw[2] == 0.0


def test_matches_scikit_learn_on_subsample():
    """Macro F1 within noise of scikit-learn's tree on a 40,000 record sample."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.tree import DecisionTreeClassifier

    if not cfg.RAW_PATH.exists():
        pytest.skip("dataset not available")

    X, y, num_cols, nom_cols = data.load_clean()
    keep, _ = train_test_split(np.arange(len(y)), train_size=40_000,
                               stratify=y, random_state=cfg.RANDOM_STATE)
    trn, val = train_test_split(keep, test_size=0.25, stratify=y[keep],
                                random_state=cfg.RANDOM_STATE)

    enc = data.BinEncoder(num_cols, nom_cols).fit(X.iloc[trn])
    Xt, Xv = enc.transform(X.iloc[trn]), enc.transform(X.iloc[val])

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    oe.fit(X.iloc[trn][nom_cols])

    def raw(rows):
        return np.hstack([X.iloc[rows][enc.ordered_cols].to_numpy(float),
                          oe.transform(X.iloc[rows][nom_cols])])

    ours, theirs = [], []
    for seed in range(5):
        t = DecisionTree(criterion="entropy", seed=seed)
        t.fit(Xt, y[trn], enc.is_nominal, cfg.N_CLASSES)
        ours.append(f1_score(y[val], t.predict(Xv), average="macro"))

        sk = DecisionTreeClassifier(criterion="entropy", random_state=seed)
        sk.fit(raw(trn), y[trn])
        theirs.append(f1_score(y[val], sk.predict(raw(val)), average="macro"))

    spread = max(np.std(ours), np.std(theirs))
    assert abs(np.mean(ours) - np.mean(theirs)) < max(2 * spread, 0.01)
