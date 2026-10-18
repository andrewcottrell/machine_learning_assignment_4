"""Checks of SMOTE and of Tomek link detection.

Run from src/ with: python -m pytest tests/test_resample.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg
import resample

NUM = ["a", "b", "c"]
NOM = ["proto"]


def make_frame(n, seed, coarse=False):
    """Small mixed dataset; coarse values give many ties and duplicates."""
    rng = np.random.default_rng(seed)
    if coarse:
        num = rng.integers(0, 3, size=(n, 3)).astype(float)
    else:
        num = rng.gamma(2.0, 10.0, size=(n, 3))
    X = pd.DataFrame(num, columns=NUM)
    X["proto"] = rng.choice(["tcp", "udp", "arp"], size=n, p=[.6, .3, .1])
    for flag in cfg.FLAG_COLS:
        X[flag] = rng.integers(0, 2, size=n)
    y = rng.choice(4, size=n, p=[.6, .25, .1, .05])
    return X, y


def brute_force_links(X, y):
    """Tomek links straight from the definition, in quadratic time."""
    space = resample.Space(NUM, NOM).fit(X)
    Z, C = space.numeric(X), space.codes(X)
    n = len(y)
    d = np.empty((n, n))
    for a in range(n):
        for b in range(n):
            d[a, b] = resample._sq_dist(Z, C, a, b, np.inf)
    np.fill_diagonal(d, np.inf)
    nnd = d.min(axis=1)
    in_link = np.zeros(n, dtype=bool)
    for a in range(n):
        for b in range(n):
            if y[a] != y[b] and d[a, b] == nnd[a] and d[a, b] == nnd[b]:
                in_link[a] = True
    return in_link


@pytest.mark.parametrize("coarse", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tomek_links_match_the_definition(seed, coarse):
    X, y = make_frame(300, seed, coarse)
    found = resample.tomek_links(X, y, NUM, NOM)
    assert np.array_equal(found, brute_force_links(X, y))
    if coarse:
        assert found.any() and not found.all()


def test_conflicting_duplicates_are_links():
    X, y = make_frame(200, 3)
    X = pd.concat([X, X.iloc[:10]], ignore_index=True)
    y = np.concatenate([y, (y[:10] + 1) % 4])
    found = resample.tomek_links(X, y, NUM, NOM)
    assert found[:10].all() and found[200:].all()


def test_smote_reaches_target_with_valid_records():
    X, y = make_frame(2000, 4)
    rng = np.random.default_rng(0)
    X_new, y_new = resample.smote(X, y, 400, NUM, NOM, rng)

    before = np.bincount(y, minlength=4)
    after = np.bincount(y_new, minlength=4)
    assert (after == np.maximum(before, 400)).all()

    # the original records come first and are unchanged
    assert X_new.iloc[:2000].reset_index(drop=True).equals(X)
    assert np.array_equal(y_new[:2000], y)

    synthetic, y_syn = X_new.iloc[2000:], y_new[2000:]
    assert set(synthetic["proto"]) <= set(X["proto"])
    for flag in cfg.FLAG_COLS:
        assert set(synthetic[flag]) <= {0, 1}
    # a point between two records of a class stays inside the class range
    for c in np.unique(y_syn):
        real = X.loc[y == c, NUM]
        made = synthetic.loc[y_syn == c, NUM]
        assert (made.min() >= real.min() - 1e-9).all()
        assert (made.max() <= real.max() + 1e-9).all()


def test_smote_is_reproducible():
    X, y = make_frame(1000, 5)
    a, _ = resample.smote(X, y, 300, NUM, NOM, np.random.default_rng(1))
    b, _ = resample.smote(X, y, 300, NUM, NOM, np.random.default_rng(1))
    assert a.equals(b)
