"""Checks of the measures against scikit-learn.

Run from src/ with: python -m pytest tests/test_metrics.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import metrics


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_measures_equal_scikit_learn(seed):
    from sklearn import metrics as sk

    rng = np.random.default_rng(seed)
    # skewed classes; class 9 is never predicted, class 8 never occurs
    y_true = rng.choice(8, size=5000, p=[.5, .2, .1, .08, .06, .03, .02, .01])
    y_true[:20] = 9
    y_pred = np.where(rng.random(5000) < 0.7, y_true, rng.integers(0, 9, 5000))
    y_pred[y_pred == 9] = 0

    labels = np.arange(10)
    present = np.unique(np.concatenate([y_true, y_pred]))
    got = metrics.scores(y_true, y_pred, 10)

    f1 = sk.f1_score(y_true, y_pred, labels=labels, average=None,
                     zero_division=0)
    assert got["macro_f1"] * 10 == pytest.approx(f1.sum())
    assert got["macro_f1"] * 10 / len(present) == pytest.approx(
        sk.f1_score(y_true, y_pred, average="macro"))
    assert got["weighted_f1"] == pytest.approx(
        sk.f1_score(y_true, y_pred, average="weighted"))
    assert got["accuracy"] == pytest.approx(sk.accuracy_score(y_true, y_pred))
    assert got["balanced_acc"] == pytest.approx(
        sk.balanced_accuracy_score(y_true, y_pred))
    for k in labels:
        assert got[f"f1_{k}"] == pytest.approx(f1[k])
    assert np.array_equal(metrics.confusion_matrix(y_true, y_pred, 10),
                          sk.confusion_matrix(y_true, y_pred, labels=labels))
