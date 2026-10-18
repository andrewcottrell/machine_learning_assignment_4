"""Random forest built from the trees in tree.py."""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

from tree import CRITERIA, balanced_class_weight, build_tree, predict_tree

BOOTSTRAPS = ("standard", "balanced", "none")


@njit(cache=True, nogil=True)
def _add_votes(votes, pred):
    for i in range(pred.shape[0]):
        votes[i, pred[i]] += 1


def tree_seeds(seed, index):
    state = np.random.SeedSequence([seed, index]).generate_state(2, np.uint64)
    return int(state[0]), int(state[1] >> np.uint64(1))


def bootstrap_counts(y, kind, n_classes, rng, class_sample_size=None):
    """Number of times each record enters the training sample of one tree."""
    n = len(y)
    if kind == "none":
        return np.ones(n, dtype=np.int32)
    if kind == "standard":
        return np.bincount(rng.integers(0, n, size=n),
                           minlength=n).astype(np.int32)

    counts = np.zeros(n, dtype=np.int32)
    members = [np.flatnonzero(y == k) for k in range(n_classes)]
    size = class_sample_size or min(len(m) for m in members if len(m))
    for m in members:
        if len(m):
            np.add.at(counts, rng.choice(m, size=size, replace=True), 1)
    return counts


class RandomForest:
    """Bagged trees with a random subset of features tried at every split."""

    def __init__(self, n_trees=100, max_features="sqrt", max_depth=None,
                 min_samples_leaf=1, criterion="entropy", bootstrap="standard",
                 class_weight=None, class_sample_size=None, seed=0, n_jobs=-1):
        assert bootstrap in BOOTSTRAPS
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.class_sample_size = class_sample_size
        self.seed = seed
        self.n_jobs = n_jobs

    def _workers(self):
        return os.cpu_count() if self.n_jobs in (-1, None) else self.n_jobs

    def fit(self, Xc, y, is_nominal, n_classes):
        p = Xc.shape[0]
        y = np.ascontiguousarray(y, dtype=np.int64)
        self.n_classes_ = n_classes
        self.is_nominal_ = np.ascontiguousarray(is_nominal, dtype=np.bool_)

        if self.max_features is None:
            m_try = p
        elif self.max_features == "sqrt":
            m_try = max(1, int(np.sqrt(p)))
        else:
            m_try = min(int(self.max_features), p)
        self.max_features_ = m_try

        if self.class_weight is None:
            cw = np.ones(n_classes)
        else:
            assert self.class_weight == "balanced"
            cw = balanced_class_weight(y, np.ones(len(y)), n_classes)

        n_codes = int(Xc.max()) + 1
        max_depth = -1 if self.max_depth is None else self.max_depth
        criterion = CRITERIA[self.criterion]

        def grow(index):
            boot_seed, split_seed = tree_seeds(self.seed, index)
            counts = bootstrap_counts(y, self.bootstrap, n_classes,
                                      np.random.default_rng(boot_seed),
                                      self.class_sample_size)
            return build_tree(Xc, y, counts, cw, self.is_nominal_, n_codes,
                              n_classes, max_depth, self.min_samples_leaf,
                              m_try, criterion, split_seed)

        with ThreadPoolExecutor(self._workers()) as pool:
            grown = list(pool.map(grow, range(self.n_trees)))

        self.trees_ = [t[:5] for t in grown]
        raw = np.array([t[5] for t in grown])
        totals = raw.sum(axis=1, keepdims=True)
        self.importances_ = (raw / np.where(totals > 0, totals, 1)).mean(axis=0)
        self.depths_ = np.array([t[6] for t in grown])
        self.n_nodes_ = np.array([len(t[0]) for t in grown])
        return self

    def _tree_predictions(self, Xc):
        """Yields the predictions of each tree, in tree order."""
        def one(tree):
            out = np.empty(Xc.shape[1], dtype=np.int16)
            predict_tree(*tree, self.is_nominal_, Xc, out)
            return out

        workers = self._workers()
        with ThreadPoolExecutor(workers) as pool:
            # in batches, so finished predictions do not pile up in memory
            for start in range(0, len(self.trees_), 2 * workers):
                yield from pool.map(one, self.trees_[start:start + 2 * workers])

    def staged_predict(self, Xc, checkpoints):
        """Majority vote of the first t trees, for every t in checkpoints."""
        checkpoints = sorted(checkpoints)
        assert checkpoints[-1] <= len(self.trees_)
        votes = np.zeros((Xc.shape[1], self.n_classes_), dtype=np.int32)
        staged = np.empty((len(checkpoints), Xc.shape[1]), dtype=np.int16)
        row = 0
        for t, pred in enumerate(self._tree_predictions(Xc), start=1):
            _add_votes(votes, pred)
            while row < len(checkpoints) and checkpoints[row] == t:
                staged[row] = votes.argmax(axis=1)
                row += 1
            if row == len(checkpoints):
                break
        return staged

    def predict(self, Xc):
        return self.staged_predict(Xc, [len(self.trees_)])[0]
