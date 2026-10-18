"""SMOTE and Tomek link removal for mixed numeric and nominal features."""
import json
import pickle

import numpy as np
import pandas as pd
from numba import njit, prange

import config as cfg

K_NEIGHBOURS = 5
CACHE = cfg.OUT / "cache"
SUMMARY_DIR = cfg.OUT / "resample"


@njit(cache=True)
def _sq_dist(Z, C, a, b, limit):
    d = 0.0
    for j in range(C.shape[1]):
        if C[a, j] != C[b, j]:
            d += 1.0
    if d > limit:
        return d
    for j in range(Z.shape[1]):
        diff = Z[a, j] - Z[b, j]
        d += diff * diff
        if d > limit:
            return d
    return d


@njit(parallel=True, cache=True)
def _knn(Z, C, k):
    n = Z.shape[0]
    out = np.empty((n, k), np.int64)
    for a in prange(n):
        best_d = np.full(k, np.inf)
        best_i = np.full(k, -1, np.int64)
        for b in range(n):
            if b == a:
                continue
            d = _sq_dist(Z, C, a, b, best_d[k - 1])
            if d < best_d[k - 1]:
                pos = k - 1
                while pos > 0 and best_d[pos - 1] > d:
                    best_d[pos] = best_d[pos - 1]
                    best_i[pos] = best_i[pos - 1]
                    pos -= 1
                best_d[pos] = d
                best_i[pos] = b
        out[a] = best_i
    return out


@njit(parallel=True, cache=True)
def _neighbour_modes(C, nbrs):
    """Most common level among the neighbours of every record."""
    n, k = nbrs.shape
    out = np.empty_like(C)
    for a in prange(n):
        for j in range(C.shape[1]):
            best_level, best_count, own_count = C[a, j], 0, 0
            for s in range(k):
                level = C[nbrs[a, s], j]
                count = 0
                for t in range(k):
                    if C[nbrs[a, t], j] == level:
                        count += 1
                if level == C[a, j]:
                    own_count = count
                if count > best_count:
                    best_count, best_level = count, level
            out[a, j] = C[a, j] if own_count == best_count else best_level
    return out


@njit(parallel=True, cache=True)
def _nearest_distance(Z, C, query):
    """Squared distance from each query record to its nearest other record."""
    out = np.empty(query.shape[0])
    for q in prange(query.shape[0]):
        a = query[q]
        best = np.inf
        for b in range(Z.shape[0]):
            if b != a:
                d = _sq_dist(Z, C, a, b, best)
                if d < best:
                    best = d
        out[q] = best
    return out


@njit(parallel=True, cache=True)
def _singles_in_links(Z, C, cls, nnd, single):
    """Flags the distinct records that are part of a Tomek link.

    a and b form a link when they differ in class and neither has another
    record closer than the other: d(a, b) = nnd(a) = nnd(b).
    """
    out = np.zeros(single.shape[0], np.bool_)
    for q in prange(single.shape[0]):
        a = single[q]
        for r in range(single.shape[0]):
            b = single[r]
            if b == a or cls[b] == cls[a] or nnd[b] != nnd[a]:
                continue
            if _sq_dist(Z, C, a, b, nnd[a]) == nnd[a]:
                out[q] = True
                break
    return out


class Space:
    """The feature space in which distances are measured."""

    def __init__(self, num_cols, nom_cols):
        self.num_cols = list(num_cols)
        self.cat_cols = list(nom_cols) + list(cfg.FLAG_COLS)

    def fit(self, X):
        logged = np.log1p(X[self.num_cols].to_numpy(dtype=np.float64))
        self.mean_ = logged.mean(axis=0)
        std = logged.std(axis=0)
        self.std_ = np.where(std > 0, std, 1.0)
        self.levels_ = {c: np.unique(X[c].to_numpy()) for c in self.cat_cols}
        return self

    def numeric(self, X):
        raw = X[self.num_cols].to_numpy(dtype=np.float64)
        return np.ascontiguousarray((np.log1p(raw) - self.mean_) / self.std_)

    def codes(self, X):
        cols = [np.searchsorted(self.levels_[c], X[c].to_numpy())
                for c in self.cat_cols]
        return np.ascontiguousarray(np.column_stack(cols).astype(np.int32))

    def levels(self, codes):
        """Turns code columns back into the original levels."""
        return {c: self.levels_[c][codes[:, j]]
                for j, c in enumerate(self.cat_cols)}


def smote(X, y, target, num_cols, nom_cols, rng, k=K_NEIGHBOURS):
    """Raises every class smaller than target to target records."""
    space = Space(num_cols, nom_cols).fit(X)
    Z, C = space.numeric(X), space.codes(X)
    raw = X[space.num_cols].to_numpy(dtype=np.float64)

    new_num, new_cat, new_y = [], [], []
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        n_new = target - len(rows)
        if n_new <= 0 or len(rows) < 2:
            continue
        kc = min(k, len(rows) - 1)
        Zc, Cc = np.ascontiguousarray(Z[rows]), np.ascontiguousarray(C[rows])
        nbrs = _knn(Zc, Cc, kc)
        modes = _neighbour_modes(Cc, nbrs)

        base = rng.integers(0, len(rows), size=n_new)
        other = nbrs[base, rng.integers(0, kc, size=n_new)]
        gap = rng.random(n_new)[:, None]
        raw_c = raw[rows]
        new_num.append(raw_c[base] + gap * (raw_c[other] - raw_c[base]))
        new_cat.append(modes[base])
        new_y.append(np.full(n_new, c, dtype=y.dtype))

    if not new_y:
        return X.reset_index(drop=True), y.copy()

    synthetic = pd.DataFrame(np.vstack(new_num), columns=space.num_cols)
    for col, values in space.levels(np.vstack(new_cat)).items():
        synthetic[col] = values
    X_out = pd.concat([X, synthetic[X.columns]], ignore_index=True)
    return X_out, np.concatenate([y] + new_y)


def tomek_links(X, y, num_cols, nom_cols, scale_on=None):
    """Boolean mask of the records that belong to a Tomek link."""
    space = Space(num_cols, nom_cols).fit(X if scale_on is None else scale_on)
    Z, C = space.numeric(X), space.codes(X)

    # work on distinct feature vectors; duplicates are handled by counting
    both = np.hstack([Z, C])
    _, first, inverse = np.unique(both, axis=0, return_index=True,
                                  return_inverse=True)
    inverse = inverse.ravel()
    Zu, Cu = np.ascontiguousarray(Z[first]), np.ascontiguousarray(C[first])
    n_unique = len(first)

    multiplicity = np.bincount(inverse, minlength=n_unique)
    pairs = np.unique(np.column_stack([inverse, y]), axis=0)
    n_classes_in = np.bincount(pairs[:, 0], minlength=n_unique)

    unique_in_link = n_classes_in > 1
    single = np.flatnonzero(multiplicity == 1)
    nnd = np.zeros(n_unique)
    nnd[single] = _nearest_distance(Zu, Cu, single)
    cls = np.asarray(y)[first].astype(np.int64)
    unique_in_link[single] = _singles_in_links(Zu, Cu, cls, nnd, single)

    return unique_in_link[inverse]


def _summary(y_before, y_smote, y_after):
    rows = []
    for c in range(cfg.N_CLASSES):
        rows.append({"class": c,
                     "n_before": int((y_before == c).sum()),
                     "n_synthetic": int((y_smote == c).sum()
                                        - (y_before == c).sum()),
                     "n_removed": int((y_smote == c).sum()
                                      - (y_after == c).sum()),
                     "n_after": int((y_after == c).sum())})
    return rows


def load_or_make(fold, kind, target, X_train, y_train, num_cols, nom_cols):
    """Resampled training records of one fold, computed once and cached."""
    assert kind in ("smote", "smote_tomek")
    CACHE.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    def cached(name, make):
        path = CACHE / f"{name}_{target}_fold{fold}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        result = make()
        with open(path, "wb") as f:
            pickle.dump(result, f)
        return result

    def make_smote():
        rng = np.random.default_rng([cfg.RANDOM_STATE, fold, target])
        return smote(X_train, y_train, target, num_cols, nom_cols, rng)

    X_s, y_s = cached("smote", make_smote)
    X_out, y_out = X_s, y_s

    if kind == "smote_tomek":
        def make_tomek():
            link = tomek_links(X_s, y_s, num_cols, nom_cols, scale_on=X_train)
            return X_s[~link].reset_index(drop=True), y_s[~link]

        X_out, y_out = cached("smote_tomek", make_tomek)

    summary = SUMMARY_DIR / f"{kind}_{target}_fold{fold}.json"
    if not summary.exists():
        summary.write_text(json.dumps(_summary(y_train, y_s, y_out), indent=2))

    return X_out, y_out
