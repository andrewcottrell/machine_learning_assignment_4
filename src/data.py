"""Data loading, fold assignment and the integer encoding used by the tree."""
import numpy as np
import pandas as pd

import config as cfg


def load_clean(path=cfg.RAW_PATH, verbose=False):
    """Loads the dataset and cleans it as in assignment 2."""
    df = pd.read_csv(path, na_values="?")

    df["service"] = df["service"].fillna("none")

    df["is_ftp_login"] = (df["is_ftp_login"] > 0).astype(int)

    df = df.dropna(subset=[cfg.TARGET_COL]).drop(columns=[cfg.ID_COL])

    y = df[cfg.TARGET_COL].to_numpy(dtype=np.int64)
    X = df.drop(columns=[cfg.TARGET_COL])

    nom_cols = list(cfg.NOMINAL_COLS)
    num_cols = [c for c in X.columns
                if c not in nom_cols and c not in cfg.FLAG_COLS]

    if verbose:
        print(f"loaded {len(X):,} rows: {len(num_cols)} numeric, "
              f"{len(nom_cols)} nominal, {len(cfg.FLAG_COLS)} flags")

    return X, y, num_cols, nom_cols


def class_names():
    """Class names ordered by class code."""
    cat_map = pd.read_csv(cfg.CAT_MAP_PATH)
    by_code = dict(zip(cat_map["Mapping"], cat_map["Attack Category Name"]))
    return [by_code[c] for c in sorted(by_code)]


def make_folds(y):
    """Fold index of every record, identical to the assignment 2 split."""
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True,
                         random_state=cfg.RANDOM_STATE)
    folds = np.empty(len(y), dtype=np.int8)
    for k, (_, valid) in enumerate(cv.split(np.zeros(len(y)), y)):
        folds[valid] = k
    return folds


def load_folds(y):
    """Reads the saved folds."""
    if cfg.FOLDS_PATH.exists():
        return np.load(cfg.FOLDS_PATH)
    folds = make_folds(y)
    cfg.OUT.mkdir(exist_ok=True)
    np.save(cfg.FOLDS_PATH, folds)
    return folds


class BinEncoder:
    """Maps every feature to small integer codes, fitted on training records."""

    def __init__(self, num_cols, nom_cols, max_bins=cfg.MAX_BINS):
        self.ordered_cols = list(num_cols) + list(cfg.FLAG_COLS)
        self.nom_cols = list(nom_cols)
        self.max_bins = max_bins

    def fit(self, X):
        self.cuts_ = {}
        for c in self.ordered_cols:
            values = np.unique(X[c].to_numpy(dtype=np.float64))
            if self.max_bins is None or len(values) <= self.max_bins:
                cuts = (values[:-1] + values[1:]) / 2
            else:
                q = np.linspace(0, 1, self.max_bins + 1)[1:-1]
                cuts = np.unique(np.quantile(X[c].to_numpy(dtype=np.float64), q))
                cuts = cuts[cuts < values[-1]]
            self.cuts_[c] = cuts

        self.levels_ = {c: {v: i for i, v in enumerate(sorted(X[c].unique()))}
                        for c in self.nom_cols}

        self.feature_names = self.ordered_cols + self.nom_cols
        self.is_nominal = np.array([False] * len(self.ordered_cols)
                                   + [True] * len(self.nom_cols))
        self.n_bins = np.array(
            [len(self.cuts_[c]) + 1 for c in self.ordered_cols]
            + [len(self.levels_[c]) + 1 for c in self.nom_cols],
            dtype=np.int64)
        return self

    def transform(self, X):
        """Returns the codes as a (features, records) array."""
        out = np.empty((len(self.feature_names), len(X)), dtype=np.int32)
        for j, c in enumerate(self.ordered_cols):
            out[j] = np.searchsorted(self.cuts_[c],
                                     X[c].to_numpy(dtype=np.float64),
                                     side="left")
        for j, c in enumerate(self.nom_cols, start=len(self.ordered_cols)):
            levels = self.levels_[c]
            out[j] = X[c].map(levels).fillna(len(levels)).to_numpy(dtype=np.int32)
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)
