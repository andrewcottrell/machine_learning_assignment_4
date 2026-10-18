"""Classification tree on integer-coded features."""
import numpy as np
from numba import njit

GINI, ENTROPY = 0, 1
CRITERIA = {"gini": GINI, "entropy": ENTROPY}

# a split must lower the weighted impurity by more than this share of it
MIN_GAIN = 1e-10


@njit(cache=True, nogil=True)
def _rand_below(state, n):
    """Uniform integer in [0, n) from a splitmix64 generator held in state."""
    state[0] += np.uint64(0x9E3779B97F4A7C15)
    z = state[0]
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return np.int64(z % np.uint64(n))


@njit(cache=True, nogil=True)
def _impurity(c, cw, criterion):
    """Total weight times impurity of the weighted class counts c * cw."""
    W = 0.0
    for k in range(c.shape[0]):
        W += c[k] * cw[k]
    if W <= 0.0:
        return 0.0
    s = 0.0
    if criterion == GINI:
        for k in range(c.shape[0]):
            v = c[k] * cw[k]
            s += v * v
        return W - s / W
    for k in range(c.shape[0]):
        v = c[k] * cw[k]
        if v > 0.0:
            s -= v * np.log2(v / W)
    return s


@njit(cache=True, nogil=True)
def build_tree(Xc, y, counts, cw, is_nominal, n_codes, n_classes, max_depth,
               min_samples_leaf, max_features, criterion, seed):
    """Grows one tree depth first; max_depth < 0 means no limit."""
    p, n = Xc.shape
    K = n_classes

    m = 0
    for i in range(n):
        if counts[i] > 0:
            m += 1
    idx = np.empty(m, np.int32)
    j = 0
    for i in range(n):
        if counts[i] > 0:
            idx[j] = i
            j += 1

    cap = max(2 * m - 1, 1)
    feature = np.full(cap, -1, np.int16)
    threshold = np.zeros(cap, np.int32)
    left = np.full(cap, -1, np.int32)
    right = np.full(cap, -1, np.int32)
    pred = np.zeros(cap, np.int16)
    importance = np.zeros(p)

    hist = np.zeros((n_codes, K))
    bin_n = np.zeros(n_codes)
    touched = np.empty(n_codes, np.int64)
    tot = np.zeros(K)
    lc = np.zeros(K)
    rc = np.zeros(K)
    feats = np.arange(p)
    state = np.empty(1, np.uint64)
    state[0] = np.uint64(seed)

    s_start = np.empty(m + 2, np.int32)
    s_end = np.empty(m + 2, np.int32)
    s_depth = np.empty(m + 2, np.int32)
    s_node = np.empty(m + 2, np.int32)
    s_start[0], s_end[0], s_depth[0], s_node[0] = 0, m, 0, 0
    sp = 1
    n_nodes = 1
    depth_reached = 0

    while sp > 0:
        sp -= 1
        start, end = s_start[sp], s_end[sp]
        depth, node = s_depth[sp], s_node[sp]
        if depth > depth_reached:
            depth_reached = depth

        tot[:] = 0.0
        for q in range(start, end):
            i = idx[q]
            tot[y[i]] += counts[i]

        n_tot = 0.0
        present = 0
        best_k = 0
        best_w = -1.0
        for k in range(K):
            n_tot += tot[k]
            if tot[k] > 0.0:
                present += 1
            if tot[k] * cw[k] > best_w:
                best_w = tot[k] * cw[k]
                best_k = k
        pred[node] = best_k

        if (present <= 1 or n_tot < 2 * min_samples_leaf
                or (max_depth >= 0 and depth >= max_depth)):
            continue

        node_imp = _impurity(tot, cw, criterion)
        best_cost = node_imp * (1.0 - MIN_GAIN)
        best_f = -1
        best_t = 0

        # features are visited in random order; the search continues past
        # max_features only while no valid split has been found
        for v in range(p):
            r = v + _rand_below(state, p - v)
            feats[v], feats[r] = feats[r], feats[v]
            f = feats[v]
            row = Xc[f]

            nt = 0
            for q in range(start, end):
                i = idx[q]
                b = row[i]
                if bin_n[b] == 0.0:
                    touched[nt] = b
                    nt += 1
                bin_n[b] += counts[i]
                hist[b, y[i]] += counts[i]

            if nt > 1:
                touched[:nt].sort()
                if is_nominal[f]:
                    for a in range(nt):
                        b = touched[a]
                        n_left = bin_n[b]
                        if (n_left < min_samples_leaf
                                or n_tot - n_left < min_samples_leaf):
                            continue
                        for k in range(K):
                            lc[k] = hist[b, k]
                            rc[k] = tot[k] - lc[k]
                        cost = (_impurity(lc, cw, criterion)
                                + _impurity(rc, cw, criterion))
                        if cost < best_cost:
                            best_cost, best_f, best_t = cost, f, b
                else:
                    lc[:] = 0.0
                    n_left = 0.0
                    for a in range(nt - 1):
                        b = touched[a]
                        n_left += bin_n[b]
                        for k in range(K):
                            lc[k] += hist[b, k]
                        if n_left < min_samples_leaf:
                            continue
                        if n_tot - n_left < min_samples_leaf:
                            break
                        for k in range(K):
                            rc[k] = tot[k] - lc[k]
                        cost = (_impurity(lc, cw, criterion)
                                + _impurity(rc, cw, criterion))
                        if cost < best_cost:
                            best_cost, best_f = cost, f
                            # halfway to the next occupied code
                            best_t = (b + touched[a + 1]) // 2

            for a in range(nt):
                b = touched[a]
                bin_n[b] = 0.0
                for k in range(K):
                    hist[b, k] = 0.0

            if v + 1 >= max_features and best_f >= 0:
                break

        if best_f < 0:
            continue

        importance[best_f] += node_imp - best_cost

        row = Xc[best_f]
        nominal = is_nominal[best_f]
        lo, hi = start, end - 1
        while lo <= hi:
            b = row[idx[lo]]
            if (nominal and b == best_t) or (not nominal and b <= best_t):
                lo += 1
            else:
                idx[lo], idx[hi] = idx[hi], idx[lo]
                hi -= 1

        feature[node] = best_f
        threshold[node] = best_t
        left[node] = n_nodes
        right[node] = n_nodes + 1

        s_start[sp], s_end[sp] = lo, end
        s_depth[sp], s_node[sp] = depth + 1, n_nodes + 1
        sp += 1
        s_start[sp], s_end[sp] = start, lo
        s_depth[sp], s_node[sp] = depth + 1, n_nodes
        sp += 1
        n_nodes += 2

    return (feature[:n_nodes].copy(), threshold[:n_nodes].copy(),
            left[:n_nodes].copy(), right[:n_nodes].copy(),
            pred[:n_nodes].copy(), importance, depth_reached)


@njit(cache=True, nogil=True)
def predict_tree(feature, threshold, left, right, pred, is_nominal, Xc, out):
    """Writes the predicted class of every record into out."""
    for i in range(Xc.shape[1]):
        node = 0
        while feature[node] >= 0:
            f = feature[node]
            b = Xc[f, i]
            if is_nominal[f]:
                go_left = b == threshold[node]
            else:
                go_left = b <= threshold[node]
            node = left[node] if go_left else right[node]
        out[i] = pred[node]


def balanced_class_weight(y, counts, n_classes):
    """Inverse class frequency weights, averaging one over the sample."""
    per_class = np.bincount(y, weights=counts, minlength=n_classes)
    cw = np.zeros(n_classes)
    seen = per_class > 0
    cw[seen] = per_class.sum() / (seen.sum() * per_class[seen])
    return cw


class DecisionTree:
    """Classification tree. Features are codes from data.BinEncoder."""

    def __init__(self, criterion="entropy", max_depth=None,
                 min_samples_leaf=1, max_features=None, class_weight=None,
                 seed=0):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.seed = seed

    def fit(self, Xc, y, is_nominal, n_classes, counts=None):
        """counts gives the multiplicity of each record; None means one each."""
        if counts is None:
            counts = np.ones(Xc.shape[1], dtype=np.int32)
        counts = np.ascontiguousarray(counts, dtype=np.int32)

        if self.class_weight is None:
            cw = np.ones(n_classes)
        elif isinstance(self.class_weight, str):
            assert self.class_weight == "balanced"
            cw = balanced_class_weight(y, counts, n_classes)
        else:
            cw = np.asarray(self.class_weight, dtype=np.float64)

        p = Xc.shape[0]
        self.is_nominal_ = np.ascontiguousarray(is_nominal, dtype=np.bool_)
        (self.feature_, self.threshold_, self.left_, self.right_, self.pred_,
         importance, self.depth_) = build_tree(
            Xc, np.ascontiguousarray(y, dtype=np.int64), counts, cw,
            self.is_nominal_, int(Xc.max()) + 1, n_classes,
            -1 if self.max_depth is None else self.max_depth,
            self.min_samples_leaf,
            p if self.max_features is None else min(self.max_features, p),
            CRITERIA[self.criterion], self.seed)

        total = importance.sum()
        self.importances_ = importance / total if total > 0 else importance
        self.n_nodes_ = len(self.feature_)
        self.n_leaves_ = int((self.feature_ < 0).sum())
        return self

    def predict(self, Xc):
        out = np.empty(Xc.shape[1], dtype=np.int16)
        predict_tree(self.feature_, self.threshold_, self.left_, self.right_,
                     self.pred_, self.is_nominal_, Xc, out)
        return out
