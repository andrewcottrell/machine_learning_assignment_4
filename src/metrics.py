"""Performance measures computed from the confusion matrix."""
import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    """Counts with true classes in rows and predicted classes in columns."""
    flat = np.asarray(y_true, dtype=np.int64) * n_classes + np.asarray(y_pred)
    return np.bincount(flat, minlength=n_classes ** 2).reshape(n_classes,
                                                               n_classes)


def per_class(cm):
    """Precision, recall and F1 of every class."""
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp),
                          where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def scores(y_true, y_pred, n_classes):
    """All measures of one set of predictions."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    precision, recall, f1 = per_class(cm)
    support = cm.sum(axis=1)

    out = {
        "macro_f1": f1.mean(),
        "weighted_f1": (f1 * support).sum() / support.sum(),
        "accuracy": np.trace(cm) / cm.sum(),
        "balanced_acc": recall[support > 0].mean(),
    }
    for k in range(n_classes):
        out[f"precision_{k}"] = precision[k]
        out[f"recall_{k}"] = recall[k]
        out[f"f1_{k}"] = f1[k]
    return out
