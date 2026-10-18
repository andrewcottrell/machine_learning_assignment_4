"""Draws every report figure from the saved results.

    python figures.py
"""
import argparse
import json

import numpy as np
import pandas as pd

import config as cfg
import data
import metrics
import plotstyle as ps
from run_experiment import scores_path

MEASURE_LABEL = {"macro_f1": "Validation macro $F_1$",
                 "accuracy": "Validation accuracy"}
RARE = [7, 8, 2, 5]


def mean_se(df, key, measure, split="valid"):
    """Mean and standard error over folds, after averaging over seeds."""
    rows = df[df["split"] == split]
    per_fold = rows.groupby([key, "fold"])[measure].mean().unstack("fold")
    return per_fold.mean(axis=1), per_fold.std(axis=1, ddof=1) / np.sqrt(
        per_fold.shape[1])


def as_number(values, none=np.inf):
    """Parameter values read from a score file, with "none" as infinity."""
    return np.array([none if str(v) == "none" else float(v) for v in values])


def ordered(series):
    """Positions 0, 1, ... of a parameter's values in increasing order."""
    values = sorted(set(series), key=lambda v: as_number([v])[0])
    return values, {v: i for i, v in enumerate(values)}


def tick_label(v):
    return "none" if str(v) == "none" else str(int(float(v)))


def load(sweep, out):
    path = scores_path(sweep, out)
    return pd.read_csv(path) if path.exists() else None



def fig_trees(curves, path, selected=None):
    fig, ax = ps.figure(2.1 if len(curves) == 1 else 2.3)
    for i, (df, label, colour, marker) in enumerate(curves):
        valid = df[df["split"] == "valid"]
        by_seed = valid.groupby(["n_trees", "seed"])["macro_f1"].mean().unstack()
        mean, spread = by_seed.mean(axis=1), by_seed.std(axis=1, ddof=1)
        ax.fill_between(mean.index, mean - spread, mean + spread, color=colour,
                        alpha=0.2, linewidth=0,
                        label=("$\\pm$ one standard deviation over seeds"
                               if i == 0 else None))
        ax.plot(mean.index, mean, color=colour, marker=marker, label=label)
    if selected is not None:
        ax.axvline(selected, color=ps.MUTED, linewidth=0.8, linestyle=":")
        ax.annotate("selected", (selected, 1.0), xytext=(0, 2),
                    xycoords=("data", "axes fraction"),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7, color=ps.MUTED)
    ax.set_xscale("log")
    ax.set_xticks(mean.index)
    ax.set_xticklabels(["" if n in (300, 400) else str(n) for n in mean.index])
    ax.minorticks_off()
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Validation macro $F_1$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2,
              columnspacing=1.0)
    ps.save(fig, path)


def fig_features(df, path, measures=tuple(MEASURE_LABEL)):
    values, pos = ordered(df["max_features"])
    looks = {"standard": dict(color=ps.BLUE, marker="o", linestyle="-",
                              label="Bootstrap"),
             "none": dict(color=ps.ORANGE, marker="s", linestyle="--",
                          label="No bootstrap")}

    fig, axes = ps.figure(1.1 + 1.1 * len(measures), nrows=len(measures),
                          sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, measure in zip(axes, measures):
        for boot, look in looks.items():
            sub = df[df["bootstrap"] == boot]
            mean, se = mean_se(sub, "max_features", measure)
            x = [pos[v] for v in mean.index]
            order = np.argsort(x)
            ax.errorbar(np.array(x)[order], mean.to_numpy()[order],
                        yerr=se.to_numpy()[order], **look)
        ax.set_ylabel(MEASURE_LABEL[measure])
    axes[0].legend(loc="lower right")
    axes[-1].set_xticks(range(len(values)))
    axes[-1].set_xticklabels([tick_label(v) for v in values])
    axes[-1].set_xlabel("Features tried per split")
    fig.align_ylabels(axes)
    ps.save(fig, path)


def _tree_curve(tree_df, selection, param):
    params = selection["by_macro_f1"]["params"]
    keep = np.ones(len(tree_df), dtype=bool)
    for col in ["criterion", "max_depth", "min_samples_leaf", "class_weight"]:
        if col == param:
            continue
        want = "none" if params[col] is None else str(params[col])
        keep &= (tree_df[col].astype(str) == want).to_numpy()
    return tree_df[keep]


def _complexity_axis(ax, forest_df, tree_df, tree_selection, param, xlabel):
    curves = {"forest": forest_df}
    if tree_df is not None and tree_selection is not None:
        curves["tree"] = _tree_curve(tree_df, tree_selection, param)
    values, pos = ordered(pd.concat([c[param].astype(str)
                                     for c in curves.values()]))
    for model, df in curves.items():
        df = df.assign(**{param: df[param].astype(str)})
        for split, style, alpha in [("valid", "-", 1.0), ("train", "--", 0.55)]:
            mean, se = mean_se(df, param, "macro_f1", split)
            x = np.array([pos[v] for v in mean.index])
            order = np.argsort(x)
            look = dict(ps.STYLE[model], linestyle=style, alpha=alpha)
            name = "validation" if split == "valid" else "training"
            ax.errorbar(x[order], mean.to_numpy()[order],
                        yerr=se.to_numpy()[order] if split == "valid" else None,
                        label=f"{ps.LABEL[model]}, {name}", **look)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([tick_label(v) for v in values])
    ax.set_xlabel(xlabel)


def fig_complexity(forest_df, tree_df, tree_selection, param, xlabel, path):
    fig, ax = ps.figure(2.4)
    _complexity_axis(ax, forest_df, tree_df, tree_selection, param, xlabel)
    ax.set_ylabel("Macro $F_1$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              columnspacing=1.0, handlelength=2.2)
    ps.save(fig, path)


def fig_complexity_pair(panels, tree_df, tree_selection, path):
    fig, axes = ps.figure(2.3, ncols=2, sharey=True)
    for ax, (forest_df, param, xlabel) in zip(axes, panels):
        _complexity_axis(ax, forest_df, tree_df, tree_selection, param, xlabel)
        if len(ax.get_xticks()) > 6:
            ax.tick_params(axis="x", labelrotation=90)
    axes[0].set_ylabel("Macro $F_1$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, columnspacing=1.0, handlelength=2.2)
    fig.tight_layout(pad=0.2)
    ps.save(fig, path)


TREATMENT = {"none": "None", "class_weights": "Class weights"}


def treatment_label(name):
    if name in TREATMENT:
        return TREATMENT[name]
    kind, _, size = name.rpartition("_")
    size = "smallest class" if size == "min" else f"{int(size):,}"
    return {"balanced": "Balanced bootstrap", "smote": "SMOTE",
            "smote_tomek": "SMOTE + Tomek"}[kind] + f", {size}"


def fig_imbalance(df, path, table_path):
    """Macro F1 and accuracy of every imbalance treatment."""
    valid = df[df["split"] == "valid"]
    order = list(dict.fromkeys(df["config"]))[::-1]

    fig, axes = ps.figure(2.9, ncols=2, sharey=True)
    for ax, measure in zip(axes, MEASURE_LABEL):
        mean, se = mean_se(df, "config", measure)
        y = np.arange(len(order))
        ax.axvline(mean["none"], color=ps.MUTED, linewidth=0.8, linestyle=":")
        ax.errorbar(mean[order], y, xerr=se[order], color=ps.BLUE, marker="o",
                    linestyle="none")
        ax.set_xlabel(MEASURE_LABEL[measure])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels([treatment_label(n) for n in order])
    ps.save(fig, path)

    table = pd.DataFrame({m: mean_se(df, "config", m)[0]
                          for m in ["macro_f1", "accuracy", "balanced_acc"]})
    table["macro_f1_se"] = mean_se(df, "config", "macro_f1")[1]
    for part in ["recall", "precision", "f1"]:
        table[f"rare_{part}"] = valid.groupby("config")[
            [f"{part}_{c}" for c in RARE]].mean().mean(axis=1)
    table["n_train"] = valid.groupby("config")["n_train"].mean()
    table = table.loc[order[::-1]]
    table.insert(0, "treatment", [treatment_label(n) for n in table.index])
    table.to_csv(table_path)
    print("wrote", table_path)

    tex = table_path.parent.parent / "tables" / "treatments.tex"
    tex.parent.mkdir(exist_ok=True)
    tex.write_text("".join(
        f"{r['treatment'].replace('smallest class', '139')} & "
        f"{r['macro_f1']:.4f} & {r['rare_recall']:.3f} & "
        f"{r['rare_precision']:.3f} & {r['rare_f1']:.3f} \\\\\n"
        for _, r in table.iterrows()))
    print("wrote", tex)


def fig_confusion(y, pred_rows, names, path):
    """Row-normalised confusion matrix, averaged over the given predictions."""
    cm = sum(metrics.confusion_matrix(y, p, cfg.N_CLASSES) for p in pred_rows)
    cm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = ps.figure(3.3)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j] >= 0.005:
                ax.text(j, i, f"{cm[i, j]:.2f}".lstrip("0") if cm[i, j] < 1
                        else "1.0", ha="center", va="center", fontsize=5.5,
                        color="white" if cm[i, j] > 0.5 else ps.INK)
    ps.save(fig, path)


def fig_per_class(per_class, path):
    """F1 of every class under the three models, ordered by conflict rate."""
    rows = per_class.sort_values("conflict_pct", ascending=False)
    y = np.arange(len(rows))

    fig, ax = ps.figure(2.8)
    lo = rows[["knn_f1", "tree_f1", "forest_f1"]].min(axis=1)
    hi = rows[["knn_f1", "tree_f1", "forest_f1"]].max(axis=1)
    ax.hlines(y, lo, hi, color=ps.GRID, linewidth=1.5, zorder=1)
    for model in ["knn", "tree", "forest"]:
        look = dict(ps.STYLE[model], linestyle="none")
        ax.plot(rows[f"{model}_f1"], y, label=ps.LABEL[model], zorder=2,
                markeredgecolor="white", markeredgewidth=0.4, markersize=4.5,
                **look)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} ({p:.0f}%)" for c, p in
                        zip(rows["class"], rows["conflict_pct"])])
    ax.set_xlabel("$F_1$ score")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              columnspacing=1.0, handletextpad=0.3)
    ps.save(fig, path)


def fig_importance(imp, path, top=15):
    """Importance of the leading features under the tree and the forest."""
    rows = imp.sort_values("forest", ascending=False).head(top)[::-1]
    y = np.arange(len(rows))
    h = 0.36

    fig, ax = ps.figure(3.0)
    ax.barh(y + h / 2 + 0.02, rows["forest"], height=h, color=ps.BLUE,
            label=ps.LABEL["forest"])
    ax.barh(y - h / 2 - 0.02, rows["tree"], height=h, color=ps.ORANGE,
            label=ps.LABEL["tree"])
    ax.set_yticks(y)
    ax.set_yticklabels(rows["feature"], family="monospace", fontsize=6.5)
    ax.set_xlabel("Share of total impurity decrease")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ps.save(fig, path)



def run(out=cfg.OUT, figs=cfg.FIGS):
    ps.apply()

    def selection(sweep):
        path = out / f"sweep_{sweep}" / "selection.json"
        return json.loads(path.read_text()) if path.exists() else None

    s1 = load("1", out)
    if s1 is not None:
        sel = selection("1")
        fig_trees([(s1, "Mean", ps.BLUE, "o")], figs / "trees.pdf",
                  sel["selected"]["n_trees"] if sel else None)

    s2 = load("2", out)
    if s2 is not None:
        fig_features(s2, figs / "features.pdf", measures=("macro_f1",))

    s0 = load("0", out)
    panels = []
    for sweep, param, xlabel, name in [
            ("3a", "max_depth", "Maximum depth", "depth.pdf"),
            ("3b", "min_samples_leaf", "Minimum records per leaf", "leaf.pdf")]:
        df = load(sweep, out)
        if df is not None:
            fig_complexity(df, s0, selection("0"), param, xlabel, figs / name)
            panels.append((df, param, xlabel))
    if len(panels) == 2:
        fig_complexity_pair(panels, s0, selection("0"), figs / "complexity.pdf")

    s4 = load("4", out)
    if s4 is not None:
        fig_imbalance(s4, figs / "imbalance.pdf",
                      out / "sweep_4" / "treatments.csv")

    final = out / "final"
    if (final / "forest_y_pred.npy").exists():
        _, y, _, _ = data.load_clean()
        names = data.class_names()
        fig_confusion(y, np.load(final / "forest_y_pred.npy"), names,
                      figs / "confusion_forest.pdf")
        fig_confusion(y, [np.load(final / "tree_y_pred.npy")], names,
                      figs / "confusion_tree.pdf")
        fig_importance(pd.read_csv(final / "importances.csv"),
                       figs / "importance.pdf")
        models = json.loads((final / "models.json").read_text())
        scores = pd.read_csv(final / "scores.csv")
        curves = [(scores[scores["config"] == "forest"],
                   "14 features, no bootstrap", ps.BLUE, "o")]
        if s1 is not None:
            curves.insert(0, (s1, "6 features, bootstrap", ps.ORANGE, "s"))
        fig_trees(curves, figs / "trees_final.pdf",
                  models["forest"]["n_trees"])
    if (final / "per_class.csv").exists():
        fig_per_class(pd.read_csv(final / "per_class.csv"),
                      figs / "per_class.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=None)
    parser.add_argument("--figs", default=None)
    args = parser.parse_args()
    run(cfg.OUT if args.out is None else cfg.SRC / args.out,
        cfg.FIGS if args.figs is None else cfg.SRC / args.figs)
