#!/usr/bin/env python3
"""Week 4: explain the Week 3 support vector classifier.

Methods: held-out permutation feature importance (global) and Kernel SHAP
(local explanations of the SVC's signed decision score). The model is refitted
from scratch with the exact data split and selected hyperparameters from Week 3;
no saved model file or network data download is required.

Install: python -m pip install numpy scipy scikit-learn matplotlib shap
Run:     python week4_xai_model.py --output-dir results

Outputs: metrics.json, permutation_importance.csv, local_shap_values.csv, and
five PNG visualizations. The analysis is educational and is not a clinical tool.
"""

import argparse
import csv
import json
import platform
from pathlib import Path

import matplotlib

# Generate images without needing an interactive display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import scipy
import shap
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SPLIT_SEED = 42
SHAP_SEED = 123
BACKGROUND_SEEDS = (42, 7)
BACKGROUND_SIZE = 40
SHAP_SAMPLES = 1024
PERMUTATION_REPEATS = 30


def choose_background(X_train, y_train, ids_train, seed):
    """Sample 40 real training rows with class stratification; never use test data."""
    X_background, _, y_background, _, ids_background, _ = train_test_split(
        X_train, y_train, ids_train, train_size=BACKGROUND_SIZE,
        stratify=y_train, random_state=seed
    )
    return X_background, y_background, ids_background


def explain_one(explainer, row, nsamples=SHAP_SAMPLES):
    """Estimate one set of Kernel SHAP values with a fixed coalition RNG seed.

    The same seed is used for each run of this function. In the background
    sensitivity check this keeps the random coalition sequence controlled while
    changing the reference rows. l1_reg=0.0 avoids automatic top-10 sparsity.
    """
    np.random.seed(SHAP_SEED)
    values = explainer.shap_values(row.reshape(1, -1), nsamples=nsamples,
                                   l1_reg=0.0, silent=True)
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != row.size:
        raise RuntimeError("Unexpected SHAP output shape for a single-output classifier.")
    return values


def local_case_record(name, test_position, y_test, y_pred, ids_test, scores, row,
                      phi, base, feature_names):
    """Collect a transparent additive check and the strongest local effects."""
    actual = "malignant" if y_test[test_position] == 1 else "benign"
    predicted = "malignant" if y_pred[test_position] == 1 else "benign"
    score = float(scores[test_position])
    reconstructed = float(base + np.sum(phi))
    residual = reconstructed - score
    # Additivity is a numerical consistency check, not proof of causality or
    # correctness of the approximate attributions.
    if not np.isclose(reconstructed, score, rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"SHAP additive reconstruction failed for {name}: {residual}")
    order = np.argsort(-np.abs(phi), kind="stable")
    strongest = [
        {"feature": str(feature_names[i]), "feature_value": float(row[i]),
         "shap_value": float(phi[i])}
        for i in order[:8]
    ]
    return {"case": name, "test_position": int(test_position),
            "dataset_row_index_zero_based": int(ids_test[test_position]),
            "actual": actual, "predicted": predicted, "decision_score": score,
            "reference_score": float(base), "sum_shap_values": float(np.sum(phi)),
            "reconstructed_score": reconstructed, "additivity_residual": float(residual),
            "top_abs_contributions": strongest}


def save_permutation_plot(out_dir, feature_names, importance, base_f1):
    """Show mean F1 drop ± standard deviation across repeated shuffles."""
    order = np.argsort(-importance.importances_mean, kind="stable")[:12]
    means, stds = importance.importances_mean[order], importance.importances_std[order]
    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    y = np.arange(len(order))
    ax.barh(y, means, xerr=stds, height=0.67, color="#247889",
            error_kw={"ecolor": "#344d54", "capsize": 3.5, "linewidth": 1.2})
    ax.axvline(0, color="#52666b", linewidth=0.9)
    ax.set_yticks(y, labels=[feature_names[i] for i in order], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Decrease in malignant F1 after shuffling (mean ± 1 shuffle SD)", fontsize=12)
    ax.set_title(f"Global permutation importance • test baseline F1 = {base_f1:.3f}",
                 fontweight="bold", fontsize=12.5, pad=12)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(out_dir / "global_permutation_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_local_plot(out_dir, filename, case, phi, feature_names, row):
    """Display the eight largest local effects and the net of remaining effects."""
    top = np.argsort(-np.abs(phi), kind="stable")[:8]
    rest = np.setdiff1d(np.arange(len(phi)), top)
    values = np.r_[phi[top], np.sum(phi[rest])]
    labels = [f"{feature_names[i]} = {row[i]:.4g}" for i in top] + ["Other 22 features (net)"]
    colors = ["#b94f3e" if v >= 0 else "#28758b" for v in values]
    fig, ax = plt.subplots(figsize=(8.8, 5.7), constrained_layout=True)
    y = np.arange(len(values))
    ax.barh(y, values, color=colors, height=0.69)
    ax.axvline(0, color="#586e72", linewidth=1.0)
    ax.set_yticks(y, labels=labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Contribution to signed decision score (not a probability)", fontsize=12)
    title = (f"Local Kernel SHAP • actual {case['actual']}, predicted {case['predicted']}\n"
             f"Reference {case['reference_score']:+.3f} + contributions {case['sum_shap_values']:+.3f} "
             f"= score {case['decision_score']:+.3f}; boundary = 0")
    ax.set_title(title, fontweight="bold", fontsize=12.2, pad=12)
    ax.legend(handles=[Patch(color="#b94f3e", label="Raises score toward malignant"),
                       Patch(color="#28758b", label="Lowers score toward benign")],
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=10.5, frameon=False)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    fig.savefig(out_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_sensitivity_plot(out_dir, feature_names, phi_a, phi_b):
    """Contrast false-negative SHAP values under two training reference samples."""
    order = np.argsort(-np.maximum(np.abs(phi_a), np.abs(phi_b)), kind="stable")[:8]
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    y = np.arange(len(order))
    ax.barh(y - 0.19, phi_a[order], height=0.34, color="#237b88", label="Reference A (seed 42)")
    ax.barh(y + 0.19, phi_b[order], height=0.34, color="#a05c91", label="Reference B (seed 7)")
    ax.axvline(0, color="#586e72", linewidth=1.0)
    ax.set_yticks(y, labels=[feature_names[i] for i in order], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP contribution to decision score", fontsize=12)
    ax.set_title("Reference sensitivity • same false-negative case", fontweight="bold", fontsize=12.5, pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=10.5, frameon=False)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    fig.savefig(out_dir / "shap_reference_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="results", help="Folder for metrics, tables and plots")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False})

    # 1. Load the same public data and encode malignant as the positive class.
    data = load_breast_cancer()
    X = data.data
    y = 1 - data.target  # sklearn's original labels are 0=malignant, 1=benign.
    feature_names = np.asarray(data.feature_names)
    ids = np.arange(len(y))  # Dataset row indices are not patient identifiers.
    if not np.isfinite(X).all():
        raise ValueError("Unexpected non-finite values in the input features.")

    # 2. Re-create the Week 3 stratified 80/20 split. The additional index array
    # follows the same partition and allows the example selection to be audited.
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.20, stratify=y, random_state=SPLIT_SEED
    )
    # One measured example of the strong feature correlation relevant to both
    # attribution techniques. This descriptive check uses training data only.
    radius_col = int(np.flatnonzero(feature_names == "worst radius")[0])
    perimeter_col = int(np.flatnonzero(feature_names == "worst perimeter")[0])
    radius_perimeter_corr = float(np.corrcoef(X_train[:, radius_col], X_train[:, perimeter_col])[0, 1])

    # 3. Refit the Week 3 winner on the original training split. The scaler is
    # fitted only on training data and remains inside the model pipeline. This
    # is a refit, not a load of a previously serialized estimator.
    model = Pipeline([
        ("scale", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=10, gamma=0.01, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores = model.decision_function(X_test)  # Signed scores, not probabilities.
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    baseline_f1 = float(f1_score(y_test, y_pred, pos_label=1))
    metrics = {"accuracy": float(accuracy_score(y_test, y_pred)),
               "precision_malignant": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
               "recall_malignant": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
               "f1_malignant": baseline_f1,
               "confusion_counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}

    # 4. Global XAI: perturb one test-set feature at a time and measure how much
    # malignant F1 drops. Use 30 shuffles; their SD is not a confidence interval.
    # The labels and the test set are used only for post-hoc explanation here.
    importance = permutation_importance(
        model, X_test, y_test, scoring="f1", n_repeats=PERMUTATION_REPEATS,
        random_state=SPLIT_SEED, n_jobs=1
    )
    order = np.argsort(-importance.importances_mean, kind="stable")
    global_rows = []
    for i in order:
        row = {"feature": str(feature_names[i]),
               "mean_f1_drop": float(importance.importances_mean[i]),
               "std_f1_drop": float(importance.importances_std[i])}
        row.update({f"shuffle_{k + 1:02d}": float(v) for k, v in enumerate(importance.importances[i])})
        global_rows.append(row)
    with (out_dir / "permutation_importance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(global_rows[0]))
        writer.writeheader()
        writer.writerows(global_rows)
    save_permutation_plot(out_dir, feature_names, importance, baseline_f1)

    # 5. Select examples deterministically: the first false negative, first true
    # positive and first true negative in the fixed test order. These are case
    # studies, not a representative sample or a global SHAP summary.
    masks = {"false_negative": (y_test == 1) & (y_pred == 0),
             "true_positive": (y_test == 1) & (y_pred == 1),
             "true_negative": (y_test == 0) & (y_pred == 0)}
    positions = {}
    for name, mask in masks.items():
        choices = np.flatnonzero(mask)
        if len(choices):
            positions[name] = int(choices[0])
    if "false_negative" not in positions or "true_positive" not in positions or "true_negative" not in positions:
        raise RuntimeError("Expected example classes are missing; check library versions and model outputs.")

    # 6. Local XAI: Kernel SHAP explains the original-feature pipeline's signed
    # decision function using 40 stratified TRAINING rows as a masking reference.
    # Unobserved coalitions mix the case and reference rows and can be unrealistic.
    X_bg_a, y_bg_a, ids_bg_a = choose_background(X_train, y_train, ids_train, BACKGROUND_SEEDS[0])
    X_bg_b, y_bg_b, ids_bg_b = choose_background(X_train, y_train, ids_train, BACKGROUND_SEEDS[1])
    explainer_a = shap.KernelExplainer(model.decision_function, X_bg_a, link="identity")
    base_a = float(explainer_a.expected_value)
    phi = {}
    case_records = {}
    for name, pos in positions.items():
        phi[name] = explain_one(explainer_a, X_test[pos])
        case_records[name] = local_case_record(
            name, pos, y_test, y_pred, ids_test, scores, X_test[pos], phi[name], base_a, feature_names
        )

    # 7. Repeat the false-negative explanation with another 40-row training
    # background. The model, input, and coalition seed remain unchanged.
    explainer_b = shap.KernelExplainer(model.decision_function, X_bg_b, link="identity")
    base_b = float(explainer_b.expected_value)
    fn_pos = positions["false_negative"]
    phi_b = explain_one(explainer_b, X_test[fn_pos])
    case_b = local_case_record("false_negative_reference_B", fn_pos, y_test, y_pred,
                               ids_test, scores, X_test[fn_pos], phi_b, base_b, feature_names)
    top_a = set(np.argsort(-np.abs(phi["false_negative"]))[:5])
    top_b = set(np.argsort(-np.abs(phi_b))[:5])
    sensitivity = {
        "reference_a_score": base_a, "reference_b_score": base_b,
        "mean_absolute_change_in_shap": float(np.mean(np.abs(phi["false_negative"] - phi_b))),
        "max_absolute_change_in_shap": float(np.max(np.abs(phi["false_negative"] - phi_b))),
        "top_five_feature_overlap": int(len(top_a.intersection(top_b))),
        "background_row_overlap": int(len(set(ids_bg_a).intersection(set(ids_bg_b)))),
        "reference_b_case": case_b,
    }

    # 8. Save all 30 signed attributions for each case and the alternate reference.
    with (out_dir / "local_shap_values.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["case", "background", "dataset_row_index_zero_based", "feature", "feature_value", "shap_value"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, pos in positions.items():
            for i, feature in enumerate(feature_names):
                writer.writerow({"case": name, "background": "A", "dataset_row_index_zero_based": int(ids_test[pos]),
                                 "feature": str(feature), "feature_value": float(X_test[pos, i]),
                                 "shap_value": float(phi[name][i])})
        for i, feature in enumerate(feature_names):
            writer.writerow({"case": "false_negative", "background": "B",
                             "dataset_row_index_zero_based": int(ids_test[fn_pos]), "feature": str(feature),
                             "feature_value": float(X_test[fn_pos, i]), "shap_value": float(phi_b[i])})

    save_local_plot(out_dir, "shap_false_negative.png", case_records["false_negative"],
                    phi["false_negative"], feature_names, X_test[fn_pos])
    tp_pos = positions["true_positive"]
    save_local_plot(out_dir, "shap_true_positive.png", case_records["true_positive"],
                    phi["true_positive"], feature_names, X_test[tp_pos])
    save_sensitivity_plot(out_dir, feature_names, phi["false_negative"], phi_b)
    fig, ax = plt.subplots(figsize=(5.9, 5.0), constrained_layout=True)
    ConfusionMatrixDisplay(cm, display_labels=["Benign (0)", "Malignant (1)"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title("Held-out predictions: confusion matrix", fontweight="bold")
    fig.savefig(out_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 9. Record enough metadata to reproduce and audit the interpretation.
    results = {
        "dataset": {"name": "Wisconsin Diagnostic Breast Cancer (scikit-learn built-in copy)",
                    "samples": int(len(y)), "features": int(X.shape[1]),
                    "train_size": int(len(y_train)), "test_size": int(len(y_test)),
                    "train_pearson_r_worst_radius_vs_worst_perimeter": radius_perimeter_corr,
                    "encoding": {"0": "benign", "1": "malignant (positive)"}},
        "model": {"description": "Week 3 selected SVC, refitted on the same training split",
                  "scaler": "StandardScaler within Pipeline", "kernel": "rbf", "C": 10,
                  "gamma": 0.01, "class_weight": "balanced", "split_seed": SPLIT_SEED},
        "test_metrics": metrics,
        "permutation": {"dataset": "held-out test (post-hoc)", "scoring": "malignant F1",
                        "repeats": PERMUTATION_REPEATS, "random_seed": SPLIT_SEED,
                        "top_features": [{"feature": row["feature"], "mean_f1_drop": row["mean_f1_drop"],
                                          "std_f1_drop": row["std_f1_drop"]} for row in global_rows[:12]]},
        "shap": {"method": "Kernel SHAP; identity link; original-feature SVC decision_function",
                 "background_source": "stratified sample from training data only",
                 "background_size": BACKGROUND_SIZE,
                 "reference_A_counts": {"benign": int(np.sum(y_bg_a == 0)), "malignant": int(np.sum(y_bg_a == 1))},
                 "reference_B_counts": {"benign": int(np.sum(y_bg_b == 0)), "malignant": int(np.sum(y_bg_b == 1))},
                 "background_seeds": list(BACKGROUND_SEEDS), "coalition_seed": SHAP_SEED,
                 "nsamples": SHAP_SAMPLES, "l1_reg": 0.0,
                 "selection_rule": "first false negative, true positive and true negative in fixed test order",
                 "cases": case_records, "sensitivity": sensitivity},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit_learn": sklearn.__version__, "shap": shap.__version__, "matplotlib": matplotlib.__version__},
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Refitted model test F1: {baseline_f1:.4f}; confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("Top global permutation features:")
    for row in global_rows[:5]:
        print(f"  {row['feature']}: mean F1 drop {row['mean_f1_drop']:.4f} ± {row['std_f1_drop']:.4f}")
    for name, case in case_records.items():
        print(f"{name}: dataset row {case['dataset_row_index_zero_based']}, "
              f"base={case['reference_score']:.4f}, score={case['decision_score']:.4f}, "
              f"sum SHAP={case['sum_shap_values']:.4f}, residual={case['additivity_residual']:.2e}")
    print(f"False-negative top-5 overlap across backgrounds: {sensitivity['top_five_feature_overlap']}/5")
    print(f"Outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
