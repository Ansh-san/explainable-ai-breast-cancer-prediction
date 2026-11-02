# Week 4: Explainable AI and Model Interpretability

An executable global and local interpretability study of the Week 3 breast-tumour classifier. It uses **permutation feature importance** and **Kernel SHAP**, including a practical reference-sensitivity check. This is an educational demonstration, **not a clinical diagnostic tool**.

## Project files

- `week4_xai_model.py` — complete, commented, standalone Python script.
- `week4_detailed_report.docx` — report for the Word-only upload form (more than 2,000 words).
- `week4_detailed_report.pdf` — the same report in the assignment's requested PDF format.
- `submission_description.txt` — ready-to-paste submission description of more than 200 words.
- `results/` — metrics, explanation tables and five plotted results from the reference run.
- `requirements.txt` — the pinned reference environment's principal and supporting packages.

## Run the code

The reference run used Python **3.13.16**. From this directory:

```bash
python -m pip install -r requirements.txt
python week4_xai_model.py --output-dir results
```

A less strictly pinned installation is `python -m pip install numpy scipy scikit-learn matplotlib shap`. The script loads the dataset bundled with scikit-learn, so no input file or data download is required after dependencies are installed. A different Python or package version can change numerical results or the available examples; the code reports a clear error if the expected fixed-split example types are absent.

The script reconstructs the Week 3 model; it does not load a previously saved estimator. It reproduces the same stratified 80/20 split and refits a `StandardScaler` + RBF `SVC(C=10, gamma=0.01, class_weight='balanced')` pipeline on the training rows only. Malignant is encoded as positive class 1. It uses the test set for **post-hoc inspection**; this set must not be reused as a blind test for later changes.

## Analyses and outputs

- **Global**: 30 features × 30 permutations of the held-out set, scored by malignant F1. The CSV includes each shuffle, its mean F1 decrease and shuffle standard deviation. These standard deviations are not confidence intervals.
- **Local**: Kernel SHAP explains the SVC's signed `decision_function`, not a probability. Reference A contains 40 stratified training cases; up to 1,024 coalition masks and seed 123 are used for each case. The script saves all feature attributions for a false negative, true positive and true negative.
- **Sensitivity**: The false-negative case is explained again using a second 40-case training reference. The model, case and coalition seed remain fixed; the result illustrates how the reference affects attributions.
- Files generated: `metrics.json`, `permutation_importance.csv`, `local_shap_values.csv`, `global_permutation_importance.png`, `shap_false_negative.png`, `shap_true_positive.png`, `shap_reference_sensitivity.png` and `confusion_matrix.png`.

The test F1 is **0.9880**. The largest mean test F1 drop is for **worst texture: 0.0731**. In the false-negative example, reference A is **−0.2960**, the contributions sum to **−0.8106**, and the model score is **−1.1066**. Four of the top five absolute contributions overlap between the two reference samples. These are results for the specified model, data and assumptions, not causal medical findings.

## Dataset and method attribution

Wolberg, W., Mangasarian, O., Street, N. and Street, W. (1995). *Breast Cancer Wisconsin (Diagnostic)* [dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5DW2B>. Dataset licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Lundberg, S. M. and Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS. <https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>. See also the [SHAP KernelExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html).
