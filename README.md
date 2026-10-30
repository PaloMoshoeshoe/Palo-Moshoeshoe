# Interpretable vs. black-box models on the Default of Credit Card Clients data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PaloMoshoeshoe/Palo-Moshoeshoe/blob/main/Chapter3_Methodology_DoCC.ipynb)

This repository contains the analysis code for the methodology (Chapter 3) of a thesis comparing **intrinsically
interpretable** machine-learning models with **black-box** models on a structured binary classification task: predicting
credit-card default with the UCI *Default of Credit Card Clients* dataset (Yeh, 2009).

A single notebook, [`Chapter3_Methodology_DoCC.ipynb`](Chapter3_Methodology_DoCC.ipynb), runs the whole pipeline:
data download and checks, fold-wise preprocessing, seven models, repeated cross-validation, surrogate fidelity and
corrected confidence intervals. It writes ready-to-use tables (CSV) and figures (PNG and PDF).

## Research questions

**Main question.** To what extent can intrinsically interpretable models match the predictive performance of
black-box models and reproduce their predictions, including their errors, on a structured binary classification dataset?

| | Question | Answered by |
|---|---|---|
| **SQ1** | How large is the difference in discrimination between each interpretable model and each black-box model, and how precisely can that difference be estimated? | **Stage 1**: all 7 models are trained on the true target and compared on held-out folds |
| **SQ2** | How closely do the interpretable models reproduce each black-box model's decisions, errors included, when trained on its predicted labels? | **Stage 2**: each interpretable model is retrained on each black box's predictions (12 surrogates) and its agreement with that black box is measured on held-out folds |

## Repository contents

| File | Purpose |
|---|---|
| `Chapter3_Methodology_DoCC.ipynb` | The full analysis (Google Colab or Jupyter) |
| `requirements.txt` | Pinned package versions for running the notebook locally with Python 3.12 |
| `README.md` | This file |

## Quick start

### Google Colab (recommended)

1. Click the **Open in Colab** badge above.
2. Choose a Python 3.12 runtime: *Runtime → Change runtime type → Runtime version →* **2026.07**.
3. *Runtime → Run all*. Allow access to Google Drive when asked; results are saved there.
4. The first run uses `RUN_MODE = "smoke"`, a quick check of the whole pipeline. When it finishes, set
   `RUN_MODE = "full"` in the configuration cell and run all cells again to produce the thesis results.

### Local Jupyter

```bash
git clone https://github.com/PaloMoshoeshoe/Palo-Moshoeshoe.git
cd Palo-Moshoeshoe
conda create -n gosdt312 python=3.12      # or any other Python 3.12 virtual environment
conda activate gosdt312
pip install -r requirements.txt jupyterlab
jupyter lab Chapter3_Methodology_DoCC.ipynb
```

If you open the notebook from a Jupyter installation in a different environment, or from VS Code, register this
environment as a kernel and select it there:
`python -m ipykernel install --user --name gosdt312 --display-name "Python 3.12 (GOSDT)"`.

### Why Python 3.12?

GOSDT 1.0.4 ships ready-made installers (wheels) for Python 3.9–3.12 only. Colab's newest runtime uses Python 3.13,
so pick the **2026.07** runtime as above. If you stay on Python 3.13 in Colab, the setup cell compiles GOSDT from
source instead, which takes about 2–3 extra minutes. In testing, this gave identical results. Locally, with Python
3.13 or later, create a Python 3.12 environment as shown.

## Run modes and run time

| | `RUN_MODE = "smoke"` (default) | `RUN_MODE = "full"` |
|---|---|---|
| Purpose | Check that everything installs and runs | **Thesis results** |
| Data | 3,000-row stratified subsample | All 30,000 records |
| Cross-validation | 5 folds × 2 repeats | 10 folds × 5 repeats (50 held-out folds) |
| EBM | 2 outer bags (lighter, for speed) | InterpretML defaults |
| Time on Colab's free CPU runtime | about 20 minutes | roughly 10–20 hours (estimate), over several sessions |
| Results folder | `results_smoke/` | `results_full/` |

The full run takes about 6–7 minutes per fold on a fast 2-CPU machine (≈5.5 hours in total). Colab's free CPUs are
about three times slower, mostly because of the Explainable Boosting Machines.

**Interrupted runs resume automatically.** Each fold is saved as soon as it finishes; in Colab it is saved to Google
Drive (`SAVE_TO_GOOGLE_DRIVE = True`, the default). If Colab disconnects, reopen the notebook and run all cells again:
finished folds are skipped and the run continues where it stopped. On a machine
with more CPU cores, `N_PARALLEL_FOLDS` (e.g. 4 on an 8-core machine) runs several folds at once.

## What the notebook implements

### Dataset (§3.4)

* *Default of Credit Card Clients* (Yeh, 2009; UCI ML Repository id 350, [doi:10.24432/C55S3H](https://doi.org/10.24432/C55S3H)):
  30,000 clients in Taiwan (April–September 2005), 23 explanatory variables, no missing values, 6,636 defaults (22.1%).
* Downloaded automatically with `ucimlrepo`, with the original `.xls` from UCI as a fallback. The file is cached in
  `docc_interpretability/data/`, so later runs use exactly the same data. Offline, set `LOCAL_DATA_FILE` to a local
  copy (`.xls`, `.xlsx` or `.csv`).
* The notebook checks the record count, the variables, missing values and the class balance, and lists undocumented
  category codes.

### Preprocessing (§3.5): fitted on the training portion of each fold only

| Step | Implementation |
|---|---|
| Missing values | None present; no imputation |
| Category consolidation | EDUCATION codes 0, 5, 6 (345 records) → 4 "others"; MARRIAGE code 0 (54 records) → 3 "others" |
| Scaling | z-score (`StandardScaler`) for the 14 continuous and 6 ordinal repayment-status variables |
| Encoding | One-hot encoding of SEX, EDUCATION and MARRIAGE (29 model inputs in total) |
| Class imbalance | Balanced class weights for every model: each case of class *c* gets weight *n* / (2·*n*<sub>c</sub>) |

### Models (§3.6)

| Model | Family | Role | Implementation | Balanced weights via |
|---|---|---|---|---|
| L1-regularised logistic regression | Additive | Interpretable (conventional) | scikit-learn `LogisticRegression`, L1 penalty, C = 1.0 | `class_weight` |
| Explainable Boosting Machine (EBM) | Additive | Interpretable (flexible) | InterpretML `ExplainableBoostingClassifier` | `sample_weight` |
| Depth-limited CART | Tree-based | Interpretable (conventional) | `DecisionTreeClassifier`, max depth 4 | `class_weight` |
| GOSDT | Tree-based | Interpretable (flexible) | `gosdt` 1.0.4 with threshold guessing, max depth 4, regularisation 0.001 | `balance=True` |
| Random forest | Tree ensemble | Black box | `RandomForestClassifier`, library defaults | `class_weight` |
| Gradient boosting | Tree ensemble | Black box | `GradientBoostingClassifier`, library defaults | `sample_weight` |
| Multilayer perceptron | Neural network | Black box | `MLPClassifier`, library defaults | `sample_weight` |

The black boxes are **base-level**: library defaults, no hyperparameter tuning. GOSDT needs binary inputs, so a
*threshold-guessing* step (McTavish et al., 2022) learns cut-points of the form `x ≤ t` from each training fold. GOSDT
then finds the optimal sparse tree over those cut-points. Its own probabilities are 0/1 only, so ROC-AUC and PR-AUC
use the share of defaults among training cases in each leaf, the same way CART's are calculated.

### Evaluation protocol (§3.7)

* **10 × 5 repeated stratified cross-validation**, using seeds 42–46 for the five repeats. Every model sees exactly the same splits.
* **Primary metric:** ROC-AUC. **Secondary metrics:** PR-AUC (calculated as average precision) and sensitivity at each
  model's own decision rule. Accuracy and specificity are stored for reference only.

### Fidelity (§3.8)

In each fold, every black box *B* is trained on the true labels. Its predictions on the training portion become the
training labels for each interpretable surrogate *S*. On the held-out portion the notebook measures:

| Measure | Definition |
|---|---|
| Overall agreement | Share of cases where *S* and *B* predict the same label |
| Error agreement | Among the cases *B* gets wrong, the share where *S* gives the same (wrong) label |
| Correct-case agreement | The same share among the cases *B* gets right (supplementary) |
| Trivial baseline | Agreement of a "surrogate" that always predicts *B*'s most common training label |

### Statistical analysis (§3.9)

The 50 fold results share training data, so they are not independent. Every mean and difference is therefore reported
with a **corrected resampled t** interval (Nadeau & Bengio, 2003, in the repeated cross-validation form of Bouckaert &
Frank, 2004), implemented by hand:

```math
\hat\sigma^2_{\mathrm{corr}} = \left(\frac{1}{J} + \frac{n_{\mathrm{test}}}{n_{\mathrm{train}}}\right) s^2,
\qquad \mathrm{CI}_{95\%} = \bar d \pm t_{J-1,\,0.975}\,\hat\sigma_{\mathrm{corr}}
```

Here *J* = 50 and *n*<sub>test</sub>/*n*<sub>train</sub> = 1/9, which makes the intervals about 2.6 times wider than naive ones.
SQ1 differences are paired by fold (interpretable − black box). Holm-adjusted *p*-values across the 12 comparisons
are included as a supplementary check.

## Configuration

Settings are in the notebook's **Configuration** cell (Section 1), except `RUN_APPENDIX`, which is in the appendix
cell. The settings that affect the results are saved with them in `run_config.json`.

| Setting | Default | Meaning |
|---|---|---|
| `RUN_MODE` | `"smoke"` | `"smoke"` for a quick check or `"full"` for the thesis results |
| `BASE_SEED` | `42` | Seed for every model; the CV repeats use seeds 42–46 |
| `MAX_TREE_DEPTH` | `4` | Depth limit shared by CART and GOSDT |
| `LR_PARAMS` | `C=1.0`, liblinear | L1 logistic regression |
| `GOSDT_PARAMS` | regularisation 0.001, 300 s time limit, 40 threshold stumps | GOSDT search settings |
| `SURROGATE_BALANCED_WEIGHTS` | `True` | Surrogates also use balanced weights, computed on the black-box labels |
| `SURROGATE_TRAINING_LABELS` | `"in-sample"` | `"out-of-fold"` labels training cases with cross-fitted black-box predictions instead (sensitivity check) |
| `N_PARALLEL_FOLDS` | `1` | Number of folds to run at once |
| `SAVE_TO_GOOGLE_DRIVE` | `True` | Colab only: keep data and results in Google Drive |
| `LOCAL_DATA_FILE` | `""` | Optional path to a local copy of the dataset |
| `RUN_APPENDIX` | `True` | Fit each interpretable model once on all records and print it (illustration only) |

The notebook stores a fingerprint of the configuration and refuses to mix results from different settings in one
folder. To run a variant, such as the `"out-of-fold"` sensitivity check, change `PROJECT_NAME` first so its results go
to a separate folder.

## Outputs

Results are written to `docc_interpretability/`. In Colab this folder is in *My Drive*; locally it is created next to the notebook.

```text
docc_interpretability/
├── data/default_of_credit_card_clients.csv   cached copy of the dataset
└── results_full/                              (results_smoke/ for smoke runs)
    ├── run_config.json                        all settings, configuration fingerprint, data source
    ├── environment.json                       Python and package versions
    ├── folds/                                 one checkpoint file per fold (r1_f01.json … r5_f10.json)
    ├── tables/
    │   ├── per_fold_performance.csv           every model × fold: ROC-AUC, PR-AUC, sensitivity, …
    │   ├── per_fold_fidelity.csv              every surrogate × black box × fold: agreement measures
    │   ├── table_performance(_formatted).csv  mean and 95% CI per model and metric
    │   ├── table_sq1_differences.csv          SQ1: interpretable − black box, CI, t, p, Holm p
    │   └── table_sq2_fidelity(_formatted).csv SQ2: agreement, error agreement, vs. baseline
    └── figures/                               PNG (300 dpi) and PDF
        ├── fig1_performance                   per-model performance across folds
        ├── fig2_sq1_roc_auc_differences       SQ1 forest plot
        └── fig3_sq2_fidelity                  SQ2 agreement vs. trivial baseline
```

If you run the notebook locally inside this repository, add `docc_interpretability/` to `.gitignore` unless you want
to commit the data copy and checkpoint files.

## Reproducibility

* **Pinned research packages:** scikit-learn 1.7.2, interpret-core 0.7.8 (InterpretML), gosdt 1.0.4, ucimlrepo 0.0.7
  and xlrd 2.0.2. The notebook installs these itself. `requirements.txt` also pins numpy 2.0.2, pandas 2.2.2,
  scipy 1.16.3 and matplotlib 3.10.0, matching Colab's 2026.07 runtime.
* **Fixed seeds** for every model and every split.
* **Recorded run details:** each run saves its package versions, a SHA-256 hash of the data and a configuration fingerprint.
* **Tested:** the pipeline has been run in Google Colab (runtime 2026.07, Python 3.12.13) on the real dataset. In a
  separate test, Python 3.12 and Python 3.13 (with GOSDT compiled from source) gave identical results.

## Data licence and intended use

The dataset was created by I-Cheng Yeh and is available from the UCI Machine Learning Repository under a
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence. It describes credit-card clients in Taiwan in 2005.
This project uses it **only as a methodological benchmark** for comparing model performance and surrogate fidelity.
The results are not evidence about current credit-risk practice and must not be used to support lending decisions.

## References

* Bouckaert, R. R., & Frank, E. (2004). Evaluating the replicability of significance tests for comparing learning algorithms. In *Advances in Knowledge Discovery and Data Mining (PAKDD 2004)*, LNCS 3056. Springer. https://doi.org/10.1007/978-3-540-24775-3_3
* Craven, M. W., & Shavlik, J. W. (1996). Extracting tree-structured representations of trained networks. *Advances in Neural Information Processing Systems, 8*, 24–30.
* Lin, J., Zhong, C., Hu, D., Rudin, C., & Seltzer, M. (2020). Generalized and scalable optimal sparse decision trees. *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, 6150–6160.
* McTavish, H., Zhong, C., Achermann, R., Karimalis, I., Chen, J., Rudin, C., & Seltzer, M. (2022). Fast sparse decision tree optimization via reference ensembles. *Proceedings of the AAAI Conference on Artificial Intelligence, 36*(9), 9604–9613. https://doi.org/10.1609/aaai.v36i9.21194
* Molnar, C. (2022). *Interpretable machine learning* (2nd ed.). https://christophm.github.io/interpretable-ml-book
* Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning, 52*(3), 239–281. https://doi.org/10.1023/A:1024068626366
* Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). InterpretML: A unified framework for machine learning interpretability. *arXiv:1909.09223*.
* Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
* Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence, 1*(5), 206–215. https://doi.org/10.1038/s42256-019-0048-x
* Yeh, I.-C. (2009). *Default of credit card clients* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
* Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications, 36*(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020
