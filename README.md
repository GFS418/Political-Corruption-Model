# Estimating the Causal Effect of Re-election Incentives on Political Corruption

A from-scratch implementation of Double/Debiased Machine Learning (DML) in R, applied to
Brazilian municipal audit data, to estimate whether a mayor's term (first vs. second) causally
affects the corruption observed in their administration.

The DML estimators, both the Partially Linear Regression (PLR) and Interactive Regression
Model (IRM) formulations, are implemented by hand, including cross-fitting, propensity-score
trimming, and heteroskedasticity-robust (sandwich) standard errors. No off-the-shelf causal-ML
package (e.g. `DoubleML`) is used; only base learners (`glmnet`, `rpart`, `randomForest`, `nnet`)
are called.

---

## Motivation

A naive comparison of corruption between first- and second-term mayors is confounded: municipality
wealth, urbanization, inequality, the mayor's age and education, and state-level fixed effects all
correlate with both a mayor's electoral standing and the opportunity/incentive to engage in
corruption. Estimating the effect of the term indicator therefore requires adjusting for a large,
possibly nonlinear set of confounders.

Double Machine Learning is built for exactly this setting: it uses flexible ML models to partial out
the confounders from both the treatment and the outcome, then estimates the treatment effect on the
residuals. Cross-fitting removes the overfitting bias that would otherwise contaminate the estimate,
and the resulting estimator is √n-consistent and asymptotically normal, so standard inference (t-stats,
p-values) is valid.

## The economics

The empirical question follows the political-agency literature: a first-term mayor who can still run
for re-election faces an accountability incentive that a term-limited second-term mayor does not. If
that incentive disciplines behavior, second-term mayors should show more corruption. The treatment
variable `first` flags first-term status; the outcome is a corruption measure derived from federal
audit reports.

## Method

The estimation follows the standard DML recipe:

1. Nuisance functions. Two functions are learned from the data:
   - `g(X) = E[Y | X]` — the outcome model
   - `m(X) = E[D | X]` — the treatment (propensity) model
2. Cross-fitting. The sample is split into K folds. For each fold, the nuisance models are
   trained on the other K−1 folds and used to predict on the held-out fold, so no observation's
   prediction is ever produced by a model that saw it during training. This is what removes the
   regularization/overfitting bias.
3. Orthogonalization. The treatment effect θ is estimated by regressing the outcome residuals
   `Y − g(X)` on the treatment residuals `D − m(X)`.
4. Inference. Robust standard errors are computed with a sandwich estimator, yielding a t-value
   and p-value for θ.

Two model specifications are implemented:

| Estimator | File section | Notes |
|-----------|--------------|-------|
| PLR — Partially Linear Regression | `double_ml_plr()` | Assumes an additively separable treatment effect. |
| IRM — Interactive Regression Model | `double_ml_irm()` | Allows treatment-effect heterogeneity; adds propensity-score **trimming** to drop observations with extreme scores, and optionally accepts an instrument `Z`. |

Each specification is run with four base learners — LASSO (`cv.glmnet`), a regression tree
(`rpart`), a random forest (`randomForest`), and a single-hidden-layer neural network (`nnet`) — so
the estimate's robustness to the choice of ML model can be compared side by side.

## Data

The analysis expects a file named `corruptiondata.csv` in the project root. It is **not included**
in this repository. It derives from the Brazilian federal municipal-audit program (the *Programa de
Fiscalização por Sorteios Públicos* lottery audits) widely used in the political-economy literature
(e.g. Ferraz & Finan). Key columns:

- Treatment: `first` — first-term mayor indicator
- Outcomes (choose one):
  - `ncorrupt_os` — count of corruption-related audit irregularities
  - `pcorrupt` — share of audited items flagged as corrupt
  - `valor_corrupt` — monetary value associated with corruption
- Confounders: mayor age (`pref_idade_tse`) and education (`pref_escola`); municipality
  characteristics (`p_secundario`, `purb`, `lpib02`, `gini_ipea`, `mun_novo`, `op_01_04`,
  `p_cad_pref`, `ENLP2000`); and state fixed effects (`uf_d1` … `uf_d26`) — up to ~116 covariates in
  the fullest specification.

The outcomes are binarized (`> 0 → 1`) before propensity modeling.

## Confounder specifications

Four nested covariate sets are provided for each outcome, so you can see how the estimate moves as
adjustment gets richer:

1. Mayoral — mayor demographics only
2. Mayoral + Municipal — adds municipality economic/demographic controls
3. Mayoral + Municipal + Political — adds political-competition controls
4. Selected covariates — the full set including all state fixed effects

## Repository structure

```
.
├── PoliticalCorruptionMLAlgo.Rmd   # Full analysis: nuisance learners, PLR, IRM, plots, results
├── corruptiondata.csv              # NOT included — supply your own (see Data section)
└── README.md
```

## Running it

Requirements: R (≥ 4.0) with the following packages:

```r
install.packages(c(
  "glmnet", "caret", "dplyr", "nnet", "randomForest", "rpart",
  "doParallel", "foreach", "sandwich", "lmtest", "data.table", "ggplot2"
))
```

To run:

1. Place `corruptiondata.csv` in the project root.
2. Open `PoliticalCorruptionMLAlgo.Rmd` in RStudio and Knit, or run the chunks in order.
3. To change the outcome or confounder set, edit the `formula_used` assignment near the top of the
   analysis. (The corresponding `y_col` in the PLR/IRM evaluation loops and the plot labels also
   reference the outcome name — update them to match.)

Outputs:

- Distribution of estimated propensity scores by treatment status
- A summary table of PLR results (θ, standard error, t-value, p-value, and nuisance-model RMSEs)
  across all four learners
- The same summary table for the IRM specification

## Results

Across specifications the analysis finds statistically significant evidence that a mayor's term
affects corruption levels, consistent with the accountability-incentive hypothesis. The multi-learner
comparison lets you check that the sign and significance of the estimate are stable regardless of
which ML model is used for the nuisance functions rather than being an artifact of one algorithm.

(Point estimates depend on the chosen outcome and confounder set — run the notebook to reproduce the
exact figures.)

## Notes and caveats

- Missing numeric values are mean-imputed; a more careful pipeline would model missingness explicitly.
- Neural-network nuisance fits use a small single hidden layer and are the least stable of the four
  learners; LASSO and random forest are the most reliable here.
- Identification rests on the standard DML assumption of unconfoundedness given the observed
  covariates. The `first`-term indicator is treated as-if-conditionally-random given X; the IRM path
  also supports an instrument if that assumption is too strong.

## What this project demonstrates

- Implementing a modern causal-inference estimator (DML) from first principles, not from a package
- Cross-fitting, orthogonalized moment conditions, propensity trimming, and sandwich inference
- Comparing multiple ML base learners within a single estimation framework
- Applying econometric theory to a real, messy, high-dimensional dataset

## References

- Chernozhukov et al. (2018), *Double/Debiased Machine Learning for Treatment and Structural
  Parameters*, The Econometrics Journal.
- Ferraz & Finan (2011), *Electoral Accountability and Corruption: Evidence from the Audits of Local
  Governments*, American Economic Review.
