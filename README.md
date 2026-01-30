# Mandibular Condyle Age Estimation Pipeline

## Repository Purpose

This repository implements a complete and reproducible pipeline for supervised regression modeling aimed at chronological age estimation using mandibular condyle–related morphometric features extracted from orthodontic radiographic data.

The codebase covers data preprocessing, feature selection, model training, cross-validation, hyperparameter optimization, performance uncertainty estimation, and model comparison.

---

## Study Design and Data

* **Design:** Retrospective, observational, cross-sectional
* **Population:** Children and adolescents aged 5–18 years
* **Data source:** Orthodontic records from a private clinic
* **Imaging modalities:**

  * Digital panoramic radiographs
  * Lateral cephalometric radiographs
* **Ethics:** Approved by an institutional ethics committee. All data were anonymized prior to processing.

### Exclusion Criteria

* Systemic syndromes or congenital malformations
* Facial or mandibular trauma
* Temporomandibular joint pathology
* Previous craniofacial surgery
* Radiographs with insufficient quality for morphometric analysis

---

## Morphometric Feature Extraction

### Software

* **Image visualization:** Studio 3
* **Measurements:** ImageJ (NIH)

### Calibration

Radiographic images were calibrated using an internal reference provided by the imaging system. All distances were converted from pixels to millimeters prior to analysis.

### Extracted Features

From **panoramic radiographs**:

* Condyle height
* Condyle width
* Mandibular ramus height
* Mandibular ramus width

From **lateral cephalometric radiographs**:

* Condylion–Gnathion length (Co–Gn)

Bilateral measurements were averaged to reduce redundancy and multicollinearity. Sex was included as an adjustment variable.

---

## Measurement Reliability

* Two trained examiners performed all measurements.
* A random subset of images was remeasured after a two-week interval.
* Intra- and inter-examiner agreement was quantified using the Intraclass Correlation Coefficient (ICC).

---

## Feature Selection Strategy

1. Independent evaluation of each feature against chronological age using Pearson correlation.
2. Retention of predictors with statistically significant associations (p < 0.05).
3. Post hoc statistical power estimation using G*Power (exact test, bivariate normal model).
4. Aggregation of bilateral measures to mitigate multicollinearity and variance inflation.

---

## Machine Learning Models

Supervised regression algorithms implemented in this pipeline:

* Linear Regression
* Support Vector Regression (RBF kernel)
* K-Nearest Neighbors
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* AdaBoost Regressor
* Multilayer Perceptron Regressor

The selection includes linear, tree-based, instance-based, and neural network models to capture heterogeneous data structures.

---

## Training and Validation Workflow

* **Data split:**

  * Training set: 80%
  * Test set: 20% (held-out, independent)

### Cross-Validation

* Five-fold cross-validation applied exclusively to the training set.
* Each fold used once as validation to ensure stability and reduce partition bias.

---

## Hyperparameter Optimization

* Optimization performed only on training data.
* Grid search integrated with cross-validation.
* Mean Squared Error (MSE) used as the optimization objective.

This strategy minimizes overfitting and favors generalizable configurations.

---

## Model Evaluation and Uncertainty Estimation

Regression metrics computed:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* Coefficient of Determination (R²)

Uncertainty estimation:

* Bootstrap resampling (1,000 iterations)
* 95% confidence intervals generated for all metrics
* Applied to both cross-validation and test predictions

---

## Model Comparison Framework

* Pairwise comparisons based on MAE distributions.
* Differences considered relevant only when confidence intervals excluded the null value.
* Error distributions explored using kernel density estimation (KDE)–based visualizations.

---

## Feature Contribution Analysis

Feature importance was extracted using native scikit-learn tools for models supporting direct interpretability.

The following models were excluded from feature attribution due to lack of native mechanisms:

* K-Nearest Neighbors
* Support Vector Regression
* Multilayer Perceptron

---

## Computational Environment

* **Language:** Python
* **Platform:** Google Colaboratory
* **Core libraries:**

  * scikit-learn
  * NumPy
  * pandas
  * matplotlib

---

## Reproducibility

All scripts required to reproduce the full analytical pipeline are available in this repository, including data preprocessing, model training, validation, uncertainty estimation, and visualization.

The pipeline is modular and can be adapted to independent datasets with similar morphometric inputs.
