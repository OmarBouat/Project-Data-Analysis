# Mental Health Among US College Students
## A Data-Driven Analysis — Harvard Healthy Minds Study 2023–2025

**Dataset:** Harvard Healthy Minds Study (HMS) — Public Release  
**Date:** May 2025

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset](#2-dataset)
3. [Phase 1 — Setup & Data Loading](#3-phase-1--setup--data-loading)
4. [Phase 2 — Preprocessing](#4-phase-2--preprocessing)
5. [Phase 3 — Exploratory Data Analysis](#5-phase-3--exploratory-data-analysis)
6. [Phase 4 — PCA Analysis](#6-phase-4--pca-analysis)
7. [Phase 5 — Correspondence Analysis (AFC)](#7-phase-5--correspondence-analysis-afc)
8. [Phase 6 — Machine Learning](#8-phase-6--machine-learning)
9. [Key Findings](#9-key-findings)
10. [Limitations](#10-limitations)
11. [Conclusion](#11-conclusion)

---

## 1. Introduction

Mental health has become one of the most pressing concerns in higher education. Rising rates of depression, anxiety, and suicidal ideation among college students have prompted universities, policymakers, and researchers to seek a deeper understanding of the factors driving this crisis.

**Core question:** *Who are the most mentally vulnerable college students in the US — and can we predict them?*

This project analyzes data from the Harvard Healthy Minds Study (HMS), one of the largest and most rigorous annual surveys of college student mental health in the United States. By combining two consecutive survey waves (2023–2024 and 2024–2025), the analysis covers nearly **190,000 students** across hundreds of institutions.

The analysis is structured into **seven phases**: data loading, preprocessing, exploratory data analysis, principal component analysis, correspondence analysis, machine learning, and a dedicated balanced classification experiment — moving from descriptive understanding to predictive modeling.

---

## 2. Dataset

### 2.1 Source
The **Harvard Healthy Minds Study** is conducted annually by the Healthy Minds Network, a research consortium based at the University of Michigan. The dataset is publicly released for research purposes.

### 2.2 Survey Waves

| Year | Students | Columns |
|------|----------|---------|
| 2023–2024 | 104,729 | 1,554 |
| 2024–2025 | 84,735 | 1,608 |
| **Combined** | **189,464** | **83 selected** |

### 2.3 Clinical Instruments

The survey uses four internationally validated clinical scales:

| Instrument | Measures | Items | Score Range |
|-----------|----------|-------|-------------|
| **PHQ-9** | Depression severity | 9 | 0–27 |
| **GAD-7** | Anxiety severity | 7 | 0–21 |
| **Diener Flourishing Scale** | Psychological well-being | 8 | 8–56 |
| **UCLA Loneliness Scale (3-item)** | Loneliness | 3 | 3–9 |

**PHQ-9** (Kroenke & Spitzer, 2001) screens for major depressive disorder based on DSM criteria. Scores ≥ 10 indicate moderate-to-severe depression.

**GAD-7** (Spitzer et al., 2006) screens for generalized anxiety disorder. Scores ≥ 10 indicate moderate-to-severe anxiety.

**Diener Flourishing Scale** (Diener, 2009) measures positive psychological well-being — purpose, engagement, and social relationships. Unlike the first two, it measures what is going right rather than what is wrong.

**UCLA Loneliness Scale** measures perceived social isolation. Higher scores indicate greater loneliness.

### 2.4 Additional Variables
Beyond clinical scores, the dataset includes:
- **Demographics:** age, gender identity, race/ethnicity, international status, year in school
- **Financial stress:** Pell grant status (low-income proxy), housing insecurity, food insecurity
- **Behaviors:** substance use, self-injury, therapy, psychiatric medication
- **Institution:** type (2-year, 4-year public/private, graduate), size, public vs. private, geographic region
- **Treatment:** therapy use, medication use, perceived need, unmet need

---

## 3. Phase 1 — Setup & Data Loading

### Objective
Load both survey waves, select relevant columns, and combine into a single analysis-ready dataset.

### Process
1. Both raw CSV files loaded (104,729 + 84,735 rows)
2. From 1,554+ columns, ~80 relevant variables selected across six groups: demographics, institution characteristics, PHQ-9 items, GAD-7 items, Diener items, UCLA Loneliness items, outcome variables, treatment variables, and stress variables
3. Only columns present in **both years** retained to ensure comparability
4. Both years concatenated with a `year` column added
5. Column reference table created documenting each variable's meaning and coding

### Output
Combined dataset: **189,464 students × 83 columns**

---

## 4. Phase 2 — Preprocessing

### Objective
Clean, encode, and enrich the raw combined dataset for analysis.

### Steps

**Missing Value Audit**  
Identified columns with missing data. Most clinical score columns had 8–15% missing, primarily due to survey skip logic (students who did not complete certain sections).

**Type Conversion & Label Encoding**  
- All numeric columns forced to proper float/int types
- Gender encoded as: Man, Woman, Non-binary, Genderqueer, Other/Unknown
- Race encoded as: White, Black, Hispanic, Asian, Indigenous, Other/Multiracial
- Year in school mapped to: 1st Year through 5th+ Year
- Institution type mapped to: 2-Year, 4-Year Public, 4-Year Private, Graduate, Other
- Institution size mapped to: <1k, 1–5k, 5–10k, 10–20k, >20k

**Derived Scores**  
- `phq9_total`: sum of 9 PHQ items (0–27)
- `gad7_total`: sum of 7 GAD items (0–21)
- `flourishing`: sum of 8 Diener items (8–56)
- `loneliness`: sum of 3 UCLA items (3–9)
- `mh_burden`: composite mental health burden (0–1), weighted average of normalized PHQ-9, GAD-7, and loneliness scores

**Derived Flags**  
- `low_income`: Pell grant recipient (binary)
- `housing_insecure`: housing worry ≥ 2 (binary)
- `sui_level`: composite suicidality (None / Ideation / Plan / Attempt)
- `unmet_need`: perceives need but receives no therapy or medication

### Output
Clean dataset saved as `HMS_clean.csv` — **189,464 rows × 83 columns**

### Note on Imputation
No imputation was performed. The dataset is large enough (189,464 rows) that complete-case analysis is appropriate — even after dropping rows with missing clinical scores, each analysis retains 150,000–170,000 complete cases. Additionally, missing values in HMS are primarily structural (survey skip logic), meaning students who skipped the PHQ-9 likely have lower distress — imputing their scores with the mean would be misleading.

---

## 5. Phase 3 — Exploratory Data Analysis

### Objective
Answer 10 specific research questions through targeted visualizations.

### Questions & Key Findings

**Q1: Who are the students?**  
The sample is predominantly female (66%), aged 18–22 (64%), attending 4-year public institutions (52%). The largest racial group is White (55%), followed by Hispanic (16%) and Asian (14%).

**Q2: How prevalent are depression, anxiety, and suicidality?**  
- Any depression (PHQ-9 based): ~39% of students
- Major depression (PHQ-9 ≥ 10): ~19%
- Any anxiety (GAD-7 based): ~35%
- Severe anxiety (GAD-7 ≥ 15): ~16%
- Suicidal ideation (past year): ~12%
- Suicide plan: ~6%
- Suicide attempt: ~1.4%

**Q3: Did mental health change from 2023–24 to 2024–25?**  
Depression prevalence decreased slightly from 39.2% to 35.8% (Δ = −3.4 percentage points). Anxiety showed a similar modest decline. Suicidal ideation dropped from 12.7% to 10.4%.

**Q4: Which demographic groups are most at risk?**  
Non-binary and genderqueer students show substantially higher rates across all outcomes. Among racial groups, Indigenous students report the highest anxiety severity. Women report higher depression and anxiety than men.

**Q5: Does financial stress predict mental health outcomes?**  
Pell grant recipients (low-income proxy) show higher rates of depression, anxiety, and suicidal ideation than non-recipients. Housing insecurity shows a strong dose-response relationship — students with high housing worry have markedly worse outcomes than those with no worry.

**Q6: How does loneliness relate to depression and anxiety?**  
Strong positive correlations: loneliness ↔ depression (r ≈ 0.55), loneliness ↔ anxiety (r ≈ 0.50). Students with high loneliness scores cluster heavily in the severe depression category.

**Q7: Does institution type/size affect mental health?**  
2-year college students show slightly higher severe depression rates than 4-year students. Institution size shows minimal effect. Public vs. private shows small differences.

**Q8: How many students need help but don't get it?**  
Among depressed students, approximately 45–50% are not in therapy and not on medication. A significant proportion of students with perceived unmet need receive no treatment — the treatment gap is one of the most actionable findings in the dataset.

**Q9: PHQ-9 symptom profile of depressed vs. non-depressed students**  
Depressed students score 3–5× higher on every PHQ-9 item. The largest gaps are on PHQ4 (fatigue), PHQ2 (hopelessness), and PHQ6 (feeling like a failure).

**Q10: Who is most at risk for suicidality?**  
Non-binary students have the highest suicidal ideation rates (~28% vs ~10% for men and women). Among racial groups, Indigenous students show the highest rates. Suicidality peaks in 2nd and 3rd year students.

---

## 6. Phase 4 — PCA Analysis

### Objective
Uncover the latent structure across 15 diverse variables spanning four domains: mental health, financial stress, behaviors, and institutional context.

### Why Not Item-Level PCA
Running PCA on PHQ-9 + GAD-7 items produces one dominant component (PC1 ≈ 52% variance) because items within a clinical scale are designed to be internally consistent. This is mathematically expected but analytically uninformative. Instead, PCA was applied to conceptually diverse variables across four domains.

### Variables (15 total)

| Domain | Variables |
|--------|-----------|
| Mental Health | PHQ-9 total, GAD-7 total, Flourishing, Loneliness |
| Financial | Low income, Housing insecurity, Food insecurity |
| Behavior | Substance use, Self-injury, In therapy |
| Context | Age, International student, Year in school, Public institution, Institution size |

### Results

Cross-domain correlations are low (e.g., PHQ-9 vs. institution size r = −0.017), confirming the variables are genuinely diverse and PCA will find distinct components.

**Variance explained:**
- PC1: ~28–32% — Mental Health Burden (high depression, anxiety, loneliness; low flourishing)
- PC2: ~15–18% — Socioeconomic Vulnerability (low income, housing/food insecurity)
- PC3: ~10–12% — Behavioral Response (therapy use, self-injury, substance use)
- PC4: ~8–10% — Institutional Context (institution size, public vs. private)

**Key insight:** The four domains are separable — mental health burden, financial vulnerability, behavioral patterns, and institutional context represent genuinely distinct dimensions of student experience.

---

## 7. Phase 5 — Correspondence Analysis (AFC)

### Objective
Map associations between categorical clinical variables to reveal *how* they are associated — which severity levels cluster together, which repel, and which are independent.

### Method
Analyse Factorielle des Correspondances (AFC) — a dimensionality reduction method for contingency tables. Unlike chi-square which only answers "are they associated?", AFC shows the geometric structure of the association.

### Analyses (all selected with Cramér's V ≥ 0.25)

| AFC | Variables | Cramér's V | Effect |
|-----|-----------|-----------|--------|
| AFC1 | Depression Severity × Anxiety Severity | **0.532** | Large |
| AFC2 | Loneliness Level × Depression Severity | **0.380** | Medium |
| AFC3 | Self-Injury × Depression Severity | **0.363** | Medium |
| AFC4 | Flourishing Level × Depression Severity | **0.349** | Medium |
| AFC5 | Flourishing Level × Loneliness Level | **0.335** | Medium |
| AFC6 | Unmet Need × Depression Severity | **0.259** | Small-Medium |

### Key Findings

**AFC1 (Depression × Anxiety, V=0.532):** The strongest association. Severe depression clusters tightly with severe anxiety — the two scales are not independent dimensions but largely co-occur. Mild depression clusters with mild anxiety, forming a clear diagonal gradient on the correspondence map.

**AFC2 (Loneliness × Depression, V=0.380):** High loneliness clusters strongly with severe depression. Low loneliness clusters with mild/moderate depression. The gradient is clean and monotonic.

**AFC3 (Self-Injury × Depression, V=0.363):** Students who self-injure cluster overwhelmingly with severe and moderately severe depression. Non-self-injuring students cluster with mild/moderate categories.

**AFC4 (Flourishing × Depression, V=0.349):** High flourishing clusters with mild depression; low flourishing clusters with severe depression. Confirms that positive well-being and depression severity are inversely related but not perfectly opposite.

**AFC5 (Flourishing × Loneliness, V=0.335):** High flourishing clusters with low loneliness; low flourishing clusters with high loneliness. The two well-being dimensions form a clean diagonal — they are measuring related but distinct constructs.

**AFC6 (Unmet Need × Depression, V=0.259):** Students with unmet mental health need cluster with moderate-to-severe depression. Treated students are spread across severity levels. Students with no perceived need cluster with mild depression.

---

## 8. Phase 6 — Machine Learning

### Objective
Two complementary approaches: unsupervised clustering to discover natural student profiles, and supervised classification to predict suicidal ideation. A separate notebook (`classification_balanced.ipynb`) repeats the classification on a perfectly balanced dataset (1:1 undersampling) to assess the effect of class imbalance on model performance.

---

### Part A: K-Means Clustering

**Goal:** Find natural groupings of students based on PHQ-9, GAD-7, Loneliness, and Flourishing scores — without predefined labels.

**Method:** K-Means clustering with K=4 (selected via elbow method). Features standardized before clustering.

**Results — 4 Cluster Profiles:**

| Cluster | Profile | Characteristics |
|---------|---------|----------------|
| Resilient | Low distress, high well-being | Low PHQ-9/GAD-7/Loneliness, High Flourishing |
| Moderate Distress | Average on all dimensions | Mid-range scores across all four features |
| High Distress | Elevated symptoms, lonely | High PHQ-9/GAD-7/Loneliness, Low-Medium Flourishing |
| Severe Distress | Maximum burden | Very high PHQ-9/GAD-7/Loneliness, Very low Flourishing |

Clusters are visualized via radar charts (one per cluster) and in PCA space with centroids marked.

---

### Part B: Classification — Predicting Suicidal Ideation

**Target:** `sui_idea` — suicidal ideation in the past year (binary: 0/1)  
**Class balance:** 12% positive (18,750 / 156,363)  
**Split:** 80% train / 20% test, stratified

**Features (14 variables):**
GAD-7 score, loneliness, flourishing, low income, housing insecurity, food insecurity, self-injury, therapy use, medication use, public institution, institution size, year in school, woman (binary), non-binary (binary)

**Models:** Logistic Regression (interpretable baseline) and Random Forest (non-linear). Two training strategies were compared:
- **Original (imbalanced):** `class_weight='balanced'` to compensate for the 12% positive rate
- **Balanced (1:1 undersampling):** majority class randomly reduced to match minority class size (~37,500 total training samples)

**Results — Original (imbalanced) dataset:**

| Model | ROC-AUC | Avg Precision | Recall | Precision | Accuracy |
|-------|---------|--------------|--------|-----------|----------|
| Logistic Regression | 0.8475 | 0.4528 | 0.77 | 0.31 | 0.76 |
| Random Forest | 0.8485 | 0.4560 | 0.78 | 0.30 | 0.76 |

**Results — Balanced (1:1 undersampled) dataset:**

| Model | ROC-AUC | Avg Precision | Recall | Precision | Accuracy |
|-------|---------|--------------|--------|-----------|----------|
| Logistic Regression | ~0.845 | ~0.80 | ~0.77 | ~0.72 | ~0.78 |
| Random Forest | ~0.848 | ~0.82 | ~0.78 | ~0.74 | ~0.79 |

Undersampling significantly improves precision (from ~0.31 to ~0.72) and accuracy (from 0.76 to ~0.79) while maintaining similar AUC and recall. The balanced experiment is documented in `classification_balanced.ipynb`.

**Feature Importance:**  
The most important predictors of suicidal ideation (consistent across both models):
1. GAD-7 score (anxiety)
2. Loneliness
3. Flourishing (inverse)
4. Self-injury history
5. Food insecurity

Demographic and institutional variables (gender, institution type, size) contribute minimally compared to clinical and financial stress variables.

**Cross-Validation (5-fold):**
- Logistic Regression: AUC = 0.847 ± 0.003
- Random Forest: AUC = 0.849 ± 0.002

Stable performance across folds confirms the models generalize well.

**Additional figures produced:**
- Threshold analysis (precision/recall/F1 vs decision threshold)
- Predicted probability distributions by true class
- Calibration curves
- Learning curves (training vs validation accuracy)
- Model comparison summary chart across all metrics

---

## 9. Key Findings

1. **High prevalence:** ~39% of US college students meet criteria for any depression, ~35% for any anxiety, and ~12% report suicidal ideation — rates that are clinically alarming at population scale.

2. **Modest improvement:** Depression and suicidal ideation rates declined slightly from 2023–24 to 2024–25 (−3.4pp and −2.3pp respectively), suggesting a small positive trend but no resolution of the crisis.

3. **Non-binary students are most at risk:** Non-binary and genderqueer students show dramatically higher rates across all outcomes — suicidal ideation ~28% vs ~10% for men and women.

4. **Financial stress is a strong predictor:** Housing insecurity shows a clear dose-response relationship with depression, anxiety, and suicidality. Low-income students (Pell grant recipients) are consistently more affected.

5. **Loneliness is central:** Loneliness correlates strongly with both depression (r ≈ 0.55) and anxiety (r ≈ 0.50), and high loneliness clusters tightly with severe depression in the AFC analysis.

6. **Treatment gap is large:** Approximately 45–50% of depressed students receive no treatment. A significant proportion of students with perceived unmet need access no care.

7. **Depression and anxiety are largely one dimension:** AFC1 (V=0.532) and PCA both confirm that depression and anxiety severity co-occur strongly — they are not independent constructs in this population.

8. **Clinical variables dominate prediction:** In the ML classification, anxiety score, loneliness, flourishing, and self-injury history are far more predictive of suicidal ideation than demographic or institutional variables.

---

## 10. Limitations

- **Self-report bias:** All measures are self-reported; students may under- or over-report symptoms.
- **Sampling bias:** HMS surveys students who opt in; severely distressed students may be underrepresented (too unwell to complete the survey) or overrepresented (more motivated to participate).
- **Cross-sectional design:** Both survey waves are cross-sectional snapshots, not longitudinal tracking of the same students. Year-over-year comparisons reflect different cohorts, not individual change.
- **Missing data:** 8–15% missing on clinical score columns due to survey skip logic. Complete-case analysis was used, which may introduce bias if missingness is not random.
- **Mixed variable types in PCA:** The PCA in Phase 4 mixes continuous and binary variables, which violates the strict assumptions of standard PCA. Results should be interpreted with this caveat.
- **Class imbalance in ML:** The 12% positive rate for suicidal ideation means precision for the positive class is inherently limited in the original experiment. The balanced experiment (`classification_balanced.ipynb`) addresses this by undersampling the majority class to a 1:1 ratio, substantially improving precision at the cost of a smaller training set.
- **Undersampling information loss:** The balanced dataset uses only ~37,500 of the available 156,363 training samples, discarding ~75% of the negative class. This may reduce the model's exposure to the full diversity of non-suicidal students.

---

## 11. Conclusion

This analysis provides a comprehensive, multi-method examination of mental health among US college students using one of the largest available datasets on the topic. The findings confirm the scale of the crisis — with roughly one in three students meeting criteria for depression or anxiety — while also revealing important nuances: non-binary students are disproportionately affected, financial stress is a strong and modifiable risk factor, loneliness is a central mediating variable, and a large treatment gap persists.

The machine learning results demonstrate that suicidal ideation can be predicted with reasonable accuracy (AUC ≈ 0.85) from observable clinical and financial variables, suggesting that targeted early intervention is feasible. The most actionable finding is the treatment gap: nearly half of depressed students receive no care, pointing to a systemic failure in access rather than a failure of awareness.

---

*Report generated from analysis notebooks: phase1 through phase6 + classification_balanced*  
*Dataset: HMS 2023-2024 and 2024-2025 Public Release*  
*Notebooks: phase1_setup_data_loading.ipynb, phase2_preprocessing.ipynb, phase3_eda.ipynb, phase4_pca.ipynb, phase5_afc.ipynb, phase6_ml.ipynb, classification_balanced.ipynb*
