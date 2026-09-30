# US Accidents Severity Prediction

Predict accident **Severity (1–4)** using only info known **when the accident is first reported** (time, location, weather, road features).

- **Dataset:** US Accidents (2016–2023), Moosavi et al., Kaggle. Local file: `US_Accidents_March23_sampled_500k.csv` (500K-row sample, already downloaded).
- **Task:** multi-class classification, 4 classes, heavily imbalanced (~80% Severity 2, ~2.6% Severity 4).
- **Main metric:** Macro F1 (dummy baseline ≈ 0.22). Also per-class recall, balanced accuracy.
- **Models:** Dummy → Logistic Regression → Decision Tree → Random Forest → LightGBM/XGBoost.
- **Seed:** `42` everywhere.
- **Severity = traffic impact (delay), not injuries.** Never call it "how dangerous".

> **The one rule:** a feature is allowed only if its value is known at the moment the accident is reported. `End_Time`, `End_Lat`, `End_Lng`, `Distance(mi)`, `Description` leak the answer and must be dropped.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Put `US_Accidents_March23_sampled_500k.csv` in `data/` (not in git — too big). Select the `.venv` kernel in Jupyter / VS Code.

## How to use this file

Each phase below is a ready-to-paste prompt. Run them in order. Don't start a phase until the previous phase's output exists on disk. Paste to Claude:

```
Do Phase N from README.md
```

or copy the prompt block directly.

## Progress

- [x] Phase 1 — Setup
- [ ] Phase 2 — Load & sample
- [ ] Phase 3 — Leakage audit
- [ ] Phase 4 — Cleaning
- [ ] Phase 5 — EDA
- [ ] Phase 6 — Splits
- [ ] Phase 7 — Feature engineering
- [ ] Phase 8 — Class imbalance
- [ ] Phase 9 — Modelling & tuning
- [ ] Phase 10 — Evaluation
- [ ] Phase 11 — Interpretation
- [ ] Phase 12 — Report & presentation

---

## Phase 1 — Setup and organisation

```
Phase 1: Set up the project in this folder.
- Create the folder structure:
    data/       (CSV + all parquet/model files, not pushed to GitHub)
    notebooks/  (01_load_and_sample, 02_cleaning, 03_eda, 04_features, 05_modelling, 06_evaluation .ipynb)
    reports/    (figures + markdown write-ups)
- Write requirements.txt: pandas numpy pyarrow matplotlib seaborn folium scikit-learn
  imbalanced-learn lightgbm xgboost shap.
- Write .gitignore excluding data/ contents.
- git init. Use random seed 42 everywhere.
- Check Python 3.10+ is available and install requirements.
Output: working environment + folder structure. Tick Phase 1 in README Progress.
```

## Phase 2 — Load the data and build the working sample

```
Phase 2: In notebooks/01_load_and_sample.ipynb:
- Load data/US_Accidents_March23_sampled_500k.csv (already a ~500K sample of the
  7.7M-row full dataset, so no chunked sampling needed).
- Print shape, dtypes, Severity counts and shares, rows per year, rows per state.
- Compare Severity shares to the known full-data shares (~80% class 2, ~2.6% class 4)
  and note whether the sample looks representative.
- Save as data/sample.parquet.
- Record the class counts in a markdown cell (they get quoted in the report).
Output: data/sample.parquet, class shares checked.
```

## Phase 3 — Leakage audit and column selection

```
Phase 3: Leakage audit. For every column ask: "could this be known at the moment the
accident is first reported?" If not, drop it.
Apply these decisions:
- DROP: End_Time, End_Lat, End_Lng, Distance(mi), Description (leakage);
  ID (identifier); Country (constant "US"); Weather_Timestamp, Timezone, Airport_Code
  (redundant); Turning_Loop if nunique() == 1.
- KEEP for analysis: Source (ablation later in Phase 9).
- KEEP with care: City, County, Zipcode, Street (high cardinality — no one-hot, encode in Phase 7).
- KEEP: all weather, road-feature, light and time columns.
Verify each decision with actual data (nunique, examples). Save the kept/dropped table
with reasons to reports/leakage_audit.md. Put the code at the top of 02_cleaning.ipynb.
Output: leakage audit table.
```

## Phase 4 — Data cleaning

```
Phase 4: In notebooks/02_cleaning.ipynb, clean sample.parquet (after Phase 3 drops).
Keep a cleaning log: rows and columns before/after every step -> reports/cleaning_log.md.
4.1 Types & duplicates:
- pd.to_datetime(df["Start_Time"], format="mixed").
- Road-feature True/False columns -> 0/1.
- Drop exact duplicates, log count.
- Drop near-duplicates: the same accident reported by two providers (Source).
  Key = Start_Time rounded to 5 min + Start_Lat/Start_Lng rounded to 3 decimals (~100 m).
  Keep the first, log count. Otherwise copies land in both train and test and inflate scores.
4.2 Missing values (measure first: df.isna().mean().sort_values()):
- Columns > 40% missing: drop.
- Wind_Chill(F): drop (mostly missing, ~copy of Temperature).
- Precipitation(in): fill 0 + add precip_missing flag.
- Other numeric weather (Temperature, Humidity, Pressure, Visibility, Wind_Speed):
  leave NaN — median imputation happens inside the Pipeline on train only (Phase 9).
- Categorical (Weather_Condition, Wind_Direction, light columns): fill "Unknown".
- Few rows missing City/Zipcode/light columns: drop those rows.
4.3 Impossible values -> NaN: Temperature > 130°F or < -50°F, Wind_Speed > 200 mph,
  Visibility > 100 mi. Check df.describe() + box plots, document every threshold.
4.4 Standardise categories:
- Wind_Direction: merge Calm/CALM, West/W, South/S, Variable/VAR, etc. into
  16 compass points + CALM + VAR.
- Weather_Condition (100+ labels) -> ~8 groups: Clear, Cloudy, Rain,
  Heavy rain/Thunderstorm, Snow/Ice, Fog/Haze/Smoke, Windy, Other.
Output: data/clean.parquet + cleaning log.
```

## Phase 5 — Exploratory data analysis

```
Phase 5: In notebooks/03_eda.ipynb, make 10–15 charts saved to reports/.
Each chart: title, axis labels, units, and a 1–2 sentence insight in a markdown cell.
Answer:
1. How imbalanced is Severity? (bar: counts + %)
2. When do accidents happen? (counts by hour, weekday, month, year)
3. Are rush-hour/night accidents more severe? (100% stacked bar of Severity by hour)
4. Where? (top 10 states & cities; folium heat map on a 50K subsample -> HTML)
5. Does weather matter? (Severity share per weather group; box plots of Visibility and
   Temperature by Severity)
6. Do road features matter? (% Severity 3–4 near Junction, Traffic_Signal, Crossing, Stop)
7. Day or night? (Severity share by Sunrise_Sunset)
8. Numeric correlations (heat map)
9. Stable over time? (Severity share by year and by Source — drift/reporting differences).
   Check where Severity 1 comes from: if it clusters in one Source/period, say so.
Drop any chart without an insight. Summarise key findings at the end of the notebook.
Output: figures + written insights.
```

## Phase 6 — Train / validation / test split

```
Phase 6: At the top of notebooks/04_features.ipynb, split clean.parquet BEFORE any
feature is fitted (encoding, clustering, imputation, resampling, scaling).
- Main split: stratified on Severity, 70% train / 15% val / 15% test, random_state=42.
- Time-based robustness set: train on 2016–2021, test on 2022–2023.
- Save all splits to data/ (train/val/test + time_train/time_test parquet).
- Print class shares per split to confirm stratification.
Rules: val for every choice (imbalance method, model, hyperparams). Test used ONCE at the end.
Output: fixed splits on disk.
```

## Phase 7 — Feature engineering

```
Phase 7: In notebooks/04_features.ipynb, add features to each split from Phase 6.
Stateless row-by-row features (one function, applied identically to every split):
- hour, weekday, month, year from Start_Time
- is_weekend (weekday >= 5)
- is_rush_hour (weekday and hour in 7–9 or 16–19)
- is_night (Sunrise_Sunset == "Night")
- season (month -> Winter/Spring/Summer/Autumn)
- weather_group + bad_weather flag (rain, snow, fog, storm)
- road_feature_count (sum of road-feature flags)
- is_highway (Street contains "I-", "Interstate", "Hwy", "Highway", "US-", "Fwy")
Fitted features — inside the sklearn Pipeline + ColumnTransformer, fit on train only:
- State: one-hot
- City, County: frequency encoding (counts from train)
- location_cluster: KMeans(~50, random_state=42) on Start_Lat/Start_Lng
- median imputer; scaler (Logistic Regression only)
Caveat: `year` must be dropped for the time-based run — trees treat unseen years
(2022–23) as the last seen year (2021), so it carries no signal there.
Save the featured splits to data/ and a data dictionary (reports/data_dictionary.md).
Output: model-ready splits + data dictionary.
```

## Phase 8 — Handling class imbalance

```
Phase 8: In notebooks/05_modelling.ipynb, compare at least two imbalance strategies on
the validation set using macro F1 (with one fixed model, e.g. LightGBM or LogReg):
- class_weight="balanced" (start here)
- Random undersampling of Severity 2 (train set only)
- SMOTENC (imbalanced-learn; plain SMOTE interpolates one-hot columns, which is
  meaningless) — train set only, on a 100K subsample; never on val/test. Skip if too slow.
- Decision-threshold tuning: multiply each class probability by a weight w_k, pick the
  weights on val to maximise macro F1 (e.g. coordinate search over a grid), predict
  argmax(w_k * p_k). Freeze the weights and apply them unchanged on test.
Report a table: strategy vs macro F1, per-class recall. Pick and justify one.
Optional fallback: binary target high impact (3–4) vs low (1–2).
Output: chosen strategy with justification.
```

## Phase 9 — Modelling and tuning

```
Phase 9: In notebooks/05_modelling.ipynb, using the Phase 8 strategy, train on train,
score on val (macro F1), and record training time for:
0. DummyClassifier(most_frequent) — expect macro F1 ≈ 0.22
1. LogisticRegression (multinomial, balanced, scaled)
2. DecisionTreeClassifier
3. RandomForestClassifier
4. LightGBM (or XGBoost)
5. (optional) MLPClassifier
Use Pipeline(ColumnTransformer(SimpleImputer(median) for num,
OneHotEncoder(handle_unknown="ignore", min_frequency=0.01) for cat), clf).
Tune the best model with RandomizedSearchCV(3-fold StratifiedKFold, scoring="f1_macro")
on a 100–200K train subsample. LightGBM params: num_leaves, learning_rate, n_estimators,
min_child_samples, subsample, colsample_bytree. n_jobs=-1, random_state=42.
Ablation: best model with vs without Source, report the difference.
Single-Source run: retrain the best model on the largest Source only (train + val
filtered), report macro F1. Shows how much the model learns reporting practice vs road/weather.
Save the tuned model (joblib) to data/.
Output: trained models, tuned best model, training times.
```

## Phase 10 — Evaluation

```
Phase 10: In notebooks/06_evaluation.ipynb, evaluate on the TEST set (once):
- Macro F1 (main), classification_report (per-class P/R/F1), balanced accuracy,
  weighted F1, OvR ROC-AUC, PR-AUC for Severity 4 (better than ROC-AUC for a rare class),
  accuracy (explain why it misleads: all-2 gives ~80%).
- Also report macro F1 over classes 2–4 only as a side number (Severity 1 is <1% of rows
  and drags macro F1 down).
- Fill the results table: Model | Macro F1 | Balanced acc | Recall Sev 4 | ROC-AUC OvR | Train time
  (Dummy row: ≈0.22 | 0.25 | 0.00 | 0.50 | —). Include val and test.
- Normalised confusion matrix (expect 1↔2 and 3↔4 confusion).
- Error analysis: where the model fails by State, hour, weather group, Source.
- Time-based check: train on 2016–2021, test on 2022–2023 (without `year`); compare with
  random split, explain drop.
- (optional) Spatial check: GroupKFold / group split by County or location_cluster, so
  test locations are unseen. Compare with the random split to measure location memorisation.
Save figures to reports/ and table to reports/results.md.
Output: comparison table, confusion matrix, error analysis.
```

## Phase 11 — Interpretation

```
Phase 11: Explain the best model.
- Built-in feature importance (gain) + permutation importance on the test set.
- SHAP summary plot on 5–10K rows.
- Write 3–5 plain-language insights, each backed by a number
  (e.g. "highway accidents at rush hour are X times more likely to be Severity 4").
- Limitations paragraph: severity = traffic impact not injury; providers (Source) may
  record severity differently; sampled data; data ends March 2023; coverage varies by state.
Save plots to reports/ and text to reports/interpretation.md.
Output: importance plots, key insights, limitations.
```

## Phase 12 — Report and presentation

```
Phase 12: Final deliverables.
- Report (PDF) with chapters: Abstract (~150 words), 1 Introduction, 2 Dataset (+ citation),
  3 Data preparation (sampling, leakage audit, cleaning), 4 EDA, 5 Feature engineering,
  6 Methodology (split, imbalance, models, tuning, metrics), 7 Results (table, confusion
  matrix, time-based check), 8 Interpretation, 9 Limitations & future work,
  10 Conclusion, References & appendix. Build it from the reports/*.md files and figures.
- Slides (10–12): problem → data → leakage → EDA highlights → models → results →
  insights → limitations.
- Repo cleanup: README "how to run" section, requirements.txt, notebooks 01–06 run
  top to bottom, fixed seeds.
- Optional: small Streamlit app (enter time, place, weather -> predicted severity).
- Prep Q&A notes on leakage, imbalance, metric choice.
Output: report PDF, slides, clean repo.
```

---

## Deliverables checklist

- [ ] Sample + cleaned dataset (Parquet) with data dictionary
- [ ] Leakage audit table
- [ ] Cleaning log
- [ ] Notebooks 01–06 run top to bottom
- [ ] EDA figures with insights
- [ ] Model comparison table (val + test)
- [ ] Confusion matrix, per-class report, time-based result
- [ ] Feature importance + SHAP with interpretation
- [ ] Final report (PDF)
- [ ] Slides
- [ ] Repo with README + requirements.txt

## Risks

| Risk | Mitigation |
|---|---|
| Leakage inflates scores | Phase 3 audit; near-duplicate removal; split before fitting features |
| Imbalance hides minority performance | Macro F1, per-class recall, class weights |
| Source bias | EDA by Source; ablation with/without Source; single-Source run |
| Drift over years | Time-based test (Phase 6) |
| Slow training/tuning | Tune on subsample, LightGBM, `n_jobs=-1` |
| Schedule slips | Freeze scope after Phase 6; binary-target fallback |

## References

1. Moosavi, S., Samavatian, M. H., Parthasarathy, S., & Ramnath, R. (2019). *A Countrywide Traffic Accident Dataset.* arXiv:1906.05409.
2. Moosavi, S., Samavatian, M. H., Parthasarathy, S., Teodorescu, R., & Ramnath, R. (2019). *Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights.* ACM SIGSPATIAL 2019.
3. Moosavi, S. *US Accidents (2016–2023)* [Dataset]. Kaggle. https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
4. Moosavi, S. *US-Accidents: A Countrywide Traffic Accident Dataset.* https://smoosavi.org/datasets/us_accidents
