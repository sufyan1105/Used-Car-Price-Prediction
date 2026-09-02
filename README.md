# Used Car Price Predictor

A regression project predicting used car selling prices from CarDekho listings, comparing a linear baseline against two tree-based ensembles.

**Best model: Random Forest — R² 0.9333 on the held-out test set, cross-validated at 0.9262 ± 0.0105.**

---

## Dataset

CarDekho used car listings — 15,411 rows, 14 columns, no missing values.

| Column | Description |
|---|---|
| `car_name`, `brand`, `model` | Vehicle identifiers |
| `vehicle_age` | Age in years (0–29) |
| `km_driven` | Kilometers driven (100 – 3,800,000) |
| `seller_type` | Dealer / Individual / Trustmark Dealer |
| `fuel_type` | Petrol / Diesel / CNG / LPG / Electric |
| `transmission_type` | Manual / Automatic |
| `mileage` | Fuel efficiency, kmpl (4.0 – 33.5) |
| `engine` | Displacement in CC (793 – 6,592) |
| `max_power` | Power output in bhp (38.4 – 626.0) |
| `seats` | Seating capacity |
| `selling_price` | **Target** — price in INR (₹40,000 – ₹3.95 crore) |

The target is steeply right-skewed: median ₹5.56 lakh against a mean of ₹7.75 lakh and a maximum of ₹3.95 crore.

---

## Approach

### Exploratory analysis

- **Target distribution.** Raw `selling_price` is dominated by a long luxury tail. A `log1p` transform produces a far more symmetric distribution.
- **Correlations.** `max_power` and `engine` relate most strongly to price. `vehicle_age` and `km_driven` correlate negatively at moderate strength (~0.33).
- **`km_driven` outliers.** The raw histogram is unreadable — a maximum of 3.8 million km against a median of 50,000 compresses every realistic value into a sliver at the origin. Log-transforming reveals a roughly bell-shaped distribution centered near 50,000 km.
- **Scatter plots** of price against `km_driven`, `max_power`, `engine`, and `vehicle_age` confirm the direction of each relationship. Age shows a near-linear decline rather than the steep-then-flat depreciation curve seen in some markets.
- **Boxplots** across `fuel_type`, `seller_type`, and `transmission_type` show clear separation between categories.
- **Brand averages** rank manufacturers by mean selling price, with luxury marques separating sharply from mass-market brands.

### Preprocessing

| Step | Decision | Reasoning |
|---|---|---|
| `Unnamed: 0` | Dropped | Redundant row index from CSV export |
| `car_name` | Dropped | Duplicates information already split into `brand` and `model` |
| `model` | Dropped | 100+ unique values; one-hot encoding would produce sparse columns covering a handful of cars each, inviting overfitting on 15k rows |
| Missing values | None to handle | Verified zero nulls across all columns |
| `km_driven` | `log1p` applied, raw column dropped | Extreme right skew driven by implausible outliers |
| `selling_price` | `log1p` applied, raw column retained | Skewed target; log-space training helps the linear model. Raw column kept so predictions can be evaluated in rupees |
| Categoricals | One-hot with `drop_first=True` | `brand`, `fuel_type`, `seller_type`, `transmission_type` are nominal — integer encoding would imply a false ordering |

Final feature matrix: **44 columns**, split 12,328 train / 3,083 test.

### Models

All three trained on the log-transformed target, with predictions converted back via `expm1` before metrics are computed — so reported RMSE and MAE are in rupees, not log units.

| Model | Parameters |
|---|---|
| Linear Regression | Baseline, untuned |
| Random Forest | `n_estimators=100`, `max_depth=None`, `random_state=42` |
| Gradient Boosting | `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, `random_state=42` |

---

## Results

| Model | R² | RMSE (INR) | MAE (INR) |
|---|---|---|---|
| **Random Forest** | **0.9333** | **223,999** | **101,298** |
| Gradient Boosting | 0.9225 | 241,494 | 118,994 |
| Linear Regression | 0.8423 | 344,580 | 149,017 |

**5-fold cross-validation (Random Forest):** fold scores 0.935, 0.917, 0.916, 0.942, 0.921 — mean **0.9262**, standard deviation **0.0105**. The tight spread indicates performance does not hinge on which subset the model happens to see.

### Feature importance

| Feature | Importance |
|---|---|
| `max_power` | 0.642 |
| `vehicle_age` | 0.223 |
| `engine` | 0.044 |
| `km_driven_log` | 0.034 |
| `mileage` | 0.020 |
| `brand_Toyota` | 0.007 |
| Remaining 38 features | < 0.005 each |

Two features account for roughly 86% of the model's decision-making.

---

## What the results show

**The relationships are genuinely non-linear.** Random Forest beats Linear Regression by nine R² points and cuts mean absolute error by ₹48,000. Straight-line assumptions cannot capture how price responds to power and age in this data.

**`engine` is redundant, not unimportant.** It correlates strongly with price on its own, yet scores only 0.044 in feature importance. Displacement and maximum power carry overlapping information, and the trees consistently prefer `max_power` when splitting — so the credit accrues there.

**Correlation understated `vehicle_age`.** Its Pearson correlation with price is a moderate 0.33, but it ranks second in importance at 0.223. Correlation captures only linear association; the trees found more signal than that measure could detect.

**The log transform mattered for one model, not all three.** It lifted Linear Regression meaningfully, but barely moved the ensembles — trees split on thresholds, and a monotonic transform leaves every split decision unchanged. It remains worth keeping for the linear baseline and for making the EDA histograms legible.

**Errors scale with price.** The residual plot shows a pronounced funnel: predictions below ₹15 lakh cluster tightly around zero, while errors on expensive vehicles reach ±₹3 million. This is why MAE (₹101k) is less than half of RMSE (₹224k) — squaring lets a small number of large errors dominate.

**The model overpredicts at the top of the range.** Residuals skew negative for expensive cars. Random Forest predicts by averaging leaf values and cannot extrapolate past its training range, so sparse high-price examples get pulled toward the denser middle. The model is dependable for typical used cars and less so above roughly ₹50 lakh.

---

## Known limitations

- Random Forest is cross-validated; Gradient Boosting is not. The headline comparison rests on a single split for two of the three models.
- Gradient Boosting used conservative defaults. A lower learning rate with more estimators would likely close, and possibly reverse, the gap with Random Forest.
- `seats` has a minimum of 0 in the raw data, which is not physically meaningful. Left untouched here, but it suggests a small number of malformed rows worth investigating.
- The `model` column was discarded wholesale. Target encoding or frequency encoding would retain some of that signal without the dimensionality cost of one-hot.

---

## Possible extensions

- Cross-validate all three models so the comparison is on equal footing.
- Tune Gradient Boosting properly via `GridSearchCV` or `RandomizedSearchCV`.
- Retrain on the top five features alone to quantify what the other 39 actually contribute.
- Handle the high-price band separately — either with more luxury training examples or a dedicated model.

---

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Place `cardekho_dataset.csv` in the same directory as the notebook and run `used_car_price_predictor.ipynb` top to bottom.

**Stack:** Python · pandas · NumPy · matplotlib · seaborn · scikit-learn
