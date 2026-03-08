# Data Science Style Guide – Pandas & NumPy Best Practices

> Supplement to [PEP8](https://peps.python.org/pep-0008/) and the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).  
> Covers: Pandas, NumPy, Scikit-learn, and general DS patterns.

---

## Why This Guide Exists

General-purpose style guides like PEP8 and the Google Python Style Guide define how to write clean, readable Python. They do not address the specific patterns and failure modes that appear when working with data – and that gap is where most Data Science code quality problems actually live.

A notebook that passes a linter can still silently corrupt a DataFrame through a missing `.copy()`, inflate model metrics through preprocessing leakage, or produce irreproducible results because a random seed was set in the wrong place. These are not syntax issues. They are structural mistakes that look correct to a code reviewer unfamiliar with the Pandas/NumPy execution model, and they only surface as bugs in production or as inflated benchmark numbers that never generalize.

This guide codifies the patterns that prevent these mistakes. Each rule targets a specific, recurring failure mode observed in real DS projects:

- **Correctness** – rules that prevent silent data corruption and invalid evaluation results
- **Performance** – rules that replace Python-level loops with C-level array operations
- **Reproducibility** – rules that ensure experiments can be re-run and results verified
- **Maintainability** – rules that make intent explicit so code can be reviewed, extended, and handed off

The rules are not opinions about style. They have concrete, measurable consequences when violated. The explanations in each section describe exactly what goes wrong without them.

---

## 1. Vectorization – Priority Order

**Rule:** Always use the highest available level:
1. Vectorized Pandas/NumPy operation ← always prefer
2. `.map(dict)` ← for simple value lookups
3. `np.where` (binary) / `np.select` (multiple categories) ← for conditional columns
4. `apply(named_function)` ← last resort

**Never use `apply(lambda ...)`.**

```python
# ✓ Binary condition → np.where
df['category'] = np.where(df['value'] > 100, 'high', 'low')

# ✓ Multiple categories → np.select
conditions = [df['value'] > 200, df['value'] > 100, df['value'] > 50]
choices    = ['very high', 'high', 'medium']
df['category'] = np.select(conditions, choices, default='low')

# ✓ Simple lookup → .map(dict)  – more readable than np.select for mappings
mapping = {'Sales': 'Revenue', 'IT': 'Technology', 'HR': 'People'}
df['dept_group'] = df['department'].map(mapping)

# ✓ String operations → .str accessor (vectorized, not apply)
df['has_gmbh']   = df['company'].str.contains('GmbH', case=False, na=False)
df['domain']     = df['email'].str.extract(r'@(.+)$')
df['name_lower'] = df['name'].str.lower()

# ✓ Row combination → direct vectorized operation
df['description'] = df['name'] + ' from ' + df['department']

# ✓ Complex logic that cannot be vectorized → apply with named function
df['col'] = df['col'].apply(my_function)

# ✗ Never
df['category']   = df['value'].apply(lambda x: 'high' if x > 100 else 'low')
df['description'] = df.apply(lambda row: ..., axis=1)
df['has_gmbh']   = df['company'].apply(lambda x: 'GmbH' in str(x))  # use .str instead
```

**Why:**  
Pandas and NumPy operations run on compiled C/Fortran code under the hood and operate on entire arrays at once. `apply()` is essentially a Python `for` loop in disguise – it processes one row or value at a time and carries the full overhead of the Python interpreter for each iteration. On large datasets this difference is not marginal: vectorized operations are commonly **10x–100x faster** than `apply()`. Beyond performance, vectorized code expresses *what* you want rather than *how* to compute it row by row, which makes the intent immediately clear to any reader.

The `.str` accessor is a particularly common blind spot: many people reach for `apply()` when working with string columns because it feels natural, but every string method available in plain Python (`contains`, `extract`, `split`, `lower`, `replace`, etc.) has a vectorized equivalent under `Series.str`. Similarly, `.map(dict)` is the idiomatic and readable way to do simple value substitutions – it is more expressive than `np.select` with parallel lists when you just need a lookup table.

---

## 2. DataFrame Safety

**Rule:** Always call `.copy()` on slices. Never overwrite original DataFrames.

```python
# ✓ Correct
subset = df[df['department'] == 'IT'].copy()
subset['bonus'] = subset['salary'] * 0.1

# ✗ Wrong
subset = df[df['department'] == 'IT']
subset['bonus'] = subset['salary'] * 0.1   # SettingWithCopyWarning
```

**Why:**  
When you slice a DataFrame, Pandas may return either a *view* (a reference to the original memory) or a *copy* (independent memory), depending on the operation. This behavior is not guaranteed and can change between Pandas versions. If you receive a view and then modify it, you may silently modify the original DataFrame – a bug that is extremely difficult to trace because no error is raised, the data just changes unexpectedly. The `SettingWithCopyWarning` is Pandas telling you it cannot guarantee which one you have.

**Note on Pandas 3.0:** Copy-on-Write (CoW) became the default behavior in Pandas 3.0, which means views and copies now behave consistently and the warning disappears. However, explicitly calling `.copy()` is still the recommended practice: it signals *intent* to the reader ("I want an independent DataFrame here"), keeps the code backwards-compatible with earlier Pandas versions, and prevents subtle bugs in mixed-version environments. Do not drop `.copy()` just because your Pandas version no longer warns you.

---

## 3. Sorting

**Rule:** Use `sort_values()` instead of `sorted(key=lambda ...)`. Use `nlargest()` / `nsmallest()` when you only need the top or bottom n rows.

```python
# ✓ General sorting
df.sort_values('salary', ascending=False)
df.sort_values(['department', 'salary'], ascending=[True, False])

# ✓ Top/bottom n → nlargest / nsmallest (more readable and faster than sort + head)
df.nlargest(10, 'salary')
df.nsmallest(5, 'error_rate')

# ✗ Wrong
sorted(records, key=lambda x: x['salary'])
df.sort_values('salary', ascending=False).head(10)   # works, but nlargest is cleaner
```

**Why:**  
`sorted()` with `key=lambda` is a Python built-in designed for plain lists. In a Data Science context your data almost always lives in a DataFrame already. `sort_values()` operates directly on the DataFrame in vectorized C code, returns a DataFrame you can immediately continue working with, and supports multi-column sorting with per-column direction in a single readable call.

`nlargest()` and `nsmallest()` go one step further: internally they use a partial sort (heap-based algorithm) that only needs to find the top n elements rather than sorting the entire dataset. On a DataFrame with millions of rows, `nlargest(10, 'salary')` is significantly faster than `sort_values('salary').head(10)` and expresses the intent more directly – you want the largest values, not a sorted table you happen to truncate.

---

## 4. Config Dicts & Hyperparameters

**Rule:** Collect hyperparameters in dicts and unpack with `**`. Use lists of configs for experiment comparison.

```python
# ✓ Basic usage
rf_config = {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
model = RandomForestClassifier(**rf_config)

# ✓ Comparing multiple configurations
configs = [
    {'n_estimators': 50,  'max_depth': 3,  'random_state': 42},
    {'n_estimators': 100, 'max_depth': 5,  'random_state': 42},
    {'n_estimators': 200, 'max_depth': 10, 'random_state': 42},
]

for config in configs:
    model = RandomForestClassifier(**config)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Config {config} → Score: {score:.2%}")

# ✗ Wrong
model = RandomForestClassifier(100, 5, 42)   # What do these numbers mean?
```

**Why:**  
Positional arguments are unreadable without consulting the documentation – `RandomForestClassifier(100, 5, 42)` tells you nothing about what those values control. Config dicts make every parameter self-documenting at the point of use.

More importantly, config dicts are the natural unit for experiment tracking: you can log them directly to MLflow or Weights & Biases, version them in YAML files, and iterate over a list of configs for systematic comparison – as shown above. This pattern scales naturally from manual experimentation to automated hyperparameter search without restructuring your code. The `**` unpacking operator is the idiomatic Python bridge between a dict and keyword arguments, and it works identically whether you have 2 or 20 parameters.

---

## 5. Reproducibility

**Rule:** Set `random_state=42` on everything that uses randomness. Use `np.random.default_rng(seed=42)` instead of the legacy `np.random.seed()`. Set `PYTHONHASHSEED=0` for full pipeline reproducibility.

```python
# ✓ Correct
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
rng = np.random.default_rng(seed=42)

# For deep learning
# torch.manual_seed(42)        # PyTorch
# tf.random.set_seed(42)       # TensorFlow

# ✗ Wrong
np.random.seed(42)   # legacy global state, avoid
```

```bash
# Set before running your script for full reproducibility
PYTHONHASHSEED=0 python train.py
```

**Why:**  
Machine learning experiments involve randomness at multiple levels: train/test splits, model initialization, bootstrapping in ensemble methods, dropout in neural networks, and more. Without fixed seeds, re-running the same code produces different results every time, making it impossible to determine whether a performance change is due to a code change or random variation. `random_state=42` is the community convention.

`np.random.default_rng()` is preferred over the legacy `np.random.seed()` because it creates an *isolated* random number generator object rather than setting global state. Global state is dangerous in complex pipelines where multiple libraries may interact with the same random stream in unpredictable ways.

`PYTHONHASHSEED=0` addresses a less obvious source of non-reproducibility: Python randomizes the hash order of sets and dicts by default since Python 3.3. If your feature pipeline iterates over a set or dict (e.g. when building a vocabulary or encoding categories), the order of features can differ between runs, producing different model inputs even with identical data. Setting `PYTHONHASHSEED=0` disables this randomization. For deep learning projects, the framework-specific seed calls (`torch.manual_seed()`, `tf.random.set_seed()`) are additionally required.

---

## 6. Data Leakage Prevention

**Rule:** Always wrap preprocessing steps in a scikit-learn `Pipeline`. Never call `fit()` on the full dataset before the train/test split. Be aware of target leakage as a separate, unrelated risk.

```python
# ✓ Correct – Pipeline prevents preprocessing leakage
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # imputation inside pipeline (see §7)
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression()),
])
pipeline.fit(X_train, y_train)   # all steps only learn from X_train

# ✗ Wrong – preprocessing leakage
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # learns statistics from test data
X_train, X_test = train_test_split(X_scaled, ...)
```

**Why:**  
Data leakage is one of the most common and damaging mistakes in applied ML. It produces models that appear to perform well in evaluation but fail in production.

**Preprocessing leakage** occurs when a transformer (e.g. `StandardScaler`, `SimpleImputer`) is fitted on the full dataset before splitting. The transformer then "knows" statistics from the test set – information the model should not have access to. This inflates evaluation metrics and leads to overly optimistic performance estimates. A `Pipeline` makes this structurally impossible: `pipeline.fit(X_train, y_train)` automatically applies `fit_transform` on training data and `transform` (without fitting) on test data, exactly as it would behave in production.

**Target leakage** is a separate and equally dangerous category that a Pipeline cannot prevent. It occurs when a feature is causally downstream of the target – for example, using `claim_processed = True` as a feature to predict `fraud`, when claims are only processed *after* fraud is confirmed. The feature encodes the answer. Target leakage must be caught during exploratory data analysis through domain knowledge and correlation analysis, not through code patterns.

---

## 7. Null Values

**Rule:** Define and document a null strategy before training. Use `fillna()` with a justified value. Only use `dropna()` when rows are genuinely unusable. Preserve the missingness signal with an indicator column. Perform imputation inside the Pipeline (not before it).

```python
# ✓ Imputation inside Pipeline – consistent with §6
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model',   RandomForestClassifier(random_state=42)),
])

# ✓ Preserve missingness as a signal before imputation
df['salary_was_missing'] = df['salary'].isna().astype(int)
df['salary'] = df['salary'].fillna(df['salary'].median())

# ✓ Drop only rows without a target – they are genuinely unusable
df = df.dropna(subset=['target'])

# ✗ Wrong
df = df.dropna()                      # silently discards potentially valuable data
scaler.fit(df[['salary']].fillna(0))  # imputation outside Pipeline → leakage risk
```

**Why:**  
Null values in real-world data are rarely random. A missing salary field might indicate a contractor rather than a full-time employee. A missing sensor reading might indicate equipment failure. Blindly dropping all rows with nulls can introduce systematic bias – if a certain group is more likely to have missing values, dropping them skews every downstream analysis and model.

The missingness itself is often predictive. The pattern `df['col_was_missing'] = df['col'].isna().astype(int)` preserves this signal as an explicit binary feature before imputing, so the model can learn from both the imputed value and the fact that the original was absent.

Critically, imputation must happen **inside the Pipeline** (via `SimpleImputer`), not before it. Imputing on the full dataset before the train/test split is a form of preprocessing leakage (see §6): the imputer learns fill values from test data, which contaminates the evaluation. Placing `SimpleImputer` as the first step in a Pipeline ensures it only learns from training data.

---

## 8. Evaluation

**Rule:** Never report accuracy alone. Use context-appropriate metrics. Use `cross_val_score` for model comparison. Use a held-out test set or nested CV for final performance estimation. Always report `mean ± std`. Include regression metrics when applicable.

```python
# ✓ Classification – context-appropriate metric with variance
scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"F1: {scores.mean():.2%} ± {scores.std():.2%}")

# ✓ Regression metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2   = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")

# ✓ Final performance: separate holdout set, not the same CV folds used for tuning
X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(
    X, y, test_size=0.15, random_state=42
)
# tune and select model on X_train_val via CV, then evaluate once on X_holdout

# ✗ Wrong
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")  # single split, wrong metric
```

**Why:**  
**On accuracy:** Accuracy is the fraction of correct predictions – a number that is actively misleading on imbalanced datasets. A model that predicts "not fraud" for every transaction in a dataset where 99% of transactions are legitimate achieves 99% accuracy while being completely useless. F1 score, precision, and recall measure how well the model handles the minority class, which is almost always the class you actually care about.

**On regression metrics:** MAE measures average absolute error in the original unit, making it directly interpretable. RMSE penalizes large errors more heavily due to squaring, making it sensitive to outliers – useful when large errors are particularly costly. R² expresses the proportion of variance explained by the model, providing a scale-independent quality measure. Reporting all three gives a complete picture.

**On cross-validation vs. holdout:** `cross_val_score` on the full dataset is appropriate for *comparing models* because it uses all available data and reduces variance. However, if you use CV scores to select hyperparameters and then report the same CV score as your final performance estimate, you introduce optimistic bias – the model has indirectly been tuned on all folds. For a trustworthy final estimate, hold out a separate test set before any modeling decisions, tune on the remaining data using CV, and evaluate the final model exactly once on the holdout.

**On variance:** A model with `F1: 85% ± 1%` is meaningfully different from one with `F1: 85% ± 8%` – both have the same mean but the second is unreliable. Always report standard deviation alongside the mean.
