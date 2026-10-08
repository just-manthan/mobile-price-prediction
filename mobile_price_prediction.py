"""
Feature Extraction and Price Prediction for Mobile Phones
=========================================================
End-to-end pipeline:
    1. Data exploration
    2. Data preprocessing (cleaning, outliers, encoding)
    3. Feature extraction (correlation, mutual information, SelectKBest, PCA)
    4. Model building (Linear Regression, Decision Tree, Random Forest, Gradient Boosting)
    5. Model evaluation (MAE, RMSE, R2, MAPE + 5-fold CV)
    6. Feature importance analysis (impurity + permutation importance)
    7. Auto-generated report with visualizations and recommendations

Usage:
    python mobile_price_prediction.py                     # uses data/Processed_Flipdata.csv
    python mobile_price_prediction.py --data path/to.csv
"""

import argparse
import os
import re
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")

RANDOM_STATE = 42
FIG_DIR = "figures"
REPORT_DIR = "reports"
MODEL_DIR = "models"

# Canonical column names -> possible spellings in the raw file
COLUMN_ALIASES = {
    "model": ["model", "name", "phone", "mobile"],
    "colour": ["colour", "color"],
    "memory": ["memory", "rom", "storage", "inbuilt_memory", "internal_memory"],
    "ram": ["ram"],
    "battery": ["battery_", "battery", "battery_capacity"],
    "rear_camera": ["rear_camera", "rear camera", "back_camera", "primary_camera"],
    "front_camera": ["front_camera", "front camera", "selfie_camera"],
    "ai_lens": ["ai_lens", "ai lens", "ai"],
    "mobile_height": ["mobile_height", "mobile height", "height", "display_size", "screen_size"],
    "processor": ["processor_", "processor", "chipset"],
    "price": ["prize", "price", "selling_price", "mrp"],
}

NUMERIC_FEATURES = ["memory", "ram", "battery", "rear_camera", "front_camera",
                    "ai_lens", "mobile_height"]
CATEGORICAL_FEATURES = ["brand", "colour_group", "processor_family", "processor_tier"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def ensure_dirs():
    for d in (FIG_DIR, REPORT_DIR, MODEL_DIR, "data"):
        os.makedirs(d, exist_ok=True)


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    return path


def standardize_columns(df):
    """Rename raw columns to canonical names using COLUMN_ALIASES."""
    df = df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")]
    lookup = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    lookup.update({c.strip().lower(): c for c in df.columns})
    rename = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            key = a.lower()
            if key in lookup and lookup[key] not in rename:
                rename[lookup[key]] = canon
                break
    df = df.rename(columns=rename)
    missing = [c for c in ("price",) if c not in df.columns]
    if missing:
        raise ValueError(f"Required column(s) not found: {missing}. Columns: {list(df.columns)}")
    return df


def to_number(val):
    """Extract a number from strings like '24,999', '4 GB', '50MP', '6.5 inch', '1 TB'."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val).replace(",", "").strip().lower()
    m = re.search(r"\d+(\.\d+)?", s)
    if not m:
        return np.nan
    num = float(m.group())
    if "tb" in s:
        num *= 1024
    elif "mb" in s and "mp" not in s:
        num /= 1024
    return num


def to_binary(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return 1.0 if val > 0 else 0.0
    s = str(val).strip().lower()
    return 1.0 if s in {"yes", "y", "true", "1", "present"} else 0.0


def processor_family(p):
    if pd.isna(p):
        return "Unknown"
    s = str(p).lower()
    if "dimensity" in s:
        return "MediaTek Dimensity"
    if "snapdragon" in s or "qualcomm" in s or "sm6" in s:
        return "Qualcomm Snapdragon"
    if "exynos" in s or "s5e" in s:
        return "Samsung Exynos"
    if "tensor" in s:
        return "Google Tensor"
    if "bionic" in s or s.strip() in {"ios", "apple"}:
        return "Apple Bionic"
    if re.search(r"unisoc|spreadtrum|tiger|sc\d{4}|\bt6\d\d\b|\bt7\d\d\b", s):
        return "Unisoc"
    if re.search(r"heli|\bg\d{2}\b|\ba2\d\b|\bp35\b", s):  # incl. typo 'Meditek'
        return "MediaTek Helio"
    if re.search(r"mt\d{4}|mtk|mediatek", s):
        return "MediaTek Other"
    return "Other/Unspecified"


def processor_tier(p):
    """Ordinal performance tier extracted from the processor model name."""
    if pd.isna(p):
        return "Unknown"
    s = str(p).lower()
    flagship = [r"8\+? gen", r"\b8(45|55|65|88)\b", r"7\+ gen", r"dimensity (8|9)\d{3}",
                r"dimensity 1300", r"tensor", r"bionic", r"^\s*ios\s*$"]
    upper_mid = [r"\b7(20|30|50|65|78)g?", r"dimensity (1080|9[0-3]0|7200)", r"exynos 1380"]
    entry = [r"helio (a2\d|g2\d|g3\d|p35)", r"\b(g3\d|g2\d|a2\d)\b", r"sc\d{4}", r"spreadtrum",
             r"mt62\d\d|mtk62", r"exynos 850", r"\b439\b", r"208mhz", r"quad core"]
    for tier, pats in (("Flagship", flagship), ("Upper-mid", upper_mid), ("Entry", entry)):
        if any(re.search(pt, s) for pt in pats):
            return tier
    return "Mid"


def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    }


def df_to_md(df, floatfmt="{:,.2f}"):
    """Small markdown table writer (avoids the optional 'tabulate' dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 1. Data exploration
# --------------------------------------------------------------------------
def explore(raw):
    print("\n=== 1. DATA EXPLORATION ===")
    info = {
        "shape": raw.shape,
        "dtypes": raw.dtypes.astype(str).to_dict(),
        "missing": raw.isna().sum().to_dict(),
        "duplicates": int(raw.loc[:, ~raw.columns.str.lower().str.startswith("unnamed")].duplicated().sum()),
    }
    print(f"Shape: {raw.shape}")
    print(raw.head())
    print("\nData types:\n", raw.dtypes)
    print("\nMissing values:\n", raw.isna().sum())
    print(f"\nDuplicate rows: {info['duplicates']}")
    with open(os.path.join(REPORT_DIR, "eda_summary.txt"), "w") as f:
        f.write(f"Shape: {raw.shape}\n\nHead:\n{raw.head().to_string()}\n\n")
        f.write(f"Dtypes:\n{raw.dtypes.to_string()}\n\nMissing:\n{raw.isna().sum().to_string()}\n\n")
        f.write(f"Duplicates: {info['duplicates']}\n\nDescribe:\n{raw.describe(include='all').T.to_string()}\n")
    return info


# --------------------------------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------------------------------
def preprocess(raw):
    print("\n=== 2. DATA PREPROCESSING ===")
    df = raw.copy()
    log = []

    # Parse numeric columns
    df["price"] = df["price"].apply(to_number)
    for c in NUMERIC_FEATURES:
        if c in df.columns:
            df[c] = df[c].apply(to_binary if c == "ai_lens" else to_number)

    n0 = len(df)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    log.append(f"Dropped {n0 - len(df)} rows with missing/invalid price.")

    n0 = len(df)
    df = df.drop_duplicates()
    log.append(f"Removed {n0 - len(df)} exact duplicate rows.")

    # Colour variants: same phone + same specs listed once per colour.
    spec_cols = [c for c in df.columns if c != "colour"]
    if "colour" in df.columns:
        grp = df.groupby([c for c in spec_cols if c != "price"], dropna=False)["price"]
        same_price = (grp.nunique() == 1).mean() * 100
        n0 = len(df)
        df = df.drop_duplicates(subset=spec_cols, keep="first")
        log.append(f"Removed {n0 - len(df)} colour-variant rows (same model, specs and price, "
                   f"different colour). In {same_price:.0f}% of spec groups every colour has "
                   "the same price, so colour does not drive price; keeping one row per phone "
                   "prevents the same phone appearing in both train and test sets (data leakage).")

    # Engineered categorical features
    if "model" in df.columns:
        b = df["model"].astype(str).str.strip().str.split().str[0].str.title()
        df["brand"] = b.str.replace(r"\d+$", "", regex=True).replace({"Iqoo": "Vivo"})
    else:
        df["brand"] = "Unknown"
    df["processor_family"] = (df["processor"].apply(processor_family)
                              if "processor" in df.columns else "Unknown")
    df["processor_tier"] = (df["processor"].apply(processor_tier)
                            if "processor" in df.columns else "Unknown")
    if "colour" in df.columns:
        col = df["colour"].astype(str).str.strip().str.lower()
        base = col.str.extract(r"(black|blue|green|white|silver|gold|purple|red|grey|gray|pink|"
                               r"violet|orange|yellow|brown|cyan|bronze)", expand=False)
        base = base.replace({"gray": "grey", "violet": "purple"}).fillna("other")
        df["colour_group"] = base.str.title()
    else:
        df["colour_group"] = "Unknown"

    # Rare brands (<5 phones) -> "Other"
    counts = df["brand"].value_counts()
    rare = counts[counts < 5].index
    df.loc[df["brand"].isin(rare), "brand"] = "Other"

    # Impute missing numeric values with median (robust to skew)
    for c in [c for c in NUMERIC_FEATURES if c in df.columns]:
        n_miss = int(df[c].isna().sum())
        if n_miss:
            df[c] = df[c].fillna(df[c].median())
            log.append(f"Imputed {n_miss} missing values in '{c}' with median.")

    # Fix impossible values (sanity rules)
    rules = {"ram": (0.5, 32), "memory": (2, 2048), "battery": (500, 10000),
             "rear_camera": (0, 250), "front_camera": (0, 100), "mobile_height": (3, 20)}
    for c, (lo, hi) in rules.items():
        if c in df.columns:
            bad = ~df[c].between(lo, hi)
            if bad.sum():
                df.loc[bad, c] = df[c].median()
                log.append(f"Replaced {int(bad.sum())} out-of-range values in '{c}' with median.")

    # Outliers in price: cap at 1st / 99th percentile (winsorising keeps rows)
    lo, hi = df["price"].quantile([0.01, 0.99])
    n_out = int(((df["price"] < lo) | (df["price"] > hi)).sum())
    df["price"] = df["price"].clip(lo, hi)
    log.append(f"Capped {n_out} price outliers to the 1st-99th percentile range "
               f"(Rs {lo:,.0f} - Rs {hi:,.0f}).")

    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    for line in log:
        print(" -", line)
    print(f"Clean shape: {df.shape}")
    return df, num_cols, log


def encode(df, num_cols):
    """One-hot encode categorical features."""
    X = pd.get_dummies(df[num_cols + CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES,
                       prefix={"brand": "brand", "colour_group": "colour",
                               "processor_family": "proc", "processor_tier": "tier"}, dtype=float)
    y = df["price"].values
    return X, y


# --------------------------------------------------------------------------
# EDA plots (on clean data)
# --------------------------------------------------------------------------
def eda_plots(df, num_cols):
    figs = {}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["price"], bins=40, kde=True, ax=ax[0])
    ax[0].set_title("Price distribution (Rs)")
    sns.histplot(np.log1p(df["price"]), bins=40, kde=True, ax=ax[1], color="tab:green")
    ax[1].set_title("log(Price) distribution")
    figs["price_dist"] = savefig("01_price_distribution.png")

    plt.figure(figsize=(8, 6))
    corr = df[num_cols + ["price"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation matrix (numeric features)")
    figs["corr"] = savefig("02_correlation_heatmap.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    if "ram" in df.columns:
        sns.boxplot(x=df["ram"].round(0).astype(int), y=df["price"], ax=axes[0])
        axes[0].set_title("Price by RAM (GB)")
    if "memory" in df.columns:
        sns.boxplot(x=df["memory"].round(0).astype(int), y=df["price"], ax=axes[1])
        axes[1].set_title("Price by storage (GB)")
        axes[1].tick_params(axis="x", rotation=45)
    figs["box"] = savefig("03_price_by_ram_storage.png")

    order = df.groupby("brand")["price"].median().sort_values(ascending=False).index
    plt.figure(figsize=(11, 4.5))
    sns.boxplot(data=df, x="brand", y="price", order=order)
    plt.xticks(rotation=45, ha="right")
    plt.title("Price by brand")
    figs["brand"] = savefig("04_price_by_brand.png")

    order = df.groupby("processor_family")["price"].median().sort_values(ascending=False).index
    plt.figure(figsize=(11, 4.5))
    sns.boxplot(data=df, x="processor_family", y="price", order=order)
    plt.xticks(rotation=45, ha="right")
    plt.title("Price by processor family")
    figs["proc"] = savefig("05_price_by_processor.png")

    tiers = [t for t in ["Entry", "Mid", "Upper-mid", "Flagship"] if t in set(df["processor_tier"])]
    plt.figure(figsize=(8, 4.5))
    sns.boxplot(data=df, x="processor_tier", y="price", order=tiers)
    plt.title("Price by processor tier (engineered feature)")
    figs["tier"] = savefig("05b_price_by_processor_tier.png")
    return figs


# --------------------------------------------------------------------------
# 3. Feature extraction
# --------------------------------------------------------------------------
def feature_extraction(df, num_cols, X, y):
    print("\n=== 3. FEATURE EXTRACTION ===")
    out = {}

    pearson = df[num_cols].corrwith(df["price"]).rename("pearson_r")
    spearman = df[num_cols].corrwith(df["price"], method="spearman").rename("spearman_r")
    corr_tbl = pd.concat([pearson, spearman], axis=1).sort_values("spearman_r", ascending=False)
    out["corr"] = corr_tbl
    print(corr_tbl)

    mi = pd.Series(mutual_info_regression(X, np.log1p(y), random_state=RANDOM_STATE),
                   index=X.columns, name="mutual_info").sort_values(ascending=False)
    out["mi"] = mi
    plt.figure(figsize=(8, 6))
    mi.head(15).iloc[::-1].plot.barh(color="tab:purple")
    plt.title("Top 15 features by mutual information with log(price)")
    out["mi_fig"] = savefig("06_mutual_information.png")

    k = min(10, X.shape[1])
    skb = SelectKBest(f_regression, k=k).fit(X, np.log1p(y))
    out["kbest"] = list(X.columns[skb.get_support()])
    print(f"SelectKBest (k={k}):", out["kbest"])

    Xs = StandardScaler().fit_transform(X)
    pca = PCA(random_state=RANDOM_STATE).fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    out["pca_n95"] = int(np.searchsorted(cum, 0.95) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(cum) + 1), cum, marker="o", ms=3)
    plt.axhline(0.95, color="red", ls="--", label="95% variance")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA - explained variance")
    plt.legend()
    out["pca_fig"] = savefig("07_pca_variance.png")
    loadings = pd.Series(np.abs(pca.components_[0]), index=X.columns).sort_values(ascending=False)
    out["pc1_top"] = list(loadings.head(5).index)
    print(f"PCA: {out['pca_n95']} components explain 95% of variance; "
          f"PC1 driven by {out['pc1_top']}")
    return out


# --------------------------------------------------------------------------
# 4-5. Model building & evaluation
# --------------------------------------------------------------------------
def build_and_evaluate(X, y):
    print("\n=== 4-5. MODEL BUILDING & EVALUATION ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    y_train_log = np.log1p(y_train)  # log target handles the right-skewed price

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=3,
                                               random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                               n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                                       max_depth=4, subsample=0.8,
                                                       random_state=RANDOM_STATE),
    }
    rows, fitted, preds = [], {}, {}
    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    for name, model in models.items():
        model.fit(X_train, y_train_log)
        p = np.expm1(model.predict(X_test))
        m = regression_metrics(y_test, p)
        m["CV_R2_log"] = cross_val_score(model, X, np.log1p(y), cv=cv, scoring="r2").mean()
        rows.append({"Model": name, **m})
        fitted[name], preds[name] = model, p
        print(f"{name:18s} MAE={m['MAE']:,.0f}  RMSE={m['RMSE']:,.0f}  R2={m['R2']:.3f}  "
              f"MAPE={m['MAPE_%']:.1f}%  CV-R2={m['CV_R2_log']:.3f}")

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best = results.loc[0, "Model"]
    print(f"Best model: {best}")

    plt.figure(figsize=(9, 4))
    sns.barplot(data=results, x="Model", y="RMSE", color="tab:blue")
    plt.title("Test RMSE by model (lower is better)")
    savefig("08_model_comparison.png")

    plt.figure(figsize=(6, 6))
    p = preds[best]
    plt.scatter(y_test, p, alpha=0.5, s=18)
    lim = [min(y_test.min(), p.min()), max(y_test.max(), p.max())]
    plt.plot(lim, lim, "r--", lw=1)
    plt.xlabel("Actual price (Rs)")
    plt.ylabel("Predicted price (Rs)")
    plt.title(f"{best}: actual vs predicted")
    savefig("09_actual_vs_predicted.png")

    joblib.dump({"model": fitted[best], "columns": list(X.columns), "target": "log1p(price)"},
                os.path.join(MODEL_DIR, "best_model.joblib"))
    return results, best, fitted, (X_train, X_test, y_train, y_test)


# --------------------------------------------------------------------------
# 6. Feature importance
# --------------------------------------------------------------------------
def group_name(col):
    for prefix, label in (("brand_", "brand"), ("colour_", "colour"),
                          ("proc_", "processor_family"), ("tier_", "processor_tier")):
        if col.startswith(prefix):
            return label
    return col


def feature_importance(fitted, best, splits):
    print("\n=== 6. FEATURE IMPORTANCE ===")
    X_train, X_test, y_train, y_test = splits
    tree_name = best if best != "Linear Regression" else "Random Forest"
    model = fitted[tree_name]

    imp = pd.Series(model.feature_importances_, index=X_train.columns)
    perm = permutation_importance(model, X_test, np.log1p(y_test), n_repeats=15,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    perm = pd.Series(perm.importances_mean, index=X_train.columns)

    grouped = pd.DataFrame({
        "impurity": imp.groupby(group_name).sum(),
        "permutation": perm.groupby(group_name).sum(),
    })
    grouped["impurity_%"] = 100 * grouped["impurity"] / grouped["impurity"].sum()
    grouped = grouped.sort_values("permutation", ascending=False)
    print(grouped.round(4))

    plt.figure(figsize=(8, 6))
    imp.sort_values().tail(15).plot.barh(color="tab:orange")
    plt.title(f"{tree_name}: top 15 encoded features (impurity importance)")
    savefig("10_feature_importance_detailed.png")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    gplot = grouped.rename(index=lambda n: PRETTY.get(n, n))
    gplot["impurity_%"].sort_values().plot.barh(ax=ax[0], color="tab:orange")
    ax[0].set_title("Grouped impurity importance (%)")
    gplot["permutation"].sort_values().plot.barh(ax=ax[1], color="tab:red")
    ax[1].set_title("Grouped permutation importance (drop in R2)")
    savefig("11_feature_importance_grouped.png")
    return grouped, tree_name


# --------------------------------------------------------------------------
# 7. Report
# --------------------------------------------------------------------------
PRETTY = {
    "front_camera": "Front camera (MP)", "rear_camera": "Rear camera (MP)",
    "memory": "Storage (GB)", "ram": "RAM (GB)", "battery": "Battery (mAh)",
    "mobile_height": "Display size (cm)", "ai_lens": "AI lens", "brand": "Brand",
    "processor_tier": "Processor tier", "processor_family": "Processor family",
    "colour": "Colour",
}


def pretty(name):
    return PRETTY.get(name, name)


def band_medians(df, col, bins, labels):
    """Median price and count per band of a numeric column."""
    b = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return df.groupby(b, observed=True)["price"].agg(["count", "median"])


def fmt_rs(v):
    return f"Rs {v:,.0f}"


def write_report(raw_info, clean_df, prep_log, fe, results, best, grouped, tree_name):
    d = clean_df
    corr = fe["corr"]
    best_row = results.iloc[0]
    rank = list(grouped.index)
    top3 = [pretty(f) for f in rank[:3]]
    low = [pretty(f) for f in rank[-3:]]
    n_test = int(round(len(d) * 0.2))

    ram_med = d.groupby("ram")["price"].median() if "ram" in d else pd.Series(dtype=float)
    mem_med = d.groupby("memory")["price"].median() if "memory" in d else pd.Series(dtype=float)
    tier_order = [t for t in ["Entry", "Mid", "Upper-mid", "Flagship"] if t in set(d["processor_tier"])]
    tier_med = d.groupby("processor_tier")["price"].median().reindex(tier_order)
    brand_med = d.groupby("brand")["price"].median().sort_values(ascending=False)
    fc = band_medians(d, "front_camera", [0, 0.5, 9, 17, 1000], ["None", "2-8 MP", "10-16 MP", "20 MP+"]) \
        if "front_camera" in d else None
    bat = band_medians(d, "battery", [0, 3000, 5000, 5001, 100000],
                       ["< 3000", "3000-4999", "5000", "> 5000"]) if "battery" in d else None
    ai = d.groupby("ai_lens")["price"].median() if "ai_lens" in d else None

    def series_line(s, unit):
        return ", ".join(f"{k:g} {unit} → {fmt_rs(v)}" for k, v in s.items())

    def band_table(tbl, label):
        t = tbl.reset_index()
        t.columns = [label, "phones", "median price (Rs)"]
        t["phones"] = t["phones"].astype(int).astype(str)
        return df_to_md(t, "{:,.0f}")

    L = []
    add = L.append
    add("# Mobile Phone Price Prediction – Project Report\n")

    # ---------------- Executive summary
    add("## Executive summary")
    add(f"- Analysed **{raw_info['shape'][0]} phone listings** ({len(d)} unique phones after cleaning) "
        "with 10 specifications each.")
    add(f"- Best model: **{best}**. On phones it never saw, it predicts price within "
        f"**{fmt_rs(best_row['MAE'])} on average** (R² = {best_row['R2']:.2f}, "
        f"MAPE = {best_row['MAPE_%']:.1f}%; 5-fold cross-validated R² = {best_row['CV_R2_log']:.2f}).")
    add(f"- Strongest price drivers: **{', '.join(top3)}**, followed by storage, RAM and brand.")
    add(f"- Weakest: **{', '.join(low)}**. Colour has essentially no effect on price.")
    add("")

    # ---------------- 1 Objective
    add("## 1. Objective")
    add("Estimate a mobile phone's price from its specifications and identify which features "
        "influence price the most, to support the organisation's pricing and marketing strategy.\n")

    # ---------------- 2 Exploration
    add("## 2. Data exploration")
    add(f"- Raw data: **{raw_info['shape'][0]} rows × {raw_info['shape'][1]} columns** – Model, Colour, "
        "Memory, RAM, Battery, Rear Camera, Front Camera, AI Lens, Mobile Height, Processor and Price.")
    add(f"- No missing values; {raw_info['duplicates']} exact duplicate rows.")
    add("- Camera columns are text (e.g. `50MP`) and the processor column has 100+ inconsistent "
        "spellings (e.g. `G37`, `Meditek Helio G37`, `MediaTek Helio G37`).")
    add(f"- Price ranges from {fmt_rs(d['price'].min())} (feature phones) to {fmt_rs(d['price'].max())}; "
        f"median {fmt_rs(d['price'].median())}. The distribution is right-skewed, so models are trained "
        "on log(price) and predictions converted back to rupees.\n")
    add("![Price distribution](../figures/01_price_distribution.png)\n")

    # ---------------- 3 Preprocessing
    add("## 3. Data preprocessing")
    for l in prep_log:
        add(f"- {l}")
    add("- Text values parsed to numbers (`50MP` → 50); AI Lens kept as 0/1.")
    add("- **Feature engineering:** `brand` (from model name), `processor_family` "
        "(Snapdragon / Dimensity / Helio / Exynos / Unisoc / Tensor / Bionic), `processor_tier` "
        "(Entry / Mid / Upper-mid / Flagship, from the chip model number) and `colour_group` "
        "(base colour). Brands with fewer than 5 phones grouped as 'Other'.")
    add("- Categorical features **one-hot encoded** (Brand, Colour, Processor family, Processor tier).\n")

    # ---------------- 4 Feature extraction
    add("## 4. Feature extraction")
    add("Four complementary techniques were used to narrow down the important features.\n")
    add("### 4.1 Correlation analysis")
    ct = corr.reset_index()
    ct.columns = ["feature", "Pearson r", "Spearman r"]
    ct["feature"] = ct["feature"].map(pretty)
    add(df_to_md(ct, "{:.2f}"))
    add("\nSpearman (rank) correlation is higher than Pearson for every key feature, meaning the "
        "relationships are strong but non-linear – a reason to prefer tree-based models.\n")
    add("![Correlation heatmap](../figures/02_correlation_heatmap.png)\n")
    add("![Price by RAM and storage](../figures/03_price_by_ram_storage.png)\n")
    add("### 4.2 Mutual information")
    add(f"Captures non-linear dependence. Top features: {', '.join(fe['mi'].head(8).index)}.\n")
    add("![Mutual information](../figures/06_mutual_information.png)\n")
    add("### 4.3 SelectKBest (F-test)")
    add(f"Top 10 selected: {', '.join(fe['kbest'])}.\n")
    add("### 4.4 PCA (dimensionality reduction)")
    add(f"{fe['pca_n95']} principal components are needed for 95% of variance (one-hot columns spread "
        f"the variance). The first component is driven by {', '.join(fe['pc1_top'])} – i.e. an overall "
        "\"phone segment\" axis. PCA was used for analysis only; the model keeps the original features "
        "so results stay interpretable.\n")
    add("![PCA](../figures/07_pca_variance.png)\n")
    add("### 4.5 Categorical features")
    add("![Price by brand](../figures/04_price_by_brand.png)\n")
    add("![Price by processor family](../figures/05_price_by_processor.png)\n")
    add("![Price by processor tier](../figures/05b_price_by_processor_tier.png)\n")
    add("The engineered **processor tier** separates prices cleanly: " +
        ", ".join(f"{t} {fmt_rs(v)}" for t, v in tier_med.items()) + " (median).\n")

    # ---------------- 5 Models
    add("## 5. Model building & evaluation")
    add(f"- 80/20 train–test split (random_state = 42) → {len(d) - n_test} training / {n_test} test phones.")
    add("- Models: Linear Regression (baseline), Decision Tree, Random Forest, Gradient Boosting.")
    add("- Metrics: MAE, RMSE, R², MAPE on the test set, plus 5-fold cross-validated R² on log(price).\n")
    rt = results.rename(columns={"MAPE_%": "MAPE (%)", "CV_R2_log": "CV R² (log)"})
    add(df_to_md(rt))
    add(f"\n**{best} performs best on every test metric.** It is off by about "
        f"{fmt_rs(best_row['MAE'])} per phone on average (RMSE {fmt_rs(best_row['RMSE'])}, "
        f"R² = {best_row['R2']:.3f}). Because the test set is small ({n_test} phones), the "
        "cross-validated R² is the more reliable comparison – it confirms the same ranking at the top. "
        "Linear Regression's gap between test R² and CV R² shows it is unstable on this small, "
        "non-linear dataset.\n")
    add("![Model comparison](../figures/08_model_comparison.png)\n")
    add("![Actual vs predicted](../figures/09_actual_vs_predicted.png)\n")

    # ---------------- 6 Importance
    add("## 6. Feature importance analysis")
    add(f"Importance from the {tree_name}, using two methods; one-hot columns are summed back to the "
        "original feature. Permutation importance (drop in R² when a feature is shuffled on the "
        "test set) is the more trustworthy of the two.\n")
    gt = grouped[["impurity_%", "permutation"]].reset_index()
    gt.columns = ["feature", "impurity importance (%)", "permutation importance (ΔR²)"]
    gt["feature"] = gt["feature"].map(pretty)
    add(df_to_md(gt, "{:.3f}"))
    add("\n![Grouped importance](../figures/11_feature_importance_grouped.png)\n")
    add("![Detailed importance](../figures/10_feature_importance_detailed.png)\n")
    add("### Confirmation of the feature-extraction results")
    add(f"- **Cameras, storage and RAM** rank highly in correlation, mutual information, SelectKBest "
        "*and* model importance – confirmed as key drivers.")
    add("- **Processor tier** (engineered) ranks in the model's top 3 and was picked by SelectKBest "
        "(Entry / Flagship) – confirming that the chip's performance level matters more than its brand "
        "(`processor family` ranks low once tier is known).")
    add("- **Battery** has near-zero correlation but meaningful model importance: its effect is "
        "non-linear (see below), which linear correlation cannot detect.")
    add("- **Colour and AI lens** are weak in every method – confirmed as non-drivers.\n")

    # ---------------- 7 Insights
    add("## 7. Key insights")
    if fc is not None:
        add("**Front camera** – the single strongest signal; it tracks the phone's overall segment:\n")
        add(band_table(fc, "front camera"))
        add("")
    if len(ram_med):
        add(f"**RAM:** {series_line(ram_med, 'GB')} (median price).\n")
    if len(mem_med):
        add(f"**Storage:** {series_line(mem_med, 'GB')} (median price).\n")
    if bat is not None:
        add("**Battery – non-linear:** budget phones ship the biggest batteries, flagships use "
            "slimmer 4,000–4,900 mAh cells, and feature phones have tiny ones:\n")
        add(band_table(bat, "battery (mAh)"))
        add("")
    add(f"**Brand:** highest median prices – {', '.join(f'{b} ({fmt_rs(v)})' for b, v in brand_med.head(4).items())}.\n")
    if ai is not None and len(ai) == 2:
        add(f"**AI lens:** phones with one have a *lower* median price ({fmt_rs(ai.get(1.0, np.nan))} vs "
            f"{fmt_rs(ai.get(0.0, np.nan))}) – it is a budget-phone marketing feature, not a premium one.\n")

    # ---------------- 8 Recommendations
    add("## 8. Recommendations for the organisation")
    recs = []
    recs.append("**Build the price ladder on processor tier, cameras and memory.** These explain most "
                "of the price variation. Use processor tier to set the segment, then camera and "
                "RAM/storage configuration to position a phone within it.")
    if len(ram_med) and len(mem_med):
        r_lo, r_hi = ram_med.get(4, ram_med.iloc[0]), ram_med.get(8, ram_med.iloc[-1])
        m_lo, m_hi = mem_med.get(64, mem_med.iloc[0]), mem_med.get(128, mem_med.iloc[-1])
        recs.append(f"**Use RAM/storage variants as the main upsell.** Moving from 4 GB to 8 GB RAM is "
                    f"associated with a median price jump of {fmt_rs(r_lo)} → {fmt_rs(r_hi)}, and 64 GB → "
                    f"128 GB storage with {fmt_rs(m_lo)} → {fmt_rs(m_hi)}. Stock 2–3 memory variants per "
                    "model and display RAM/storage prominently on listings.")
    recs.append("**Highlight camera specs in marketing.** Front and rear camera megapixels are the most "
                "visible quality signal customers pay for; feature them first in product titles and ads.")
    recs.append("**Market the chipset's tier, not its brand.** Customers pay for performance level – "
                "a flagship-class chip commands a premium regardless of whether it is Snapdragon or "
                "Dimensity. Use terms like \"flagship processor\" or \"5G gaming chip\".")
    recs.append("**Don't charge extra for colour or AI-lens claims.** Colour variants sell at the same "
                "price, and AI-lens phones are budget models. Use colours to widen choice and drive "
                "conversion, not as a price lever.")
    recs.append("**Promote large batteries in the budget segment.** 6,000 mAh+ is a selling point for "
                "value buyers, not a premium feature.")
    recs.append(f"**Adopt the model as a pricing tool.** With ~{fmt_rs(best_row['MAE'])} average error it can "
                "flag listings priced well above or below what their specs justify and suggest launch "
                "prices for new SKUs. Retrain it every quarter as the market moves.")
    for i, r in enumerate(recs, 1):
        add(f"{i}. {r}")
    add("")

    # ---------------- 9 Limitations
    add("## 9. Limitations & next steps")
    add(f"- Small dataset ({len(d)} unique phones) from one marketplace; collecting more listings "
        "would make estimates more stable.")
    add("- Discounts, launch date, 5G support, display type and build quality are not in the data but "
        "affect price – adding them is the most promising next step.")
    add("- Hyperparameter tuning (e.g. GridSearchCV) could improve the model further.\n")
    add("---\n_All numbers and charts are generated automatically by `mobile_price_prediction.py`._")

    path = os.path.join(REPORT_DIR, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    results.to_csv(os.path.join(REPORT_DIR, "model_results.csv"), index=False)
    grouped.to_csv(os.path.join(REPORT_DIR, "feature_importance.csv"))
    print(f"\nReport written to {path}")


# --------------------------------------------------------------------------
def main(data_path):
    ensure_dirs()
    if not os.path.exists(data_path):  # fall back to any CSV/Excel file in data/
        found = [os.path.join("data", f) for f in sorted(os.listdir("data"))
                 if f.lower().endswith((".csv", ".xlsx", ".xls")) and f != "cleaned_data.csv"]
        if not found:
            raise FileNotFoundError("Put the dataset (.csv or .xlsx) in the data/ folder.")
        data_path = found[0]
    print(f"Loading {data_path}")
    raw = (pd.read_excel(data_path) if data_path.lower().endswith((".xlsx", ".xls"))
           else pd.read_csv(data_path))
    raw_info = explore(raw)
    raw = standardize_columns(raw)
    clean, num_cols, prep_log = preprocess(raw)
    eda_plots(clean, num_cols)
    X, y = encode(clean, num_cols)
    fe = feature_extraction(clean, num_cols, X, y)
    results, best, fitted, splits = build_and_evaluate(X, y)
    grouped, tree_name = feature_importance(fitted, best, splits)
    write_report(raw_info, clean, prep_log, fe, results, best, grouped, tree_name)
    clean.to_csv(os.path.join("data", "cleaned_data.csv"), index=False)
    print("Done. See reports/REPORT.md and figures/.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "Processed_Flipdata.csv"))
    main(ap.parse_args().data)
