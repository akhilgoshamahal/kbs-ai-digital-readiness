"""Reproducible knowledge-based AI experiment for digital-readiness decisions.

One script downloads official public data, constructs a country-year panel,
runs grouped out-of-country validation, evaluates knowledge-rule integration,
and creates exactly two paper figures and two paper tables.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sklearn
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
EIS_URL = (
    "https://research-and-innovation.ec.europa.eu/document/download/"
    "9c0d1397-f01c-42cd-9571-400475d67404_en"
    "?filename=ec_rtd_eis-2025-neighborhood-database.xlsx"
)
EUROSTAT_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "isoc_e_dii?lang=en"
)
RANDOM_SEED = 42
CLASS_NAMES = ["Emerging", "Progressing", "Ready"]
REVIEW_COST = 0.35
DEFAULT_REVIEW_THRESHOLD = 0.55


@dataclass
class FoldOutput:
    index: np.ndarray
    probabilities: np.ndarray
    prediction: np.ndarray
    hybrid_prediction: np.ndarray
    rule_prediction: np.ndarray
    fold: int
    low_rule: np.ndarray
    high_rule: np.ndarray


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 1000:
        return
    headers = {"User-Agent": "Mozilla/5.0 (research reproducibility; contact in README)"}
    response = requests.get(url, headers=headers, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)


def jsonstat_to_frame(payload: dict) -> pd.DataFrame:
    ids = payload["id"]
    shape = tuple(payload["size"])
    inverse = {}
    for dim in ids:
        index = payload["dimension"][dim]["category"]["index"]
        inverse[dim] = {position: code for code, position in index.items()}
    rows = []
    for flat_index, value in payload.get("value", {}).items():
        coordinates = np.unravel_index(int(flat_index), shape)
        row = {dim: inverse[dim][coordinates[i]] for i, dim in enumerate(ids)}
        row["value"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def load_digital_intensity(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = jsonstat_to_frame(payload)
    frame = frame[(frame["size_emp"] == "GE10") & (frame["geo"].str.fullmatch(r"[A-Z]{2}"))]
    frame["year"] = frame["time"].astype(int)
    pivot = frame.pivot_table(index=["geo", "year"], columns="indic_is", values="value")
    candidates = []
    for version, very_low, direct in [
        (4, "E_DI4_VLO", "E_DI4_GELO"),
        (3, "E_DI3_VLO", "E_DI3_GELO"),
        (2, "E_DI2_VLO", None),
        (1, "E_DI_VLO", None),
    ]:
        series = pivot[direct] if direct and direct in pivot else 100.0 - pivot.get(very_low)
        candidates.append(pd.DataFrame({"dii_basic": series, "dii_version": version}))
    selected = pd.DataFrame(index=pivot.index, columns=["dii_basic", "dii_version"], dtype=float)
    for candidate in candidates:
        missing = selected["dii_basic"].isna() & candidate["dii_basic"].notna()
        selected.loc[missing, ["dii_basic", "dii_version"]] = candidate.loc[
            missing, ["dii_basic", "dii_version"]
        ]
    return selected.dropna().reset_index().rename(columns={"geo": "Country Code"})


def build_panel(eis_path: Path, dii_path: Path) -> Tuple[pd.DataFrame, list[str]]:
    eis = pd.read_excel(eis_path, sheet_name="Dimension_scores")
    eis = eis[eis["Country Code"].astype(str).str.fullmatch(r"[A-Z]{2}")].copy()
    eis["Year"] = eis["Year"].astype(int)
    dimension_cols = [c for c in eis.columns if isinstance(c, str) and c.count(".") == 1]
    dii = load_digital_intensity(dii_path)
    panel = eis.merge(
        dii,
        left_on=["Country Code", "Year"],
        right_on=["Country Code", "year"],
        how="inner",
    ).drop(columns="year")
    target = panel[["Country Code", "Year", "dii_basic"]].copy()
    target["Year"] -= 1
    target = target.rename(columns={"dii_basic": "next_dii"})
    panel = panel.merge(target, on=["Country Code", "Year"], how="inner")
    # Country-relative tiers avoid treating non-comparable DII questionnaire versions
    # as if their absolute cut points were identical across survey years.
    panel["readiness"] = panel.groupby("Year")["next_dii"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False)
    ).astype(int)
    feature_cols = dimension_cols + ["dii_basic"]
    panel = panel.dropna(subset=["readiness", "Country Code"])
    panel = panel.sort_values(["Country Code", "Year"]).reset_index(drop=True)
    return panel, feature_cols


def model_definitions(numeric_features: list[str]) -> Dict[str, object]:
    scaled = ColumnTransformer(
        [("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                                ("scale", StandardScaler())]), numeric_features)]
    )
    imputed = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), numeric_features)]
    )
    logistic = Pipeline([
        ("preprocess", scaled),
        ("model", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=RANDOM_SEED)),
    ])
    forest = Pipeline([
        ("preprocess", imputed),
        ("model", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ])
    hgb = Pipeline([
        ("preprocess", imputed),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=250, max_leaf_nodes=15,
            l2_regularization=0.2, random_state=RANDOM_SEED,
        )),
    ])
    calibrated_hgb = CalibratedClassifierCV(hgb, method="sigmoid", cv=3)
    return {
        "Logistic regression": logistic,
        "Random forest": forest,
        "Calibrated HGB": calibrated_hgb,
    }


def knowledge_predictions(
    train: pd.DataFrame, test: pd.DataFrame, ml_prediction: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    low_dii = train["dii_basic"].quantile(0.33)
    high_dii = train["dii_basic"].quantile(0.67)
    low_digital = train["2.3"].quantile(0.33)
    high_innovators = train["3.1"].quantile(0.67)
    high_linkages = train["3.2"].quantile(0.67)
    low_rule = (test["dii_basic"] <= low_dii) & (test["2.3"] <= low_digital)
    high_rule = (
        (test["dii_basic"] >= high_dii)
        & (test["3.1"] >= high_innovators)
        & (test["3.2"] >= high_linkages)
    )
    hybrid = ml_prediction.copy()
    hybrid[low_rule.to_numpy()] = np.minimum(hybrid[low_rule.to_numpy()], 0)
    hybrid[high_rule.to_numpy()] = np.maximum(hybrid[high_rule.to_numpy()], 2)
    rule_only = np.ones(len(test), dtype=int)
    rule_only[low_rule.to_numpy()] = 0
    rule_only[high_rule.to_numpy()] = 2
    return hybrid, rule_only, low_rule.to_numpy(), high_rule.to_numpy()


def asymmetric_cost(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    # Overstating readiness is assigned three times the per-tier cost of
    # understating readiness because premature investment can strand resources.
    return np.where(y_pred > y_true, 3.0 * (y_pred - y_true), 1.0 * (y_true - y_pred))


def ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correctness = (predicted == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    score = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence > left) & (confidence <= right)
        if mask.any():
            score += mask.sum() / total * abs(correctness[mask].mean() - confidence[mask].mean())
    return float(score)


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1])[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def grouped_bootstrap_difference(
    frame: pd.DataFrame, groups: pd.Series, repetitions: int = 5000
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    unique = np.array(sorted(groups.unique()))
    diffs = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([np.flatnonzero(groups.to_numpy() == group) for group in sampled])
        diffs.append(frame.iloc[positions]["hybrid_minus_ml"].mean())
    return float(np.mean(diffs)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def evaluate(panel: pd.DataFrame, features: list[str]):
    X = panel[features]
    y = panel["readiness"].to_numpy()
    groups = panel["Country Code"]
    splitter = GroupKFold(n_splits=5)
    models = model_definitions(features)
    model_outputs = {}
    primary_folds: list[FoldOutput] = []

    for model_name, model in models.items():
        probabilities = np.zeros((len(panel), 3), dtype=float)
        predictions = np.zeros(len(panel), dtype=int)
        fold_ids = np.zeros(len(panel), dtype=int)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            fitted = clone(model).fit(X.iloc[train_idx], y[train_idx])
            proba = fitted.predict_proba(X.iloc[test_idx])
            pred = proba.argmax(axis=1)
            probabilities[test_idx] = proba
            predictions[test_idx] = pred
            fold_ids[test_idx] = fold
            if model_name == "Logistic regression":
                hybrid, rule_only, low_rule, high_rule = knowledge_predictions(
                    panel.iloc[train_idx], panel.iloc[test_idx], pred
                )
                primary_folds.append(FoldOutput(
                    index=test_idx, probabilities=proba, prediction=pred,
                    hybrid_prediction=hybrid, rule_prediction=rule_only, fold=fold,
                    low_rule=low_rule, high_rule=high_rule,
                ))
        model_outputs[model_name] = (probabilities, predictions, fold_ids)

    hybrid = np.zeros(len(panel), dtype=int)
    rule_only = np.zeros(len(panel), dtype=int)
    low_rule = np.zeros(len(panel), dtype=bool)
    high_rule = np.zeros(len(panel), dtype=bool)
    fold_ids = np.zeros(len(panel), dtype=int)
    for output in primary_folds:
        hybrid[output.index] = output.hybrid_prediction
        rule_only[output.index] = output.rule_prediction
        low_rule[output.index] = output.low_rule
        high_rule[output.index] = output.high_rule
        fold_ids[output.index] = output.fold

    return model_outputs, hybrid, rule_only, low_rule, high_rule, fold_ids


def metrics_row(name: str, y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None,
                review: np.ndarray | None = None) -> dict:
    if review is None:
        review = np.zeros(len(y), dtype=bool)
    automated = ~review
    individual_cost = asymmetric_cost(y, pred)
    decision_cost = np.where(review, REVIEW_COST, individual_cost)
    return {
        "Method": name,
        "Accuracy": accuracy_score(y, pred),
        "Balanced accuracy": balanced_accuracy_score(y, pred),
        "Macro-F1": f1_score(y, pred, average="macro"),
        "Automated accuracy": accuracy_score(y[automated], pred[automated]) if automated.any() else np.nan,
        "Review rate": review.mean(),
        "Asymmetric decision cost": decision_cost.mean(),
        "Log loss": log_loss(y, proba, labels=[0, 1, 2]) if proba is not None else np.nan,
        "Brier score": multiclass_brier(y, proba) if proba is not None else np.nan,
        "ECE": ece(y, proba) if proba is not None else np.nan,
    }


def create_figure_1(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.6, 4.8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    titles = ["Enterprise data", "Probabilistic AI", "Knowledge rules", "Selective decision", "Managerial action"]
    subtitles = [
        "Innovation dimensions\nand prior digital intensity",
        "Country-grouped\nnext-year readiness forecast",
        "Training-fold thresholds\ncap or raise readiness",
        "Low confidence cases\nenter human review",
        "Prioritized capability gaps\nand investment sequence",
    ]
    colors = ["#DCECF4", "#DDE8F9", "#EEE4F7", "#FBE8C8", "#DDEFE2"]
    xs = [0.02, 0.22, 0.42, 0.62, 0.82]
    for x, title, subtitle, color in zip(xs, titles, subtitles, colors):
        box = FancyBboxPatch((x, 0.34), 0.16, 0.38, boxstyle="round,pad=0.014",
                             facecolor=color, edgecolor="#17324D", linewidth=1.6)
        ax.add_patch(box)
        ax.text(x + 0.08, 0.60, title, ha="center", fontsize=11, fontweight="bold", color="#102A43")
        ax.text(x + 0.08, 0.45, subtitle, ha="center", va="center", fontsize=8.5, color="#334E68")
    for left, right in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((left + 0.16, 0.53), (right, 0.53), arrowstyle="-|>",
                                     mutation_scale=15, linewidth=1.5, color="#486581"))
    ax.text(0.5, 0.88, "Knowledge-based decision architecture", ha="center",
            fontsize=16, fontweight="bold", color="#102A43")
    ax.text(0.5, 0.14,
            "Every recommendation retains the probability, activated rule, confidence, review flag and capability-gap trace.",
            ha="center", fontsize=9.5, color="#486581")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_figure_2(path: Path, y: np.ndarray, probabilities: np.ndarray,
                    ml_prediction: np.ndarray, hybrid_prediction: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.35, 0.76, 0.05):
        review = probabilities.max(axis=1) < threshold
        for method, pred in [("Probabilistic AI", ml_prediction), ("Knowledge-based AI", hybrid_prediction)]:
            cost = np.where(review, REVIEW_COST, asymmetric_cost(y, pred)).mean()
            rows.append({"Threshold": threshold, "Method": method,
                         "Review rate": review.mean(), "Decision cost": cost})
    sensitivity = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=300)
    styles = {"Probabilistic AI": ("#5470C6", "o"), "Knowledge-based AI": ("#2A9D8F", "s")}
    for method, group in sensitivity.groupby("Method"):
        color, marker = styles[method]
        ax.plot(group["Review rate"] * 100, group["Decision cost"], label=method,
                color=color, marker=marker, linewidth=2.2, markersize=5)
    ax.set_xlabel("Cases referred for human review (%)")
    ax.set_ylabel("Mean asymmetric decision cost")
    ax.set_title("Decision cost under alternative review thresholds")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return sensitivity


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    eis_path = DATA / "eis_2025_database.xlsx"
    dii_path = DATA / "eurostat_isoc_e_dii.json"
    download(EIS_URL, eis_path)
    download(EUROSTAT_URL, dii_path)
    panel, features = build_panel(eis_path, dii_path)
    panel.to_csv(DATA / "analysis_panel.csv", index=False)

    model_outputs, hybrid, rule_only, low_rule, high_rule, folds = evaluate(panel, features)
    y = panel["readiness"].to_numpy()
    primary_proba, primary_prediction, _ = model_outputs["Logistic regression"]
    review = primary_proba.max(axis=1) < DEFAULT_REVIEW_THRESHOLD

    table_1_rows = []
    for name, (proba, prediction, _) in model_outputs.items():
        table_1_rows.append(metrics_row(name, y, prediction, proba))
    table_1_rows.append(metrics_row("Rule-only knowledge system", y, rule_only, None))
    table_1_rows.append(metrics_row("Knowledge-based AI", y, hybrid, primary_proba, review))
    table_1 = pd.DataFrame(table_1_rows)
    table_1.to_csv(RESULTS / "Table_1_model_performance.csv", index=False, float_format="%.4f")

    decision = panel[["Country Code", "Country Name", "Year", "next_dii", "readiness"]].copy()
    decision["fold"] = folds
    decision["ml_prediction"] = primary_prediction
    decision["hybrid_prediction"] = hybrid
    decision["review"] = review
    decision["low_rule"] = low_rule
    decision["high_rule"] = high_rule
    decision["ml_cost"] = asymmetric_cost(y, primary_prediction)
    decision["hybrid_cost"] = np.where(review, REVIEW_COST, asymmetric_cost(y, hybrid))
    decision["hybrid_minus_ml"] = decision["hybrid_cost"] - decision["ml_cost"]
    decision.to_csv(RESULTS / "decision_trace.csv", index=False)

    fold_stats = decision.groupby("fold").agg(
        ml_cost=("ml_cost", "mean"), hybrid_cost=("hybrid_cost", "mean"),
        review_rate=("review", "mean"), low_rules=("low_rule", "sum"), high_rules=("high_rule", "sum"),
    ).reset_index()
    try:
        statistic, p_value = wilcoxon(fold_stats["hybrid_cost"], fold_stats["ml_cost"], alternative="two-sided")
    except ValueError:
        statistic, p_value = np.nan, np.nan
    boot_mean, boot_low, boot_high = grouped_bootstrap_difference(
        decision, panel["Country Code"], repetitions=5000
    )
    table_2 = pd.DataFrame([
        {"Evidence": "Low-readiness rule activations", "Value": int(low_rule.sum()), "Uncertainty / note": "Held-out cases"},
        {"Evidence": "High-readiness rule activations", "Value": int(high_rule.sum()), "Uncertainty / note": "Held-out cases"},
        {"Evidence": "Human-review rate", "Value": float(review.mean()), "Uncertainty / note": f"Threshold={DEFAULT_REVIEW_THRESHOLD:.2f}"},
        {"Evidence": "Hybrid minus probabilistic-AI cost", "Value": float(decision["hybrid_minus_ml"].mean()),
         "Uncertainty / note": f"Grouped bootstrap 95% CI [{boot_low:.4f}, {boot_high:.4f}]"},
        {"Evidence": "Paired fold Wilcoxon p-value", "Value": float(p_value),
         "Uncertainty / note": f"Statistic={statistic:.3f}; five country-grouped folds"},
    ])
    table_2.to_csv(RESULTS / "Table_2_knowledge_decision_evidence.csv", index=False)

    create_figure_1(RESULTS / "Figure_1_knowledge_architecture.png")
    sensitivity = create_figure_2(
        RESULTS / "Figure_2_cost_review_tradeoff.png", y, primary_proba, primary_prediction, hybrid
    )
    sensitivity.to_csv(RESULTS / "review_threshold_sensitivity.csv", index=False)

    # Fit a final artifact for reproducibility; evaluation claims remain out-of-fold.
    final_model = clone(model_definitions(features)["Logistic regression"]).fit(panel[features], y)
    joblib.dump(final_model, RESULTS / "readiness_model.joblib")
    importance = permutation_importance(
        final_model, panel[features], y, n_repeats=40, scoring="neg_log_loss", random_state=RANDOM_SEED
    )
    pd.DataFrame({"Feature": features, "Importance": importance.importances_mean,
                  "SD": importance.importances_std}).sort_values("Importance", ascending=False).to_csv(
        RESULTS / "feature_importance.csv", index=False
    )

    summary = {
        "title": "Knowledge-Based AI Decision Support Using Enterprise Innovation Data for Digital Transformation and Entrepreneurship Readiness",
        "observations": int(len(panel)),
        "countries": int(panel["Country Code"].nunique()),
        "years": sorted(map(int, panel["Year"].unique())),
        "class_counts": panel["readiness"].value_counts().sort_index().to_dict(),
        "default_review_threshold": DEFAULT_REVIEW_THRESHOLD,
        "grouped_bootstrap_hybrid_minus_ml": {"mean": boot_mean, "lower_95": boot_low, "upper_95": boot_high},
        "wilcoxon_p": float(p_value),
        "data_hashes": {"eis": sha256(eis_path), "eurostat_dii": sha256(dii_path)},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nTable 1\n", table_1.to_string(index=False))
    print("\nTable 2\n", table_2.to_string(index=False))


if __name__ == "__main__":
    main()
