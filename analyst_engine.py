"""Reusable transformation and advanced-analysis helpers for ZubeAnalystOS."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing


MAX_UNDO_STEPS = 12


def serializable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, list):
        return [serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    return value


def operation_record(kind: str, label: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": kind,
        "label": label,
        "params": serializable(params),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def join_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    how: str,
) -> pd.DataFrame:
    """Join two working tables with stable suffixes and a clean row index."""
    if how not in {"inner", "left", "right", "outer"}:
        raise ValueError(f"Unsupported join method: {how}")
    if left_on not in left.columns or right_on not in right.columns:
        raise ValueError("Both join keys must exist in their respective datasets.")
    return left.merge(
        right,
        how=how,
        left_on=left_on,
        right_on=right_on,
        suffixes=("", "_joined"),
    ).reset_index(drop=True)


def apply_operation(df: pd.DataFrame, operation: dict[str, Any]) -> pd.DataFrame:
    """Apply one serializable transformation operation to a dataframe copy."""
    result = df.copy()
    kind = operation["kind"]
    params = operation.get("params", {})

    if kind == "missing":
        column, method = params["column"], params["method"]
        if method == "Drop rows":
            result = result.dropna(subset=[column])
        elif method == "Drop column":
            result = result.drop(columns=[column])
        elif method == "Mean":
            result[column] = result[column].fillna(result[column].mean())
        elif method == "Median":
            result[column] = result[column].fillna(result[column].median())
        elif method == "Mode":
            mode = result[column].mode(dropna=True)
            if mode.empty:
                raise ValueError(f"{column} has no populated value to use as its mode.")
            result[column] = result[column].fillna(mode.iloc[0])
        elif method == "Custom value":
            value = params.get("value")
            if pd.api.types.is_numeric_dtype(result[column]):
                value = pd.to_numeric(value)
            result[column] = result[column].fillna(value)

    elif kind == "deduplicate":
        subset = params.get("subset") or None
        result = result.drop_duplicates(subset=subset, keep=params.get("keep", "first"))

    elif kind == "convert_type":
        column, target = params["column"], params["target"]
        if target == "Number":
            result[column] = pd.to_numeric(result[column], errors="coerce")
        elif target == "Date/time":
            result[column] = pd.to_datetime(result[column], errors="coerce")
        elif target == "Text":
            result[column] = result[column].astype("string")
        elif target == "Category":
            result[column] = result[column].astype("category")
        elif target == "Boolean":
            mapping = {"true": True, "1": True, "yes": True, "y": True, "false": False, "0": False, "no": False, "n": False}
            result[column] = result[column].astype(str).str.lower().map(mapping).astype("boolean")

    elif kind == "outliers":
        column, method = params["column"], params["method"]
        series = result[column]
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - params.get("factor", 1.5) * iqr, q3 + params.get("factor", 1.5) * iqr
        if method == "Cap values":
            result[column] = series.clip(lower, upper)
        elif method == "Remove rows":
            result = result[series.between(lower, upper) | series.isna()]
        elif method == "Replace with missing":
            result.loc[~series.between(lower, upper), column] = np.nan

    elif kind == "filter":
        column, operator, value = params["column"], params["operator"], params.get("value")
        series = result[column]
        if operator == "equals":
            mask = series.astype(str) == str(value)
        elif operator == "not equal":
            mask = series.astype(str) != str(value)
        elif operator == "contains":
            mask = series.astype(str).str.contains(str(value), case=False, na=False)
        elif operator == "greater than":
            mask = series > pd.to_numeric(value)
        elif operator == "less than":
            mask = series < pd.to_numeric(value)
        elif operator == "at least":
            mask = series >= pd.to_numeric(value)
        elif operator == "at most":
            mask = series <= pd.to_numeric(value)
        elif operator == "is missing":
            mask = series.isna()
        elif operator == "is populated":
            mask = series.notna()
        else:
            raise ValueError(f"Unsupported filter: {operator}")
        result = result[mask]

    elif kind == "calculated":
        name, expression = params["name"], params["expression"]
        if not name or name in result.columns:
            raise ValueError("Choose a new, unique calculated-column name.")
        result[name] = result.eval(expression, engine="numexpr")

    elif kind == "group":
        groups, values = params["groups"], params["values"]
        aggregation = params["aggregation"]
        result = result.groupby(groups, dropna=False)[values].agg(aggregation).reset_index()

    elif kind == "pivot":
        result = pd.pivot_table(
            result,
            index=params["index"],
            columns=params["columns"],
            values=params["values"],
            aggfunc=params["aggregation"],
            fill_value=params.get("fill_value", 0),
        ).reset_index()
        result.columns = ["_".join(map(str, col)).strip("_") if isinstance(col, tuple) else str(col) for col in result.columns]

    elif kind == "melt":
        result = result.melt(
            id_vars=params["id_vars"],
            value_vars=params["value_vars"],
            var_name=params.get("var_name", "measure"),
            value_name=params.get("value_name", "value"),
        )

    elif kind == "anomaly":
        result = detect_anomalies(result, params["columns"], float(params["contamination"]))

    elif kind == "segment":
        result, _ = segment_rows(result, params["columns"], int(params["clusters"]))

    elif kind == "join":
        raise ValueError("Join operations require the companion dataset and cannot be replayed from a recipe alone.")
    else:
        raise ValueError(f"Unknown operation type: {kind}")

    return result.reset_index(drop=True)


def replay_recipe(df: pd.DataFrame, operations: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    result, skipped = df.copy(), []
    for operation in operations:
        try:
            result = apply_operation(result, operation)
        except Exception as exc:  # recipes should report partial compatibility
            skipped.append(f"{operation.get('label', operation.get('kind'))}: {exc}")
    return result, skipped


def recipe_json(name: str, operations: list[dict[str, Any]], kpis: list[dict[str, Any]]) -> str:
    return json.dumps(
        {"product": "ZubeAnalystOS", "version": 1, "name": name, "operations": operations, "kpis": kpis},
        indent=2,
        default=serializable,
    )


def compare_quality(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    def measures(frame: pd.DataFrame) -> dict[str, float]:
        return {
            "Rows": len(frame),
            "Columns": len(frame.columns),
            "Missing cells": int(frame.isna().sum().sum()),
            "Duplicate rows": int(frame.duplicated().sum()),
            "Completeness %": (1 - frame.isna().sum().sum() / max(frame.size, 1)) * 100,
        }

    old, new = measures(before), measures(after)
    return pd.DataFrame(
        [{"Metric": key, "Before": old[key], "After": new[key], "Change": new[key] - old[key]} for key in old]
    )


def statistical_test(
    df: pd.DataFrame,
    test_name: str,
    numeric_column: str | None = None,
    group_column: str | None = None,
    second_column: str | None = None,
    confidence: float = 0.95,
) -> dict[str, Any]:
    alpha = 1 - confidence
    if test_name == "Two-group mean comparison":
        groups = df[[group_column, numeric_column]].dropna().groupby(group_column)[numeric_column]
        values = [group.to_numpy() for _, group in groups]
        labels = [str(label) for label, _ in groups]
        if len(values) != 2:
            raise ValueError("Select a field containing exactly two populated groups.")
        statistic, pvalue = stats.ttest_ind(values[0], values[1], equal_var=False)
        estimate = {labels[i]: float(np.mean(values[i])) for i in range(2)}
        effect = float(np.mean(values[0]) - np.mean(values[1]))
        result_name = "Welch's t-test"
    elif test_name == "Multi-group mean comparison":
        groups = df[[group_column, numeric_column]].dropna().groupby(group_column)[numeric_column]
        values = [group.to_numpy() for _, group in groups if len(group) >= 2]
        if len(values) < 2:
            raise ValueError("At least two groups with two observations each are required.")
        statistic, pvalue = stats.f_oneway(*values)
        estimate = {str(label): float(group.mean()) for label, group in groups}
        effect = None
        result_name = "One-way ANOVA"
    elif test_name == "Category association":
        table = pd.crosstab(df[group_column], df[second_column])
        if table.shape[0] < 2 or table.shape[1] < 2:
            raise ValueError("Both fields need at least two categories.")
        statistic, pvalue, _, _ = stats.chi2_contingency(table)
        estimate = table.to_dict()
        effect = float(np.sqrt(statistic / (table.values.sum() * max(1, min(table.shape) - 1))))
        result_name = "Chi-square association test"
    elif test_name == "Numeric relationship":
        paired = df[[numeric_column, second_column]].dropna()
        if len(paired) < 4:
            raise ValueError("At least four complete paired observations are required.")
        statistic, pvalue = stats.pearsonr(paired[numeric_column], paired[second_column])
        estimate, effect = float(statistic), float(statistic)
        result_name = "Pearson correlation test"
    else:
        raise ValueError(f"Unsupported statistical test: {test_name}")

    return {
        "name": result_name,
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "confidence": confidence,
        "significant": bool(pvalue < alpha),
        "estimate": serializable(estimate),
        "effect": effect,
    }


def mean_confidence_interval(series: pd.Series, confidence: float = 0.95) -> dict[str, float]:
    values = series.dropna().astype(float)
    if len(values) < 2:
        raise ValueError("At least two populated values are required.")
    sem = stats.sem(values)
    margin = float(stats.t.ppf((1 + confidence) / 2, len(values) - 1) * sem)
    mean = float(values.mean())
    return {"mean": mean, "lower": mean - margin, "upper": mean + margin, "n": len(values)}


def detect_anomalies(df: pd.DataFrame, columns: list[str], contamination: float) -> pd.DataFrame:
    usable = df[columns].replace([np.inf, -np.inf], np.nan)
    complete = usable.dropna()
    if len(complete) < 10:
        raise ValueError("At least 10 complete rows are required for anomaly detection.")
    scaled = StandardScaler().fit_transform(complete)
    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(scaled)
    scores = -model.score_samples(scaled)
    result = df.copy()
    result["anomaly"] = False
    result["anomaly_score"] = np.nan
    result.loc[complete.index, "anomaly"] = labels == -1
    result.loc[complete.index, "anomaly_score"] = scores
    return result


def segment_rows(df: pd.DataFrame, columns: list[str], clusters: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = df[columns].replace([np.inf, -np.inf], np.nan)
    complete = usable.dropna()
    if len(complete) < clusters * 3:
        raise ValueError("Use at least three complete rows per segment.")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(complete)
    model = KMeans(n_clusters=clusters, n_init="auto", random_state=42)
    labels = model.fit_predict(scaled)
    result = df.copy()
    result["segment"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result.loc[complete.index, "segment"] = labels + 1
    profiles = complete.assign(segment=labels + 1).groupby("segment")[columns].mean().round(3)
    profiles["rows"] = pd.Series(labels + 1).value_counts().sort_index().values
    return result, profiles.reset_index()


def forecast_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    periods: int,
    frequency: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = {"Day": "D", "Week": "W", "Month": "ME", "Quarter": "QE"}
    rule = rules[frequency]
    history = (
        df[[date_column, value_column]].dropna().set_index(date_column)[value_column]
        .resample(rule).sum().astype(float)
    )
    if len(history) < 6:
        raise ValueError("At least six time periods are required for a basic forecast.")
    seasonal_periods = {"Day": 7, "Week": 4, "Month": 12, "Quarter": 4}[frequency]
    use_seasonal = len(history) >= seasonal_periods * 2
    model = ExponentialSmoothing(
        history,
        trend="add",
        seasonal="add" if use_seasonal else None,
        seasonal_periods=seasonal_periods if use_seasonal else None,
        initialization_method="estimated",
    ).fit(optimized=True)
    forecast = model.forecast(periods)
    history_df = history.rename("value").reset_index()
    forecast_df = forecast.rename("forecast").reset_index().rename(columns={"index": date_column})
    residual_std = float(np.std(model.resid, ddof=1)) if len(model.resid) > 1 else 0
    forecast_df["lower"] = forecast_df["forecast"] - 1.96 * residual_std
    forecast_df["upper"] = forecast_df["forecast"] + 1.96 * residual_std
    return history_df, forecast_df
