import json

import numpy as np
import pandas as pd
import pytest

from analyst_engine import (
    apply_operation,
    compare_quality,
    detect_anomalies,
    forecast_series,
    join_frames,
    mean_confidence_interval,
    operation_record,
    recipe_json,
    replay_recipe,
    segment_rows,
    statistical_test,
)


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "region": ["West", "West", "East", "North", "East", None],
            "group": ["A", "A", "B", "B", "A", "B"],
            "revenue": [100.0, 100.0, 180.0, 240.0, np.nan, 160.0],
            "cost": [60.0, 60.0, 90.0, 130.0, 80.0, 70.0],
            "code": ["1", "1", "2", "3", "4", "5"],
        }
    )


def op(kind, **params):
    return operation_record(kind, kind, params)


@pytest.mark.parametrize("method", ["Mean", "Median", "Mode", "Custom value"])
def test_missing_value_fill_strategies(frame, method):
    params = {"column": "revenue", "method": method, "value": "125"}
    result = apply_operation(frame, op("missing", **params))
    assert result["revenue"].isna().sum() == 0


def test_missing_drop_rows_and_column(frame):
    rows = apply_operation(frame, op("missing", column="region", method="Drop rows"))
    column = apply_operation(frame, op("missing", column="region", method="Drop column"))
    assert len(rows) == 5
    assert "region" not in column


def test_deduplication_supports_keys_and_keep_modes(frame):
    exact = apply_operation(frame, op("deduplicate", subset=[], keep="first"))
    remove_all = apply_operation(frame, op("deduplicate", subset=["region", "revenue"], keep=False))
    assert len(exact) == 5
    assert len(remove_all) == 4


@pytest.mark.parametrize("target", ["Number", "Text", "Category", "Boolean", "Date/time"])
def test_type_corrections(target):
    source = pd.DataFrame({"value": ["1", "0", "yes", "2025-01-01"]})
    result = apply_operation(source, op("convert_type", column="value", target=target))
    assert len(result) == 4
    if target == "Number":
        assert pd.api.types.is_numeric_dtype(result["value"])
    elif target == "Date/time":
        assert pd.api.types.is_datetime64_any_dtype(result["value"])


@pytest.mark.parametrize("method", ["Cap values", "Remove rows", "Replace with missing"])
def test_outlier_treatments(method):
    source = pd.DataFrame({"value": [10, 11, 12, 13, 14, 1000]})
    result = apply_operation(source, op("outliers", column="value", method=method, factor=1.5))
    if method == "Cap values":
        assert result["value"].max() < 1000
    elif method == "Remove rows":
        assert len(result) == 5
    else:
        assert result["value"].isna().sum() == 1


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        ("equals", "West", 2), ("not equal", "West", 4), ("contains", "est", 2),
        ("greater than", "150", 3), ("less than", "150", 2),
        ("at least", "180", 2), ("at most", "100", 2),
        ("is missing", None, 1), ("is populated", None, 5),
    ],
)
def test_filters(frame, operator, value, expected):
    column = "region" if operator in {"equals", "not equal", "contains", "is missing", "is populated"} else "revenue"
    result = apply_operation(frame, op("filter", column=column, operator=operator, value=value))
    assert len(result) == expected


def test_calculation_group_pivot_and_melt(frame):
    calculated = apply_operation(frame, op("calculated", name="margin", expression="revenue - cost"))
    grouped = apply_operation(frame, op("group", groups=["group"], values=["cost"], aggregation="sum"))
    pivoted = apply_operation(frame, op("pivot", index=["group"], columns="region", values="cost", aggregation="sum"))
    melted = apply_operation(frame, op("melt", id_vars=["group"], value_vars=["revenue", "cost"], var_name="metric", value_name="amount"))
    assert "margin" in calculated
    assert grouped["cost"].sum() == frame["cost"].sum()
    assert len(pivoted) == 2
    assert len(melted) == len(frame) * 2


def test_invalid_calculated_field_and_unknown_operation(frame):
    with pytest.raises(ValueError):
        apply_operation(frame, op("calculated", name="cost", expression="revenue-cost"))
    with pytest.raises(ValueError):
        apply_operation(frame, op("not-real"))


def test_all_join_modes_and_validation():
    left = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
    right = pd.DataFrame({"key": [2, 3], "value": [200, 300]})
    assert len(join_frames(left, right, "id", "key", "inner")) == 1
    assert len(join_frames(left, right, "id", "key", "left")) == 2
    assert len(join_frames(left, right, "id", "key", "right")) == 2
    outer = join_frames(left, right, "id", "key", "outer")
    assert len(outer) == 3 and "value_joined" in outer
    with pytest.raises(ValueError):
        join_frames(left, right, "missing", "key", "inner")


def test_recipe_round_trip_and_partial_replay(frame):
    operations = [
        op("missing", column="revenue", method="Median", value=None),
        operation_record("join", "join is dataset-specific", {"how": "left"}),
    ]
    payload = json.loads(recipe_json("demo", operations, [{"name": "Revenue"}]))
    result, skipped = replay_recipe(frame, payload["operations"])
    assert result["revenue"].isna().sum() == 0
    assert len(skipped) == 1
    assert payload["kpis"][0]["name"] == "Revenue"


def test_quality_comparison(frame):
    cleaned = frame.dropna().drop_duplicates()
    comparison = compare_quality(frame, cleaned).set_index("Metric")
    assert comparison.loc["Missing cells", "After"] == 0
    assert comparison.loc["Completeness %", "After"] == 100


def test_confidence_and_all_statistical_tests():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "group": ["A"] * 30 + ["B"] * 30,
            "category": (["X", "Y"] * 30),
            "value": np.r_[rng.normal(10, 1, 30), rng.normal(13, 1, 30)],
        }
    )
    df["value2"] = df["value"] * 2 + rng.normal(0, .1, len(df))
    interval = mean_confidence_interval(df["value"])
    assert interval["lower"] < interval["mean"] < interval["upper"]
    two = statistical_test(df, "Two-group mean comparison", "value", "group")
    multi = statistical_test(df.assign(group3=np.tile(["A", "B", "C"], 20)), "Multi-group mean comparison", "value", "group3")
    category = statistical_test(df, "Category association", group_column="group", second_column="category")
    numeric = statistical_test(df, "Numeric relationship", "value", second_column="value2")
    assert two["significant"]
    assert multi["name"] == "One-way ANOVA"
    assert category["name"].startswith("Chi-square")
    assert numeric["estimate"] > .9


def test_anomaly_segmentation_and_forecast():
    n = 36
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="ME"),
            "revenue": np.linspace(100, 450, n),
            "cost": np.linspace(60, 210, n),
        }
    )
    df.loc[n - 1, "revenue"] = 4000
    anomalies = detect_anomalies(df, ["revenue", "cost"], .05)
    segmented, profiles = segment_rows(df, ["revenue", "cost"], 3)
    history, future = forecast_series(df, "date", "cost", 6, "Month")
    assert anomalies["anomaly"].sum() >= 1
    assert segmented["segment"].notna().all() and len(profiles) == 3
    assert len(history) == 36 and len(future) == 6


def test_models_reject_insufficient_data():
    tiny = pd.DataFrame({"x": [1, 2, 3], "y": [2, 3, 4]})
    with pytest.raises(ValueError):
        detect_anomalies(tiny, ["x", "y"], .1)
    with pytest.raises(ValueError):
        segment_rows(tiny, ["x", "y"], 2)
