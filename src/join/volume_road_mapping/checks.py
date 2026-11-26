"""[서연] ⑤ 사후 점검 (평소값 가능 여부 · 날짜).

원래 src/traffic_mapping.py 한 파일이던 것을 단계별로 나눴다 (함수 내용은 그대로).
"""
from __future__ import annotations
import argparse
import hashlib
import io
import re
import platform
import subprocess
import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd

from src.io_utils import derived_of
from .common import STABLE_LINKS_RELATIVE, json_value, normalized_road_name, select_volume_files


# [서연] Diagnose baseline feasibility without inventing weather labels or computing future-leaking baselines.
def audit_traffic_baseline(volume: pd.DataFrame, stable: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"date", "hour", "spot_id", "io_type", "road_name", "traffic_volume"}
    missing = required - set(volume.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    reference = stable[["road_id", "road_name"]].drop_duplicates().copy()
    reference["normalized_road_name"] = reference["road_name"].map(normalized_road_name)
    if reference.groupby("normalized_road_name")["road_id"].nunique().gt(1).any():
        raise ValueError("Normalized road-name collision")
    reference = reference.drop_duplicates("road_id")
    source = volume.copy()
    source["normalized_road_name"] = source["road_name"].map(normalized_road_name)
    joined = source.merge(reference[["normalized_road_name", "road_id"]],
                          on="normalized_road_name", how="left", validate="many_to_one")
    if len(joined) != len(source):
        raise ValueError("Join multiplied observation rows")
    matched = joined.loc[joined["road_id"].notna()].copy()
    matched["parsed_date"] = pd.to_datetime(matched["date"], errors="coerce")
    if matched["parsed_date"].isna().any():
        raise ValueError("Matched observations contain invalid dates")
    matched["weekday"] = matched["parsed_date"].dt.day_of_week
    matched["volume_numeric"] = pd.to_numeric(matched["traffic_volume"], errors="coerce")
    matched["valid_volume"] = matched["volume_numeric"].notna() & matched["volume_numeric"].ge(0)
    # [서연] Unique days, not rows, determine weekday/hour baseline sample size.
    keys = ["road_id", "spot_id", "io_type", "weekday", "hour"]
    grouped = matched.groupby(keys, dropna=False, observed=True)
    summary = grouped.agg(
        observation_rows=("date", "size"),
        observed_days=("parsed_date", "nunique"),
        valid_rows=("valid_volume", "sum"),
        first_date=("parsed_date", "min"),
        last_date=("parsed_date", "max"),
    ).reset_index()
    valid_days = (matched.loc[matched["valid_volume"]]
                  .groupby(keys, dropna=False, observed=True)["parsed_date"].nunique()
                  .rename("valid_days").reset_index())
    summary = summary.merge(valid_days, on=keys, how="left", validate="one_to_one")
    summary["valid_days"] = summary["valid_days"].fillna(0).astype(int)
    summary["baseline_at_least_3_days"] = summary["valid_days"].ge(3)
    summary["baseline_at_least_5_days"] = summary["valid_days"].ge(5)
    summary["first_date"] = summary["first_date"].dt.strftime("%Y-%m-%d")
    summary["last_date"] = summary["last_date"].dt.strftime("%Y-%m-%d")
    daily = matched.groupby("parsed_date").agg(rows=("date", "size"),
                                                  valid_rows=("valid_volume", "sum"),
                                                  roads=("road_id", "nunique"),
                                                  spots=("spot_id", "nunique"))
    weather_like = [c for c in volume.columns if any(
        token in c.lower() for token in ("rain", "snow", "precip", "weather", "강수", "적설", "기상"))]
    report = {
        "source_rows": len(source), "matched_rows": len(matched),
        "matched_date_count": int(matched["parsed_date"].nunique()),
        "date_min": matched["parsed_date"].min().strftime("%Y-%m-%d") if len(matched) else None,
        "date_max": matched["parsed_date"].max().strftime("%Y-%m-%d") if len(matched) else None,
        "hour_distribution": {str(k): int(v) for k, v in matched["hour"].value_counts().items()},
        "weekday_distribution": {str(k): int(v) for k, v in matched["weekday"].value_counts().sort_index().items()},
        "daily_rows_min": int(daily["rows"].min()) if len(daily) else 0,
        "daily_rows_max": int(daily["rows"].max()) if len(daily) else 0,
        "daily_rows_median": float(daily["rows"].median()) if len(daily) else 0,
        "baseline_groups": len(summary),
        "groups_with_3_valid_days": int(summary["baseline_at_least_3_days"].sum()),
        "groups_with_5_valid_days": int(summary["baseline_at_least_5_days"].sum()),
        "groups_below_3_valid_days": int((~summary["baseline_at_least_3_days"]).sum()),
        "weather_candidate_columns": weather_like,
        "all_source_columns": list(volume.columns),
        "notes": [
            "Three/five valid days is a diagnostic threshold, not a scientifically established baseline.",
            "Weather-normal-day baseline cannot be approved until weather-day flags are joined.",
            "Training-only/past-only baselines must be computed separately to avoid leakage.",
            "Daily coverage counts present records, not missing expected time-grid cells.",
        ],
    }
    return summary, report


# [서연] One read-only baseline feasibility report; no traffic baseline values are calculated.
def execute_baseline_audit(project_root: Path, data_root: Path) -> int:
    inputs = select_volume_files(derived_of(data_root) / "topis_traffic_volume")
    stable_path = project_root / STABLE_LINKS_RELATIVE
    volume = pd.concat([pd.read_parquet(inputs["train"]), pd.read_parquet(inputs["eval"])],
                       ignore_index=True)
    stable = pd.read_csv(stable_path, dtype={"road_id": "string"}, encoding="utf-8-sig")
    summary, report = audit_traffic_baseline(volume, stable)
    report["inputs"] = {"train": str(inputs["train"]), "eval": str(inputs["eval"]),
                        "stable_links": str(stable_path)}
    out_dir = project_root / "outputs" / "tables"
    log_dir = project_root / "logs" / "ysy" / "volume_road_mapping"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    version = 1
    while True:
        suffix = f"{stamp}_(성공)_v{version:02d}"
        table_path = out_dir / f"yang_seoyeon_baseline_feasibility_{suffix}.csv"
        log_path = log_dir / f"baseline_feasibility_{suffix}.json"
        if not table_path.exists() and not log_path.exists():
            break
        version += 1
    with table_path.open("x", encoding="utf-8-sig", newline="") as f:
        summary.to_csv(f, index=False)
    report["table_path"] = str(table_path)
    with log_path.open("x", encoding="utf-8") as f:
        json.dump(json_value(report), f, ensure_ascii=False, indent=2)
    for key in ("source_rows", "matched_rows", "matched_date_count", "date_min",
                "date_max", "baseline_groups", "groups_with_3_valid_days",
                "groups_with_5_valid_days", "groups_below_3_valid_days",
                "weather_candidate_columns"):
        print(f"{key}: {report[key]}")
    print(f"Baseline feasibility CSV: {table_path}")
    print(f"Validation JSON: {log_path}")
    return 0


# [서연] Audit sampled dates and whether explicit weather/comparison labels exist.
def audit_traffic_dates(volume: pd.DataFrame, stable: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {"date", "road_name", "spot_id", "io_type", "hour", "traffic_volume"}
    if required - set(volume.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(volume.columns))}")
    roads = stable[["road_id", "road_name"]].drop_duplicates().copy()
    roads["normalized_road_name"] = roads["road_name"].map(normalized_road_name)
    if roads.groupby("normalized_road_name")["road_id"].nunique().gt(1).any():
        raise ValueError("Road-name normalization collision")
    roads = roads.drop_duplicates("road_id")
    if len(roads) != 90:
        raise ValueError(f"Expected 90 analysis roads; found {len(roads)}")
    observations = volume.copy()
    observations["normalized_road_name"] = observations["road_name"].map(normalized_road_name)
    observations = observations.merge(roads[["normalized_road_name", "road_id"]],
                                      on="normalized_road_name", how="left", validate="many_to_one")
    if len(observations) != len(volume):
        raise ValueError("Observation count changed during road-name join")
    observations["parsed_date"] = pd.to_datetime(observations["date"], errors="coerce")
    if observations["parsed_date"].isna().any():
        raise ValueError("Unparseable dates; inspect source")
    observations["valid_volume"] = pd.to_numeric(observations["traffic_volume"], errors="coerce").ge(0)
    observations["in_scope"] = observations["road_id"].notna()
    dates = observations.groupby("parsed_date", observed=True).agg(
        source_rows=("date", "size"), matched_rows=("in_scope", "sum"),
        valid_volume_rows=("valid_volume", "sum"),
        unique_spots=("spot_id", "nunique"),
        unique_hours=("hour", "nunique"),
    ).reset_index().sort_values("parsed_date")
    dates["year"] = dates["parsed_date"].dt.year
    dates["month"] = dates["parsed_date"].dt.month
    dates["weekday"] = dates["parsed_date"].dt.day_of_week
    dates["gap_from_previous_days"] = dates["parsed_date"].diff().dt.days
    dates["date"] = dates["parsed_date"].dt.strftime("%Y-%m-%d")
    dates = dates.drop(columns=["parsed_date"])
    months = dates.groupby(["year", "month"], observed=True).agg(
        sampled_days=("date", "size"), source_rows=("source_rows", "sum"),
        matched_rows=("matched_rows", "sum"),
        min_date=("date", "min"), max_date=("date", "max"),
    ).reset_index()
    candidates = [c for c in volume.columns if any(t in c.lower() for t in
        ("rain", "snow", "precip", "weather", "강수", "적설", "기상", "comparison", "control", "baseline", "비교일", "악천후"))]
    report = {
        "source_rows": len(volume), "matched_rows": int(observations["in_scope"].sum()),
        "sampled_dates": len(dates),
        "year_day_counts": {str(k): int(v) for k, v in dates["year"].value_counts().sort_index().items()},
        "weekday_day_counts": {str(k): int(v) for k, v in dates["weekday"].value_counts().sort_index().items()},
        "gaps_over_30_days": int(dates["gap_from_previous_days"].gt(30).sum()),
        "maximum_gap_days": int(dates["gap_from_previous_days"].max()) if len(dates) > 1 else 0,
        "potential_weather_or_comparison_columns": candidates,
        "weather_day_labels_verified": False,
        "notes": ["Date coverage does not establish whether days are adverse-weather or controls.",
                  "Do not compute normal-weather baselines without verified weather labels.",
                  "Do not pool distant years without checking comparability."],
    }
    return dates, months, report


def execute_date_audit(project_root: Path, data_root: Path) -> int:
    inputs = select_volume_files(derived_of(data_root) / "topis_traffic_volume")
    stable_path = project_root / STABLE_LINKS_RELATIVE
    volume = pd.concat([pd.read_parquet(inputs["train"]), pd.read_parquet(inputs["eval"])], ignore_index=True)
    stable = pd.read_csv(stable_path, dtype={"road_id": "string"}, encoding="utf-8-sig")
    dates, months, report = audit_traffic_dates(volume, stable)
    report["inputs"] = {"train": str(inputs["train"]), "eval": str(inputs["eval"]), "stable_links": str(stable_path)}
    out_dir = project_root / "outputs" / "tables"
    log_dir = project_root / "logs" / "ysy" / "volume_road_mapping"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    version = 1
    while True:
        suffix = f"{stamp}_(성공)_v{version:02d}"
        dates_path = out_dir / f"yang_seoyeon_sampled_date_audit_{suffix}.csv"
        months_path = out_dir / f"yang_seoyeon_month_coverage_{suffix}.csv"
        log_path = log_dir / f"date_audit_{suffix}.json"
        if not any(p.exists() for p in (dates_path, months_path, log_path)):
            break
        version += 1
    with dates_path.open("x", encoding="utf-8-sig", newline="") as f:
        dates.to_csv(f, index=False)
    with months_path.open("x", encoding="utf-8-sig", newline="") as f:
        months.to_csv(f, index=False)
    report["outputs"] = {"dates": str(dates_path), "months": str(months_path), "log": str(log_path)}
    with log_path.open("x", encoding="utf-8") as f:
        json.dump(json_value(report), f, ensure_ascii=False, indent=2)
    for k in ("source_rows", "matched_rows", "sampled_dates", "year_day_counts", "weekday_day_counts",
              "gaps_over_30_days", "maximum_gap_days", "potential_weather_or_comparison_columns"):
        print(f"{k}: {report[k]}")
    print(f"Date audit CSV: {dates_path}")
    print(f"Monthly coverage CSV: {months_path}")
    print(f"Validation JSON: {log_path}")
    return 0
