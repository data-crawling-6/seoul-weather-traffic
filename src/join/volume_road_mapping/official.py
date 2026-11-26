"""[서연] ④ 도로 이름 기준 공식 매핑표 · 관측 결합.

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

from .candidates import MAP_AUTO_MAX_DISTANCE_M, build_mapping_versions, prepare_mapping_links
from src.io_utils import derived_of
from .common import LINES_FILENAME, STABLE_LINKS_RELATIVE, json_value, normalized_road_name, select_volume_files
from .review import select_mapping_artifact_pair


# [서연] Full-geometry road-level evaluation, not a reclassification of top-three links.
def evaluate_full_road_candidates(versions: pd.DataFrame, roads: Any,
                                  baseline: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    import geopandas as gpd
    import numpy as np
    import shapely

    required = {"coordinate_version_id", "spot_id", "io_type", "road_name", "latitude", "longitude", "valid_coordinate"}
    if required - set(versions) or {"road_id", "road_name", "link_id", "geometry"} - set(roads):
        raise ValueError("Missing columns for full road evaluation")
    if versions["coordinate_version_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate coordinate version")
    if baseline["coordinate_version_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate baseline coordinate version")
    if set(versions["coordinate_version_id"].astype(str)) != set(baseline["coordinate_version_id"].astype(str)):
        raise ValueError("Baseline and recomputed coordinate versions differ")
    if roads.crs is None or roads.geometry.isna().any() or roads.geometry.is_empty.any():
        raise ValueError("Missing road CRS or geometry")
    roads = roads.to_crs(epsg=5179).copy()
    roads["road_id"] = roads["road_id"].astype(str)
    roads["normalized_name"] = roads["road_name"].map(normalized_road_name)
    road_index = {rid: np.asarray(indices, dtype=int) for rid, indices in
                  roads.reset_index(drop=True).groupby("road_id", sort=False).indices.items()}
    roads = roads.reset_index(drop=True)
    geoms = roads.geometry.to_numpy()
    baseline_index = baseline.set_index(baseline["coordinate_version_id"].astype(str))
    rows = []
    for item in versions.to_dict("records"):
        vid = str(item["coordinate_version_id"])
        previous = baseline_index.loc[vid]
        common = {"coordinate_version_id": vid, "spot_id": item["spot_id"], "io_type": item["io_type"],
                  "road_name": item["road_name"], "baseline_status": previous["mapping_status"],
                  "baseline_selected_road_id": previous.get("selected_road_id"),
                  "max_coordinate_change_m": previous.get("max_coordinate_change_m"),
                  "invalid_rows_in_spot_direction": previous.get("invalid_rows_in_spot_direction")}
        if not bool(item["valid_coordinate"]):
            rows.append({**common, "best_road_id": "", "best_road_distance_m": None,
                         "best_road_link_id": "", "nearest_other_road_distance_m": None,
                         "other_road_gap_m": None, "matching_road_count": 0,
                         "full_geometry_evidence": "invalid_coordinate", "mapping_decision": "unchanged"})
            continue
        point = gpd.GeoSeries(gpd.points_from_xy([float(item["longitude"])], [float(item["latitude"])]),
                             crs="EPSG:4326").to_crs(epsg=5179).iloc[0]
        distances = shapely.distance(geoms, point)
        candidates = []
        for rid, indices in road_index.items():
            idx = min(indices, key=lambda i: (float(distances[i]), str(roads.iloc[i]["link_id"])))
            candidates.append((rid, str(roads.iloc[idx]["road_name"]), float(distances[idx]),
                               str(roads.iloc[idx]["link_id"])))
        name = normalized_road_name(item["road_name"])
        matching = sorted((c for c in candidates if name and normalized_road_name(c[1]) == name),
                          key=lambda c: (c[2], c[0]))
        other = sorted((c for c in candidates if normalized_road_name(c[1]) != name),
                       key=lambda c: (c[2], c[0]))
        best = matching[0] if matching else None
        gap = (other[0][2] - best[2]) if best and other else None
        if not best:
            status = "no_exact_name_road_in_analysis"
        elif best[2] > MAP_AUTO_MAX_DISTANCE_M:
            status = "matching_road_too_far"
        elif gap is None or gap < 20:
            status = "competing_road_too_close"
        else:
            status = "road_candidate_strong_geometry_only"
        rows.append({**common, "best_road_id": best[0] if best else "",
                     "best_road_distance_m": round(best[2], 3) if best else None,
                     "best_road_link_id": best[3] if best else "",
                     "nearest_other_road_distance_m": round(other[0][2], 3) if other else None,
                     "other_road_gap_m": round(gap, 3) if gap is not None else None,
                     "matching_road_count": len(matching),
                     "full_geometry_evidence": status, "mapping_decision": "unchanged"})
    result = pd.DataFrame(rows)
    # [서연] Require all valid coordinate versions for the same spot/direction to support one road.
    result["spot_direction_consistency"] = "unverified"
    for _, group in result.groupby(["spot_id", "io_type"], dropna=False):
        ids = set(group.loc[group["full_geometry_evidence"].eq("road_candidate_strong_geometry_only"), "best_road_id"])
        all_strong = group["full_geometry_evidence"].eq("road_candidate_strong_geometry_only").all()
        result.loc[group.index, "spot_direction_consistency"] = (
            "same_road_across_all_versions" if all_strong and len(ids) == 1
            else "mixed_or_incomplete_evidence")
    # [서연] Road candidates remain proposals; tunnel/layer/direction not resolved.
    result["proposed_road_only_for_review"] = (
        result["full_geometry_evidence"].eq("road_candidate_strong_geometry_only") &
        result["spot_direction_consistency"].eq("same_road_across_all_versions") &
        pd.to_numeric(result["invalid_rows_in_spot_direction"], errors="coerce").fillna(1).eq(0)
    )
    previous_confirmed = result["baseline_status"].eq("auto_confirmed_road_only")
    confirmed_id = result["baseline_selected_road_id"].fillna("").astype(str)
    result["baseline_conflict_flag"] = previous_confirmed & result["best_road_id"].ne(confirmed_id)
    counts = result["full_geometry_evidence"].value_counts().to_dict()
    report = {"coordinate_versions": len(result),
              "full_geometry_evidence_counts": {str(k): int(v) for k, v in counts.items()},
              "review_proposals_total": int(result["proposed_road_only_for_review"].sum()),
              "new_review_proposals_excluding_baseline_auto": int((result["proposed_road_only_for_review"] & ~previous_confirmed).sum()),
              "baseline_conflicts": int(result["baseline_conflict_flag"].sum()),
              "validation_checks": {"population_preserved": len(result) == len(versions),
                                    "unique_versions": bool(result["coordinate_version_id"].is_unique),
                                    "no_decisions_changed": bool(result["mapping_decision"].eq("unchanged").all())}}
    return result, report


def execute_full_road_evaluation(project_root: Path, data_root: Path) -> int:
    import geopandas as gpd
    volume_paths = select_volume_files(derived_of(data_root) / "topis_traffic_volume")
    paths = {"train": volume_paths["train"], "eval": volume_paths["eval"],
             "stable_links": project_root / STABLE_LINKS_RELATIVE,
             "lines": derived_of(data_root) / "road_lines" / LINES_FILENAME}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    volume = pd.concat([pd.read_parquet(paths["train"]), pd.read_parquet(paths["eval"])], ignore_index=True)
    versions = build_mapping_versions(volume)
    stable = pd.read_csv(paths["stable_links"], dtype={"link_id": "string"}, encoding="utf-8-sig")
    lines = gpd.read_file(paths["lines"])
    roads = prepare_mapping_links(stable, lines)
    _, baseline_path = select_mapping_artifact_pair(project_root)
    baseline = pd.read_csv(baseline_path, encoding="utf-8-sig", dtype={"coordinate_version_id": "string"})
    result, report = evaluate_full_road_candidates(versions, roads, baseline)
    if not all(report["validation_checks"].values()):
        raise ValueError(f"Failed full geometry validation: {report['validation_checks']}")
    output_dir = project_root / "outputs" / "tables"
    log_dir = project_root / "logs" / "ysy" / "volume_road_mapping"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    version = 1
    while True:
        suffix = f"{stamp}_(성공)_v{version:02d}"
        csv_path = output_dir / f"yang_seoyeon_full_road_evaluation_{suffix}.csv"
        json_path = log_dir / f"full_road_evaluation_{suffix}.json"
        if not csv_path.exists() and not json_path.exists():
            break
        version += 1
    report.update({"status": "success", "baseline": str(baseline_path),
                   "source_files": {k: str(v) for k, v in paths.items()},
                   "output_csv": str(csv_path), "output_json": str(json_path),
                   "limitations": "Review proposals only; road layers, physical carriageway and direction are unverified."})
    with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        result.to_csv(handle, index=False)
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(json_value(report), handle, ensure_ascii=False, indent=2)
    print(f"Versions: {report['coordinate_versions']}; evidence: {report['full_geometry_evidence_counts']}")
    print(f"New road-only review proposals: {report['new_review_proposals_excluding_baseline_auto']}")
    print(f"Baseline conflicts: {report['baseline_conflicts']}")
    print(f"Assessment CSV: {csv_path}")
    print(f"Validation JSON: {json_path}")
    return 0


# [서연] Official road-name mapping: no spatial inference and no source modifications.
def build_official_road_mapping(volume: pd.DataFrame, stable: pd.DataFrame
                                ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {"spot_id", "io_type", "latitude", "longitude", "road_name"}
    stable_required = {"road_id", "road_name"}
    if required - set(volume) or stable_required - set(stable):
        raise ValueError("Missing official road-name mapping inputs")
    roads = stable[["road_id", "road_name"]].drop_duplicates().copy()
    roads["normalized_road_name"] = roads["road_name"].map(normalized_road_name)
    if roads["road_id"].isna().any() or roads["road_name"].isna().any():
        raise ValueError("Missing reference road ID or name")
    if roads["normalized_road_name"].eq("").any():
        raise ValueError("Empty reference road name")
    if roads.groupby("normalized_road_name")["road_id"].nunique().gt(1).any():
        raise ValueError("Normalized road name maps to multiple road IDs")
    if roads.groupby("road_id")["normalized_road_name"].nunique().gt(1).any():
        raise ValueError("Road ID maps to multiple normalized names")
    roads = roads.drop_duplicates("road_id")
    if len(roads) != 90:
        raise ValueError(f"Expected 90 analysis roads, found {len(roads)}")
    versions = build_mapping_versions(volume)
    versions["normalized_road_name"] = versions["road_name"].map(normalized_road_name)
    lookup = roads[["road_id", "normalized_road_name"]]
    result = versions.merge(lookup, on="normalized_road_name", how="left",
                            validate="many_to_one", indicator=True)
    result["mapping_method"] = result["_merge"].map({
        "both": "official_road_name_exact", "left_only": "not_applicable",
        "right_only": "not_applicable"}).astype(str)
    result["mapping_status"] = "out_of_scope_not_in_90_roads"
    result.loc[result["_merge"].eq("both"), "mapping_status"] = "matched"
    result.loc[result["normalized_road_name"].eq(""), "mapping_status"] = "invalid_or_ambiguous"
    result = result.drop(columns=["_merge"])
    if len(result) != len(versions) or not result["coordinate_version_id"].is_unique:
        raise ValueError("Mapping changed coordinate-version population")
    if result.loc[result["mapping_status"].eq("matched"), "road_id"].isna().any():
        raise ValueError("Matched row missing road ID")
    if result.loc[~result["mapping_status"].eq("matched"), "road_id"].notna().any():
        raise ValueError("Out-of-scope row unexpectedly mapped")
    # [서연] Count distinct spots per road; coordinate versions and IO types are not extra spots.
    matched = result.loc[result["mapping_status"].eq("matched")].copy()
    spot_counts = matched.groupby("road_id")["spot_id"].nunique()
    direction_counts = matched.groupby("road_id").apply(
        lambda g: g[["spot_id", "io_type"]].drop_duplicates().shape[0], include_groups=False)
    version_counts = matched.groupby("road_id").size()
    coverage = roads.copy()
    coverage["spot_count"] = coverage["road_id"].map(spot_counts).fillna(0).astype(int)
    coverage["spot_direction_count"] = coverage["road_id"].map(direction_counts).fillna(0).astype(int)
    coverage["coordinate_version_count"] = coverage["road_id"].map(version_counts).fillna(0).astype(int)
    coverage["has_traffic_data"] = coverage["spot_count"].gt(0)
    # [서연] Observation coverage is computed only after date/time-safe volume joins.
    for name in ("observation_count", "valid_observation_count", "missing_rate",
                 "first_observation_date", "last_observation_date", "analysis_eligible"):
        coverage[name] = pd.NA
    report = {
        "coordinate_versions": len(result),
        "mapping_status_counts": {str(k): int(v) for k, v in result["mapping_status"].value_counts().items()},
        "analysis_roads": len(coverage),
        "roads_with_spots": int(coverage["has_traffic_data"].sum()),
        "roads_without_spots": int((~coverage["has_traffic_data"]).sum()),
        "distinct_matched_spots": int(matched["spot_id"].nunique()),
        "spot_count_distribution": {str(k): int(v) for k, v in
                                    coverage.loc[coverage["spot_count"].gt(0), "spot_count"].value_counts().sort_index().items()},
        "validation_checks": {
            "version_population_preserved": len(result) == len(versions),
            "unique_versions": bool(result["coordinate_version_id"].is_unique),
            "road_population_90": len(coverage) == 90,
            "unique_road_ids": bool(coverage["road_id"].is_unique),
            "all_statuses_assigned": result["mapping_status"].notna().all(),
            "spot_counts_sum_consistent": int(coverage["spot_count"].sum()) ==
                len(matched[["road_id", "spot_id"]].drop_duplicates()),
        },
        "notice": "Coverage is structural (linked spots), not verified date/time observation coverage.",
    }
    return result, coverage, report


# [서연] Generate two versioned read-only tables in one command.
def execute_official_road_mapping(project_root: Path, data_root: Path) -> int:
    volume_paths = select_volume_files(derived_of(data_root) / "topis_traffic_volume")
    stable_path = project_root / STABLE_LINKS_RELATIVE
    if not stable_path.is_file():
        raise FileNotFoundError(f"Missing stable links: {stable_path}")
    volume = pd.concat([pd.read_parquet(volume_paths["train"]),
                        pd.read_parquet(volume_paths["eval"])], ignore_index=True)
    stable = pd.read_csv(stable_path, dtype={"road_id": "string", "link_id": "string"},
                         encoding="utf-8-sig")
    mapping, coverage, report = build_official_road_mapping(volume, stable)
    if not all(report["validation_checks"].values()):
        raise ValueError(f"Official mapping validation failed: {report['validation_checks']}")
    expected = {"coordinate_versions": 649, "roads_with_spots": 52,
                "roads_without_spots": 38, "distinct_matched_spots": 84}
    expected_status = {"matched": 397, "out_of_scope_not_in_90_roads": 252}
    expected_distribution = {"1": 31, "2": 13, "3": 6, "4": 1, "5": 1}
    report["expected_reference_checks"] = {
        **{k: report[k] == v for k, v in expected.items()},
        "status_counts": report["mapping_status_counts"] == expected_status,
        "spot_count_distribution": report["spot_count_distribution"] == expected_distribution,
    }
    if not all(report["expected_reference_checks"].values()):
        raise ValueError(f"Reference counts changed; inspect source before exporting: {report['expected_reference_checks']}")
    output_dir = project_root / "outputs" / "tables"
    log_dir = project_root / "logs" / "ysy" / "volume_road_mapping"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    v = 1
    while True:
        suffix = f"{stamp}_(성공)_v{v:02d}"
        map_path = output_dir / f"yang_seoyeon_road_name_mapping_{suffix}.csv"
        coverage_path = output_dir / f"yang_seoyeon_traffic_coverage_{suffix}.csv"
        log_path = log_dir / f"official_road_mapping_{suffix}.json"
        if not any(p.exists() for p in (map_path, coverage_path, log_path)):
            break
        v += 1
    report["source_files"] = {"train": str(volume_paths["train"]), "eval": str(volume_paths["eval"]),
                              "stable_links": str(stable_path)}
    report["outputs"] = {"mapping": str(map_path), "coverage": str(coverage_path), "validation": str(log_path)}
    with map_path.open("x", encoding="utf-8-sig", newline="") as handle:
        mapping.to_csv(handle, index=False)
    with coverage_path.open("x", encoding="utf-8-sig", newline="") as handle:
        coverage.to_csv(handle, index=False)
    with log_path.open("x", encoding="utf-8") as handle:
        json.dump(json_value(report), handle, ensure_ascii=False, indent=2)
    print(f"Coordinate versions: {report['coordinate_versions']}; status: {report['mapping_status_counts']}")
    print(f"Roads with spots: {report['roads_with_spots']}; without spots: {report['roads_without_spots']}")
    print(f"Distinct matched spots: {report['distinct_matched_spots']}")
    print(f"Mapping CSV: {map_path}")
    print(f"Coverage CSV: {coverage_path}")
    print(f"Validation JSON: {log_path}")
    return 0


# [서연] Join original observations directly by official road name; never multiply by coordinate versions.
def build_official_observation_tables(volume: pd.DataFrame, stable: pd.DataFrame
                                      ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {"date", "spot_id", "io_type", "road_name", "traffic_volume"}
    if required - set(volume.columns):
        raise ValueError(f"Missing observation columns: {sorted(required - set(volume.columns))}")
    # [서연] Require an explicit clock column before claiming date-hour uniqueness.
    time_columns = [c for c in ("hour", "time", "time_slot", "hour_slot", "hour_label") if c in volume]
    if len(time_columns) != 1:
        raise ValueError(f"Expected exactly one known time column; found {time_columns}. "
                         f"Available columns: {list(volume.columns)}")
    time_col = time_columns[0]
    reference = stable[["road_id", "road_name"]].drop_duplicates().copy()
    reference["normalized_road_name"] = reference["road_name"].map(normalized_road_name)
    if reference.groupby("normalized_road_name")["road_id"].nunique().gt(1).any():
        raise ValueError("Reference normalized road-name collision")
    if reference.groupby("road_id")["normalized_road_name"].nunique().gt(1).any():
        raise ValueError("Reference road ID has multiple names")
    reference = reference.drop_duplicates("road_id")
    if len(reference) != 90:
        raise ValueError("Expected 90 reference roads")
    source = volume.copy().reset_index(drop=True)
    source["source_row_id"] = range(len(source))
    source["normalized_road_name"] = source["road_name"].map(normalized_road_name)
    result = source.merge(reference[["normalized_road_name", "road_id"]],
                          on="normalized_road_name", how="left",
                          validate="many_to_one", sort=False, indicator=True)
    result["mapping_status"] = "out_of_scope_not_in_90_roads"
    result.loc[result["_merge"].eq("both"), "mapping_status"] = "matched"
    result.loc[result["normalized_road_name"].eq(""), "mapping_status"] = "invalid_or_ambiguous"
    result["mapping_method"] = "not_applicable"
    result.loc[result["mapping_status"].eq("matched"), "mapping_method"] = "official_road_name_exact"
    result = result.drop(columns=["_merge"]).sort_values("source_row_id")
    if len(result) != len(source) or not result["source_row_id"].is_unique:
        raise ValueError("Observation join multiplied or removed rows")
    matched = result.loc[result["mapping_status"].eq("matched")].copy()
    key = ["date", time_col, "road_id", "spot_id", "io_type"]
    duplicate_keys = int(matched.duplicated(key, keep=False).sum())
    # [서연] Duplicate date-hour keys are reported, not silently aggregated or discarded.
    matched["traffic_volume_numeric"] = pd.to_numeric(matched["traffic_volume"], errors="coerce")
    matched["valid_traffic_volume"] = matched["traffic_volume_numeric"].notna() & matched["traffic_volume_numeric"].ge(0)
    matched["parsed_date"] = pd.to_datetime(matched["date"], errors="coerce")
    if matched["parsed_date"].isna().any():
        raise ValueError("Unparseable dates in matched observations")
    agg = matched.groupby("road_id", dropna=False).agg(
        observation_count=("source_row_id", "size"),
        valid_observation_count=("valid_traffic_volume", "sum"),
        first_observation_date=("parsed_date", "min"),
        last_observation_date=("parsed_date", "max"),
    )
    agg["missing_rate"] = 1 - agg["valid_observation_count"] / agg["observation_count"]
    roads = reference.copy()
    roads = roads.merge(agg, left_on="road_id", right_index=True, how="left", validate="one_to_one")
    roads["has_traffic_observations"] = roads["observation_count"].notna()
    roads["observation_count"] = roads["observation_count"].fillna(0).astype(int)
    roads["valid_observation_count"] = roads["valid_observation_count"].fillna(0).astype(int)
    roads["analysis_eligible"] = roads["valid_observation_count"].gt(0)
    spot_counts = matched.groupby("road_id")["spot_id"].nunique()
    roads["spot_count"] = roads["road_id"].map(spot_counts).fillna(0).astype(int)
    for c in ("first_observation_date", "last_observation_date"):
        roads[c] = roads[c].dt.strftime("%Y-%m-%d")
    report = {
        "source_observation_rows": len(source),
        "joined_observation_rows": len(result),
        "matched_observation_rows": len(matched),
        "time_column": time_col,
        "duplicate_date_time_road_spot_direction_rows": duplicate_keys,
        "mapping_status_counts": {str(k): int(v) for k, v in result["mapping_status"].value_counts().items()},
        "roads_with_observations": int(roads["has_traffic_observations"].sum()),
        "roads_with_valid_observations": int(roads["analysis_eligible"].sum()),
        "validation_checks": {
            "row_count_preserved": len(result) == len(source),
            "source_row_ids_unique": bool(result["source_row_id"].is_unique),
            "reference_road_ids_unique": bool(roads["road_id"].is_unique),
            "matched_road_ids_present": bool(matched["road_id"].notna().all()),
            "road_count_90": len(roads) == 90,
        },
        "notice": "Missing rate is among present observation rows; expected time-grid coverage is not calculated. "
                  "Duplicate observation keys are not automatically resolved.",
    }
    return result, roads, report


# [서연] Read-only observation-level join and observed-row coverage, versioned outputs.
def execute_official_observation_join(project_root: Path, data_root: Path) -> int:
    inputs = select_volume_files(derived_of(data_root) / "topis_traffic_volume")
    stable_path = project_root / STABLE_LINKS_RELATIVE
    volume = pd.concat([pd.read_parquet(inputs["train"]), pd.read_parquet(inputs["eval"])],
                       ignore_index=True)
    stable = pd.read_csv(stable_path, dtype={"road_id": "string"}, encoding="utf-8-sig")
    observations, coverage, report = build_official_observation_tables(volume, stable)
    if not all(report["validation_checks"].values()):
        raise ValueError(f"Observation join failed validation: {report['validation_checks']}")
    # [서연] Preserve duplicate-key evidence; do not claim unique hourly observations.
    output_dir = project_root / "outputs" / "tables"
    log_dir = project_root / "logs" / "ysy" / "volume_road_mapping"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    version = 1
    while True:
        suffix = f"{stamp}_(성공)_v{version:02d}"
        observation_path = output_dir / f"yang_seoyeon_official_traffic_observations_{suffix}.csv"
        coverage_path = output_dir / f"yang_seoyeon_observation_coverage_{suffix}.csv"
        log_path = log_dir / f"observation_join_{suffix}.json"
        if not any(p.exists() for p in (observation_path, coverage_path, log_path)):
            break
        version += 1
    report["input_files"] = {"train": str(inputs["train"]), "eval": str(inputs["eval"]),
                             "stable_links": str(stable_path)}
    report["output_files"] = {"observations": str(observation_path),
                              "coverage": str(coverage_path), "validation": str(log_path)}
    with observation_path.open("x", encoding="utf-8-sig", newline="") as f:
        observations.to_csv(f, index=False)
    with coverage_path.open("x", encoding="utf-8-sig", newline="") as f:
        coverage.to_csv(f, index=False)
    with log_path.open("x", encoding="utf-8") as f:
        json.dump(json_value(report), f, ensure_ascii=False, indent=2)
    print(f"Observation rows: {report['source_observation_rows']}; matched: {report['matched_observation_rows']}")
    print(f"Time column: {report['time_column']}; duplicate key rows: {report['duplicate_date_time_road_spot_direction_rows']}")
    print(f"Roads with observations: {report['roads_with_observations']}; with valid values: {report['roads_with_valid_observations']}")
    print(f"Observations CSV: {observation_path}")
    print(f"Coverage CSV: {coverage_path}")
    print(f"Validation JSON: {log_path}")
    return 0
