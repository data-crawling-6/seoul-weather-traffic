"""[서연] ② 지점별 TOPIS 링크 후보 만들기 · 후보 진단.

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

from .audit import audit_link_coverage
from src.io_utils import derived_of
from .common import LINES_FILENAME, STABLE_LINKS_RELATIVE, audit_provenance, build_coordinate_review, input_file_provenance, json_value, normalize_link_ids, normalized_road_name, select_volume_files


# [서연] Conservative, explicit road-level matching thresholds (metres).
MAP_CANDIDATE_LIMIT = 3


MAP_AUTO_MAX_DISTANCE_M = 50.0


MAP_FAR_DISTANCE_M = 200.0


MAP_LARGE_COORDINATE_CHANGE_M = 100.0


MAP_AUTO_MAX_COORDINATE_CHANGE_M = 50.0


# [서연] Preserve every coordinate version, including invalid/zero/missing versions.
def build_mapping_versions(volume: pd.DataFrame) -> pd.DataFrame:
    required = {"spot_id", "io_type", "latitude", "longitude", "road_name"}
    if required - set(volume.columns):
        raise ValueError(f"Traffic volume missing columns: {sorted(required - set(volume.columns))}")
    columns = ["spot_id", "io_type", "latitude", "longitude", "road_name"]
    if "spot_name" in volume:
        columns.append("spot_name")
    if "direction_detail" in volume:
        columns.append("direction_detail")
    versions = volume[columns].copy()
    versions["latitude"] = pd.to_numeric(versions["latitude"], errors="coerce")
    versions["longitude"] = pd.to_numeric(versions["longitude"], errors="coerce")
    # [서연] Distinct metadata and coordinate versions are preserved rather than arbitrarily merged.
    versions = versions.groupby(columns, dropna=False, sort=False).size().reset_index(name="source_rows")
    versions["coordinate_version_id"] = [
        f"V{i:05d}" for i in range(1, len(versions) + 1)
    ]
    versions["valid_coordinate"] = (
        versions["latitude"].between(33, 39) & versions["longitude"].between(124, 132)
    )
    return versions


# [서연] Prepare only the 1,753 stable analysis links, rejecting duplicate or missing geometry.
def prepare_mapping_links(stable: pd.DataFrame, lines: Any) -> Any:
    import geopandas as gpd

    checks = audit_link_coverage(stable, lines)
    if (
        checks["duplicate_stable_link_rows"] or checks["duplicate_geojson_link_rows"]
        or checks["links_assigned_to_multiple_roads"] or checks["unmatched_analysis_links"]
        or checks["missing_or_empty_geometries"] or checks["non_linestring_geometries"]
        or not checks["expected_1753_links"] or not checks["expected_90_roads"]
    ):
        raise ValueError(f"Unsafe mapping link inputs: {checks}")
    if lines.crs is None:
        raise ValueError("GeoJSON CRS missing")
    stable_copy = stable.copy()
    stable_copy["link_id"] = normalize_link_ids(stable_copy["link_id"], "stable links")
    geometry = lines[["link_id", "geometry"]].copy()
    geometry["link_id"] = normalize_link_ids(geometry["link_id"], "geometry")
    merged = stable_copy.merge(geometry, on="link_id", validate="one_to_one")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=lines.crs).to_crs(epsg=5179)


# [서연] Calculate up to three nearest candidates for each original coordinate version.
def generate_mapping_candidates(
    volume: pd.DataFrame, stable: pd.DataFrame, lines: Any
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import geopandas as gpd
    import numpy as np
    import shapely

    versions = build_mapping_versions(volume)
    review = build_coordinate_review(volume)
    roads = prepare_mapping_links(stable, lines)
    review_key = review.set_index(["spot_id", "io_type"])
    road_geoms = roads.geometry.to_numpy()
    candidate_records: list[dict[str, Any]] = []
    final_records: list[dict[str, Any]] = []

    for version in versions.to_dict("records"):
        key = (version["spot_id"], version["io_type"])
        pair = review_key.loc[key]
        change_m = float(pair["max_coordinate_distance_m"])
        invalid_in_pair = int(pair["invalid_coordinate_rows"]) > 0
        valid = bool(version["valid_coordinate"])
        candidates: list[dict[str, Any]] = []

        if valid:
            point = gpd.GeoSeries(
                gpd.points_from_xy([version["longitude"]], [version["latitude"]]),
                crs="EPSG:4326",
            ).to_crs(epsg=5179).iloc[0]
            distances = shapely.distance(road_geoms, point)
            order = sorted(
                range(len(roads)),
                key=lambda idx: (float(distances[idx]), str(roads.iloc[idx]["link_id"])),
            )[:MAP_CANDIDATE_LIMIT]
            for rank, idx in enumerate(order, 1):
                link = roads.iloc[idx]
                name_match = (
                    bool(normalized_road_name(version["road_name"]))
                    and normalized_road_name(version["road_name"])
                    == normalized_road_name(link["road_name"])
                )
                candidates.append({
                    "candidate_rank": rank,
                    "link_id": str(link["link_id"]),
                    "road_id": str(link["road_id"]),
                    "link_road_name": str(link["road_name"]),
                    "link_direction": str(link["direction"]),
                    "distance_m": round(float(distances[idx]), 3),
                    "road_name_match": bool(name_match),
                    # [서연] In/out is not interchangeable with up/down.
                    "direction_comparison": "not_comparable_without_verified_crosswalk",
                })

        # [서연] Road-level auto confirmation only; direction is never automatically confirmed.
        # [서연] Require close, exact-name, unambiguous top-3 road and stable coordinates.
        auto = (
            bool(candidates)
            and candidates[0]["distance_m"] <= MAP_AUTO_MAX_DISTANCE_M
            and candidates[0]["road_name_match"]
            and len(candidates) == MAP_CANDIDATE_LIMIT
            and all(c["road_id"] == candidates[0]["road_id"] for c in candidates)
            and change_m <= MAP_AUTO_MAX_COORDINATE_CHANGE_M
            and not invalid_in_pair
        )
        status = "auto_confirmed_road_only" if auto else (
            "manual_review_needed" if candidates else "unmapped"
        )
        reasons = []
        if not valid:
            reasons.append("invalid_coordinate")
        if valid and not candidates:
            reasons.append("no_candidate_link")
        if candidates and candidates[0]["distance_m"] > MAP_FAR_DISTANCE_M:
            reasons.append("far_from_analysis_links")
        if candidates and not candidates[0]["road_name_match"]:
            reasons.append("nearest_road_name_mismatch")
        if change_m > MAP_LARGE_COORDINATE_CHANGE_M:
            reasons.append("large_coordinate_change")
        if change_m > MAP_AUTO_MAX_COORDINATE_CHANGE_M:
            reasons.append("coordinate_change_exceeds_auto_threshold")
        if invalid_in_pair:
            reasons.append("invalid_coordinate_in_spot_direction")
        if candidates and len({c["road_id"] for c in candidates}) > 1:
            reasons.append("multiple_nearby_roads")
        if candidates and len(candidates) < MAP_CANDIDATE_LIMIT:
            reasons.append("fewer_than_three_candidates")
        if not auto and candidates and not reasons:
            reasons.append("auto_confirmation_conditions_not_met")

        common = {
            **version,
            "max_coordinate_change_m": round(change_m, 3),
            "invalid_rows_in_spot_direction": int(pair["invalid_coordinate_rows"]),
            "mapping_status": status,
            "review_reasons": ";".join(reasons),
        }
        for candidate in candidates:
            candidate_records.append({**common, **candidate})
        best = candidates[0] if candidates else {}
        final_records.append({
            **common,
            "selected_link_id": best.get("link_id") if auto else None,
            "selected_road_id": best.get("road_id") if auto else None,
            "nearest_candidate_link_id": best.get("link_id"),
            "nearest_candidate_road_id": best.get("road_id"),
            "nearest_distance_m": best.get("distance_m"),
            "nearest_road_name_match": best.get("road_name_match"),
            "nearest_link_direction": best.get("link_direction"),
            "direction_comparison": "not_comparable_without_verified_crosswalk",
            "candidate_count": len(candidates),
        })
    return pd.DataFrame(candidate_records), pd.DataFrame(final_records)


# [서연] Allocate timestamped output paths without overwriting audits or previous maps.
def allocate_map_paths(project_root: Path, stamp: str, success: bool) -> dict[str, Path]:
    logs = project_root / "logs" / "ysy" / "volume_road_mapping"
    tables = project_root / "outputs" / "tables"
    logs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    status = "성공" if success else "실패"
    for index in range(1, 1000):
        unique = stamp if index == 1 else f"{stamp}_{index:03d}"
        paths = {
            "json": logs / f"mapping_run_{unique}_({status}).json",
            "candidates": tables / f"yang_seoyeon_mapping_candidates_{unique}_({status})_v01.csv",
            "review": tables / f"yang_seoyeon_mapping_final_review_{unique}_({status})_v01.csv",
        }
        if not any(path.exists() for path in paths.values()):
            return paths
    raise FileExistsError("Cannot allocate unique map output paths")


# [서연] Load source files only when the map command is actually invoked.
def run_mapping(project_root: Path, data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    import geopandas as gpd

    volume_paths = select_volume_files(derived_of(data_root) / "topis_traffic_volume")
    paths = {
        **volume_paths,
        "stable_links": project_root / STABLE_LINKS_RELATIVE,
        "lines": derived_of(data_root) / "road_lines" / LINES_FILENAME,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing mapping input {name}: {path}")
    volume = pd.concat(
        [pd.read_parquet(volume_paths[split]) for split in ("train", "eval")],
        ignore_index=True,
    )
    stable = pd.read_csv(paths["stable_links"], dtype={"link_id": "string"}, encoding="utf-8-sig")
    lines = gpd.read_file(paths["lines"])
    candidates, review = generate_mapping_candidates(volume, stable, lines)
    return candidates, review, paths


# [서연] Audit every decision and count multi-label review reasons without changing rows.
def summarize_mapping_evidence(candidates: pd.DataFrame, review: pd.DataFrame) -> dict[str, Any]:
    from collections import Counter

    required = {"coordinate_version_id", "spot_id", "io_type", "source_rows", "mapping_status", "candidate_count", "review_reasons", "selected_road_id", "selected_link_id"}
    if required - set(review.columns):
        raise ValueError(f"Missing review columns: {sorted(required - set(review.columns))}")
    expected_statuses = {"auto_confirmed_road_only", "manual_review_needed", "unmapped"}
    unknown = set(review["mapping_status"].dropna().unique()) - expected_statuses
    if unknown or review["mapping_status"].isna().any():
        raise ValueError(f"Unknown/missing mapping status: {unknown}")
    reasons = Counter()
    reason_combinations = Counter()
    manual_reasons = Counter()
    for status, value in zip(review["mapping_status"], review["review_reasons"].fillna("")):
        tokens = sorted({item.strip() for item in str(value).split(";") if item.strip()})
        if tokens:
            reason_combinations[";".join(tokens)] += 1
            reasons.update(tokens)
            if status == "manual_review_needed":
                manual_reasons.update(tokens)
    statuses = {key: int(review["mapping_status"].eq(key).sum()) for key in sorted(expected_statuses)}
    per_version = candidates.groupby("coordinate_version_id").size() if len(candidates) else pd.Series(dtype="int64")
    review_ids = review["coordinate_version_id"].astype(str)
    candidate_ids = candidates["coordinate_version_id"].astype(str) if len(candidates) else pd.Series(dtype="string")
    candidate_count = review.set_index("coordinate_version_id")["candidate_count"].astype(int)
    actual_counts = per_version.reindex(candidate_count.index, fill_value=0)
    auto = review["mapping_status"].eq("auto_confirmed_road_only")
    not_auto = ~auto
    checks = {
        "review_version_ids_unique": bool(review_ids.is_unique),
        "candidate_versions_exist_in_review": bool(set(candidate_ids).issubset(set(review_ids))),
        "candidate_count_matches_review": bool(actual_counts.eq(candidate_count).all()),
        "candidate_count_within_limit": bool(candidate_count.between(0, MAP_CANDIDATE_LIMIT).all()),
        "candidate_ranks_unique_per_version": bool(not len(candidates) or not candidates.duplicated(["coordinate_version_id", "candidate_rank"]).any()),
        "status_partition_equals_versions": sum(statuses.values()) == len(review),
        "source_rows_positive": bool(pd.to_numeric(review["source_rows"], errors="coerce").gt(0).all()),
        "only_auto_confirmed_have_selected_links": bool(review.loc[not_auto, ["selected_link_id", "selected_road_id"]].isna().all().all() and review.loc[auto, ["selected_link_id", "selected_road_id"]].notna().all().all()),
        "unmapped_have_zero_candidates": bool(review.loc[review["mapping_status"].eq("unmapped"), "candidate_count"].eq(0).all()),
        "manual_review_has_reasons": bool(review.loc[review["mapping_status"].eq("manual_review_needed"), "review_reasons"].fillna("").str.len().gt(0).all()),
    }
    return {
        "review_reasons_multi_label": dict(sorted(reasons.items())),
        "manual_review_reasons_multi_label": dict(sorted(manual_reasons.items())),
        "review_reason_combinations": dict(sorted(reason_combinations.items())),
        "status_counts": statuses,
        "validation_checks": checks,
        "validation_passed": all(checks.values()),
        "validation_note": "Review reasons are multi-label; their counts can exceed manual-review versions.",
    }


# [서연] Build a reproducible processing-stage ledger for the current run.
def mapping_processing_steps() -> list[dict[str, str]]:
    return [
        {"step": "load_inputs", "rule": "Latest successful train/eval pair, stable_links v02, GeoJSON; never link-hourly parquet"},
        {"step": "coordinate_versions", "rule": "Preserve spot_id, io_type, coordinates, road name and direction metadata; invalid versions retained"},
        {"step": "candidate_search", "rule": "Project to EPSG:5179; distance to 1753 analysis link geometries; keep top 3 by distance and link ID"},
        {"step": "comparison", "rule": "Exact road name after whitespace removal; direction is not automatically crosswalked"},
        {"step": "decision", "rule": "Apply thresholds and all top-3 road agreement; never force unmapped/manual versions"},
        {"step": "evidence_validation", "rule": "Verify version coverage, candidate counts, rank uniqueness, status partition, reasons and selection consistency"},
        {"step": "persist", "rule": "Create unique timestamped CSV and JSON using exclusive file creation; preserve previous runs"},
    ]


# [서연] Record a fresh map run and immutable evidence, even if a run fails.
def execute_map(project_root: Path, data_root: Path) -> int:
    started = datetime.now().astimezone()
    candidates = None
    review = None
    error = None
    paths: dict[str, Path] = {
        "stable_links": project_root / STABLE_LINKS_RELATIVE,
        "lines": derived_of(data_root) / "road_lines" / LINES_FILENAME,
    }
    try:
        paths.update(select_volume_files(derived_of(data_root) / "topis_traffic_volume"))
        candidates, review, paths = run_mapping(project_root, data_root)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    # [서연] Validate the evidence before declaring the mapping run successful.
    evidence = None
    if error is None and candidates is not None and review is not None:
        try:
            evidence = summarize_mapping_evidence(candidates, review)
            if not evidence["validation_passed"]:
                error = f"MappingEvidenceValidationError: {evidence['validation_checks']}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    success = error is None
    finished = datetime.now().astimezone()
    summary = {
        "run_kind": "fresh_mapping_execution_not_historical_reconstruction",
        "status": "success" if success else "failure",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "record_notice": "Fresh mapping run executed now; not a reconstruction of earlier runs.",
        "error": error,
        "processing_steps": mapping_processing_steps(),
        "mapping_evidence": evidence,
        "thresholds": {
            "max_candidates": MAP_CANDIDATE_LIMIT,
            "auto_max_distance_m": MAP_AUTO_MAX_DISTANCE_M,
            "far_distance_m": MAP_FAR_DISTANCE_M,
            "large_coordinate_change_m": MAP_LARGE_COORDINATE_CHANGE_M,
            "auto_max_coordinate_change_m": MAP_AUTO_MAX_COORDINATE_CHANGE_M,
            "auto_rule": "nearest <= 50m, exact road name, all top-3 links on same road, coordinate change <= 50m, no invalid coordinates in pair; road only, not direction",
            "direction_rule": "io_type and link direction are not assumed equivalent",
        },
        "provenance": audit_provenance(project_root, paths),
        "counts": {
            "traffic_rows": None,
            "coordinate_versions": int(len(review)) if review is not None else None,
            "candidate_rows": int(len(candidates)) if candidates is not None else None,
            "auto_confirmed": int(review["mapping_status"].eq("auto_confirmed_road_only").sum()) if review is not None else None,
            "manual_review_needed": int(review["mapping_status"].eq("manual_review_needed").sum()) if review is not None else None,
            "unmapped": int(review["mapping_status"].eq("unmapped").sum()) if review is not None else None,
            "candidate_spot_directions": int(candidates[["spot_id", "io_type"]].drop_duplicates().shape[0]) if candidates is not None and len(candidates) else 0,
        },
    }
    if review is not None:
        summary["counts"]["traffic_rows"] = int(review["source_rows"].sum())
    outputs = allocate_map_paths(project_root, started.strftime("%Y%m%dT%H%M%S"), success)
    if candidates is not None:
        with outputs["candidates"].open("x", encoding="utf-8-sig", newline="") as handle:
            candidates.to_csv(handle, index=False)
    if review is not None:
        with outputs["review"].open("x", encoding="utf-8-sig", newline="") as handle:
            review.to_csv(handle, index=False)
    # [서연] Fingerprint newly written outputs for traceable evidence files.
    summary["output_artifacts"] = {
        key: input_file_provenance(path) if path.exists() else None
        for key, path in outputs.items() if key != "json"
    }
    summary["output_files"] = {
        key: str(path) if (key == "json" or path.exists()) else None
        for key, path in outputs.items()
    }
    with outputs["json"].open("x", encoding="utf-8") as handle:
        json.dump(json_value(summary), handle, ensure_ascii=False, indent=2)
    print(f"Map status: {summary['status']}")
    print(f"Map JSON: {outputs['json']}")
    print(f"Candidate CSV: {summary['output_files']['candidates']}")
    print(f"Review CSV: {summary['output_files']['review']}")
    if error:
        print(error, file=sys.stderr)
    return 0 if success else 1


# [서연] Analyze saved candidates without changing mapping decisions or source rows.
def analyze_candidate_frame(
    candidates: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    required = {
        "coordinate_version_id", "candidate_rank", "road_name_match",
        "spot_id", "io_type", "road_id", "road_name", "link_road_name",
        "link_id", "distance_m", "mapping_status", "review_reasons",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate CSV missing columns: {missing}")
    frame = candidates.copy()
    frame["candidate_rank"] = pd.to_numeric(frame["candidate_rank"], errors="coerce")
    frame["distance_m"] = pd.to_numeric(frame["distance_m"], errors="coerce")
    if frame[["candidate_rank", "distance_m"]].isna().any().any():
        raise ValueError("Candidate rank or distance contains non-numeric values")
    if frame["candidate_rank"].lt(1).any() or frame["candidate_rank"].gt(MAP_CANDIDATE_LIMIT).any():
        raise ValueError("Candidate rank is outside the configured limit")
    if frame["distance_m"].lt(0).any():
        raise ValueError("Candidate distance cannot be negative")
    if frame.duplicated(["coordinate_version_id", "candidate_rank"]).any():
        raise ValueError("Duplicate coordinate-version candidate ranks")
    top = frame.loc[frame["candidate_rank"].eq(1)].copy()
    if top["coordinate_version_id"].duplicated().any():
        raise ValueError("Duplicate rank-1 candidate")

    def as_boolean(series: pd.Series) -> pd.Series:
        return series.astype("string").str.strip().str.lower().eq("true")

    frame["road_name_match"] = as_boolean(frame["road_name_match"])
    top["road_name_match"] = as_boolean(top["road_name_match"])
    mismatches = top.loc[~top["road_name_match"]].copy()
    alternatives = frame.loc[
        frame["candidate_rank"].ne(1)
        & frame["road_name_match"]
        & frame["distance_m"].le(MAP_AUTO_MAX_DISTANCE_M)
    ]
    alternative_ids = set(alternatives["coordinate_version_id"].astype(str))
    mismatch_ids = set(mismatches["coordinate_version_id"].astype(str))
    multiple_roads = frame.groupby("coordinate_version_id")["road_id"].nunique().gt(1)

    mismatch_pairs = (
        mismatches.groupby(["road_name", "link_road_name"], dropna=False)
        .size().reset_index(name="coordinate_versions")
        .sort_values(["coordinate_versions", "road_name", "link_road_name"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    detail_columns = [
        "coordinate_version_id", "spot_id", "io_type", "road_name",
        "link_road_name", "link_id", "road_id", "distance_m",
        "mapping_status", "review_reasons",
    ]
    detail = mismatches[detail_columns].copy()
    detail["alternative_matching_candidate_within_50m"] = (
        detail["coordinate_version_id"].astype(str).isin(alternative_ids)
    )
    detail["multiple_candidate_roads"] = (
        detail["coordinate_version_id"].map(multiple_roads).fillna(False).astype(bool)
    )
    summary = {
        "candidate_rows": int(len(frame)),
        "candidate_coordinate_versions": int(frame["coordinate_version_id"].nunique()),
        "nearest_road_name_mismatch_versions": int(len(mismatches)),
        "nearest_road_name_mismatch_within_50m": int(mismatches["distance_m"].le(MAP_AUTO_MAX_DISTANCE_M).sum()),
        "nearest_mismatch_with_alternative_matching_candidate_within_50m": int(len(mismatch_ids & alternative_ids)),
        "multiple_candidate_roads_versions": int(multiple_roads.sum()),
    }
    return summary, mismatch_pairs, detail


# [서연] Locate only a completed successful candidate artifact when no path is supplied.
def select_candidate_file(project_root: Path) -> Path:
    tables = project_root / "outputs" / "tables"
    paths = sorted(tables.glob("yang_seoyeon_mapping_candidates_*_(성공)_v01.csv"))
    if not paths:
        raise FileNotFoundError(f"Successful mapping candidate CSV not found: {tables}")
    return paths[-1]


# [서연] Allocate unique candidate-diagnostic outputs without overwriting prior evidence.
def allocate_candidate_analysis_paths(
    project_root: Path, stamp: str, success: bool
) -> dict[str, Path]:
    logs = project_root / "logs" / "ysy" / "volume_road_mapping"
    tables = project_root / "outputs" / "tables"
    logs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    status = "성공" if success else "실패"
    for index in range(1, 1000):
        unique = stamp if index == 1 else f"{stamp}_{index:03d}"
        paths = {
            "json": logs / f"candidate_diagnostics_{unique}_({status}).json",
            "pairs": tables / f"yang_seoyeon_road_name_mismatch_pairs_{unique}_({status})_v01.csv",
            "review": tables / f"yang_seoyeon_road_name_mismatch_review_{unique}_({status})_v01.csv",
        }
        if not any(path.exists() for path in paths.values()):
            return paths
    raise FileExistsError("Cannot allocate unique candidate diagnostic paths")


# [서연] Persist a fresh read-only candidate diagnosis, including failures, as new evidence.
def execute_candidate_analysis(project_root: Path, candidates_path: Path | None = None) -> int:
    started = datetime.now().astimezone()
    source = candidates_path or select_candidate_file(project_root)
    summary_counts = None
    mismatch_pairs = None
    mismatch_review = None
    error = None
    try:
        candidates = pd.read_csv(source, dtype={"link_id": "string", "road_id": "string"})
        summary_counts, mismatch_pairs, mismatch_review = analyze_candidate_frame(candidates)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    success = error is None
    outputs = allocate_candidate_analysis_paths(
        project_root, started.strftime("%Y%m%dT%H%M%S"), success
    )
    if mismatch_pairs is not None:
        with outputs["pairs"].open("x", encoding="utf-8-sig", newline="") as handle:
            mismatch_pairs.to_csv(handle, index=False)
    if mismatch_review is not None:
        with outputs["review"].open("x", encoding="utf-8-sig", newline="") as handle:
            mismatch_review.to_csv(handle, index=False)
    report = {
        "run_kind": "fresh_read_only_candidate_diagnostics",
        "status": "success" if success else "failure",
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "record_notice": "Candidate diagnostics do not change mapping statuses or selected links.",
        "error": error,
        "thresholds": {"alternative_name_match_max_distance_m": MAP_AUTO_MAX_DISTANCE_M},
        "counts": summary_counts,
        "provenance": audit_provenance(project_root, {"candidates": source}),
        "notes": [
            "Review reasons are not modified.",
            "Unmapped versions without candidates remain in the mapping final-review CSV, not the candidate CSV.",
        ],
        "output_files": {
            "json": str(outputs["json"]),
            "mismatch_pairs": str(outputs["pairs"]) if outputs["pairs"].exists() else None,
            "mismatch_review": str(outputs["review"]) if outputs["review"].exists() else None,
        },
    }
    report["output_artifacts"] = {
        key: input_file_provenance(path) if path.exists() else None
        for key, path in outputs.items() if key != "json"
    }
    with outputs["json"].open("x", encoding="utf-8") as handle:
        json.dump(json_value(report), handle, ensure_ascii=False, indent=2)
    print(f"Candidate analysis status: {report['status']}")
    print(f"Candidate analysis JSON: {outputs['json']}")
    print(f"Mismatch pairs CSV: {report['output_files']['mismatch_pairs']}")
    print(f"Mismatch review CSV: {report['output_files']['mismatch_review']}")
    if error:
        print(error, file=sys.stderr)
    return 0 if success else 1
