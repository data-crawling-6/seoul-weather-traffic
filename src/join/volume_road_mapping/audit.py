"""[서연] ① 매핑 입력 점검 (좌표 · 링크 범위 · 출처).

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
from .common import LINES_FILENAME, STABLE_LINKS_RELATIVE, VERTEX_FILENAME, audit_provenance, build_coordinate_review, json_value, normalize_link_ids, select_volume_files


# [서연] Validate Vertex column variants without inventing missing coordinates.
def audit_vertices(vertex: pd.DataFrame) -> dict[str, Any]:
    cols = set(vertex.columns)
    if "link_id" not in cols:
        raise ValueError("Vertex CSV missing link_id")
    coordinate_pairs = [
        ("grs80tm_x", "grs80tm_y"), ("x", "y"), ("X", "Y"), ("vertex_x", "vertex_y"),
        ("longitude", "latitude"), ("lon", "lat"), ("coord_x", "coord_y"),
    ]
    pair = next((pair for pair in coordinate_pairs if set(pair) <= cols), None)
    if pair is None:
        raise ValueError(f"Cannot identify Vertex coordinate columns: {sorted(cols)}")
    x, y = pair
    missing = vertex[[x, y]].isna().any(axis=1)
    numeric = vertex[[x, y]].apply(pd.to_numeric, errors="coerce")
    invalid_numeric = numeric.isna().any(axis=1)
    # [서연] Validate vertex_order existence, numeric integrity, and ordering uniqueness per link.
    if "vertex_order" not in cols:
        raise ValueError("Vertex CSV missing vertex_order")
    orders = pd.to_numeric(vertex["vertex_order"], errors="coerce")
    missing_orders = int(vertex["vertex_order"].isna().sum())
    invalid_orders = int((orders.isna() | ~orders.map(lambda v: math.isfinite(v) if pd.notna(v) else False) | orders.ne(orders.round())).sum())
    ids = normalize_link_ids(vertex["link_id"], "Vertex CSV")
    duplicate_orders = int(pd.DataFrame({"link_id": ids, "vertex_order": orders}).loc[orders.notna()].duplicated().sum())
    counts = ids.value_counts()
    return {
        "rows": len(vertex),
        "unique_links": ids.nunique(),
        "coordinate_columns": [x, y],
        "missing_coordinate_rows": int(missing.sum()),
        "non_numeric_or_missing_coordinate_rows": int(invalid_numeric.sum()),
        "minimum_vertices_per_link": int(counts.min()) if len(counts) else None,
        "missing_vertex_order_rows": missing_orders,
        "invalid_vertex_order_rows": invalid_orders,
        "duplicate_vertex_order_within_link_rows": duplicate_orders,
        "links_with_fewer_than_two_vertices": int(counts.lt(2).sum()),
    }


# [서연] Validate the stable-link mapping against the complete LineString geometry.
def audit_link_coverage(stable: pd.DataFrame, lines: Any) -> dict[str, Any]:
    for col in ("link_id", "road_id", "road_name", "direction"):
        if col not in stable.columns:
            raise ValueError(f"Stable-links CSV missing {col}")
    if "link_id" not in lines.columns or "geometry" not in lines.columns:
        raise ValueError("GeoJSON missing link_id or geometry")

    stable_ids = normalize_link_ids(stable["link_id"], "stable-links CSV")
    line_ids = normalize_link_ids(lines["link_id"], "GeoJSON")
    stable_set = set(stable_ids)
    line_set = set(line_ids)
    road_count = stable["road_id"].nunique(dropna=True)
    road_assignments = stable.assign(_link_id=stable_ids).groupby("_link_id")["road_id"].nunique()
    invalid_geometry = lines.geometry.isna() | lines.geometry.is_empty
    wrong_geometry = lines.geometry.geom_type.ne("LineString")
    unmatched = sorted(stable_set - line_set)
    return {
        "stable_link_rows": len(stable),
        "stable_unique_links": len(stable_set),
        "analysis_roads": int(road_count),
        "geojson_rows": len(lines),
        "geojson_unique_links": len(line_set),
        "matched_analysis_links": len(stable_set & line_set),
        "unmatched_analysis_links": len(unmatched),
        "unmatched_link_ids": unmatched,
        "duplicate_stable_link_rows": int(stable_ids.duplicated().sum()),
        "duplicate_geojson_link_rows": int(line_ids.duplicated().sum()),
        "links_assigned_to_multiple_roads": int(road_assignments.gt(1).sum()),
        "missing_or_empty_geometries": int(invalid_geometry.sum()),
        "non_linestring_geometries": int(wrong_geometry.sum()),
        "expected_1753_links": len(stable_set) == 1753,
        "expected_90_roads": road_count == 90,
    }


# [서연] Read all four sources, recompute all audit metrics at the current execution time.
def run_input_audit(project_root: Path, data_root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    import geopandas as gpd

    volume_paths = select_volume_files(derived_of(data_root) / "topis_traffic_volume")
    stable_path = project_root / STABLE_LINKS_RELATIVE
    vertex_path = data_root / "link_coords" / VERTEX_FILENAME
    lines_path = derived_of(data_root) / "road_lines" / LINES_FILENAME
    paths = {**volume_paths, "stable_links": stable_path, "vertices": vertex_path, "lines": lines_path}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {key}: {path}")

    volume = pd.concat(
        [pd.read_parquet(volume_paths[split]) for split in ("train", "eval")],
        ignore_index=True,
    )
    for col in ("date", "spot_id", "traffic_volume", "latitude", "longitude"):
        if col not in volume.columns:
            raise ValueError(f"Traffic volume missing {col}")
    review = build_coordinate_review(volume)
    missing_volume = int(volume["traffic_volume"].isna().sum())
    zero_coordinates = int((
        pd.to_numeric(volume["latitude"], errors="coerce").eq(0)
        | pd.to_numeric(volume["longitude"], errors="coerce").eq(0)
    ).sum())
    changed = review[review["coordinate_changed"]]

    stable = pd.read_csv(stable_path, dtype={"link_id": "string"}, encoding="utf-8-sig")
    vertex = pd.read_csv(vertex_path, dtype={"link_id": "string"}, low_memory=False)
    lines = gpd.read_file(lines_path)
    report = {
        "audit_kind": "fresh_execution_recalculation_not_historical_reconstruction",
        "inputs": {name: str(path) for name, path in paths.items()},
        "traffic_volume": {
            "rows": len(volume),
            "unique_dates": volume["date"].nunique(dropna=True),
            "unique_spots": volume["spot_id"].nunique(dropna=True),
            "missing_traffic_volume_rows": missing_volume,
            "missing_traffic_volume_percent": round(missing_volume / len(volume) * 100, 6) if len(volume) else 0,
            "zero_latitude_or_longitude_rows": zero_coordinates,
            "spot_direction_pairs": len(review),
            "pairs_with_multiple_valid_coordinates": len(changed),
            "coordinate_count_distribution": {
                str(k): int(v) for k, v in review["valid_coordinate_count"].value_counts().sort_index().items()
            },
            "changed_spot_directions": changed[[
                "spot_id", "io_type", "valid_coordinate_count", "max_coordinate_distance_m"
            ]].to_dict("records"),
            "maximum_coordinate_change_m": float(review["max_coordinate_distance_m"].max()) if len(review) else 0,
        },
        "vertices": audit_vertices(vertex),
        "links": audit_link_coverage(stable, lines),
    }
    links = report["links"]
    checks = {
        "analysis_links_1753": links["expected_1753_links"],
        "analysis_roads_90": links["expected_90_roads"],
        "all_analysis_links_have_geometry": links["unmatched_analysis_links"] == 0,
        "stable_links_not_duplicated": links["duplicate_stable_link_rows"] == 0,
        "geometry_links_not_duplicated": links["duplicate_geojson_link_rows"] == 0,
        "no_cross_road_link_assignment": links["links_assigned_to_multiple_roads"] == 0,
        "valid_linestring_geometries": links["missing_or_empty_geometries"] == 0 and links["non_linestring_geometries"] == 0,
        "vertices_have_two_or_more_points": report["vertices"]["links_with_fewer_than_two_vertices"] == 0,
        "vertices_have_numeric_coordinates": report["vertices"]["non_numeric_or_missing_coordinate_rows"] == 0,
        "vertices_have_valid_vertex_order": report["vertices"]["invalid_vertex_order_rows"] == 0,
        "vertices_have_unique_order_per_link": report["vertices"]["duplicate_vertex_order_within_link_rows"] == 0,
    }
    report["checks"] = checks
    if not all(checks.values()):
        raise AuditValidationError("Input integrity checks failed", report, review)
    return report, review


# [서연] Preserve partial computed results when a validation check fails.
class AuditValidationError(Exception):
    def __init__(self, message: str, report: dict[str, Any], review: pd.DataFrame):
        super().__init__(message)
        self.report = report
        self.review = review


# [서연] Create new audit output files exclusively, with a unique timestamp if needed.
def allocate_output_paths(project_root: Path, stamp: str, success: bool) -> tuple[Path, Path]:
    logs = project_root / "logs" / "ysy" / "volume_road_mapping"
    tables = project_root / "outputs" / "tables"
    logs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    status = "성공" if success else "실패"
    for index in range(1, 1000):
        unique = stamp if index == 1 else f"{stamp}_{index:03d}"
        audit = logs / f"mapping_input_audit_{unique}_({status}).json"
        review = tables / f"yang_seoyeon_mapping_coordinate_review_{unique}_v01.csv"
        if not audit.exists() and not review.exists():
            return audit, review
    raise FileExistsError("Could not allocate unique audit output filenames")


# [서연] Persist fresh audit and coordinate review without overwriting prior runs.
def execute_audit(project_root: Path, data_root: Path) -> int:
    started = datetime.now().astimezone()
    report: dict[str, Any] = {}
    review: pd.DataFrame | None = None
    error = None
    try:
        report, review = run_input_audit(project_root, data_root)
    except AuditValidationError as exc:
        report, review = exc.report, exc.review
        error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    success = error is None
    finished = datetime.now().astimezone()
    # [서연] Fingerprint every discoverable source even when the input audit fails.
    known_paths = {
        "stable_links": project_root / STABLE_LINKS_RELATIVE,
        "vertices": data_root / "link_coords" / VERTEX_FILENAME,
        "lines": derived_of(data_root) / "road_lines" / LINES_FILENAME,
    }
    try:
        known_paths.update(select_volume_files(derived_of(data_root) / "topis_traffic_volume"))
    except (OSError, ValueError):
        pass
    report["provenance"] = audit_provenance(project_root, known_paths)
    report.update({
        "status": "success" if success else "failure",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "record_notice": "Fresh audit executed now; not a reconstruction of earlier runs.",
        "error": error,
    })
    audit_path, review_path = allocate_output_paths(
        project_root, started.strftime("%Y%m%dT%H%M%S"), success
    )
    if review is not None:
        # [서연] Exclusive file creation prevents accidental overwrite even on race.
        with review_path.open("x", encoding="utf-8-sig", newline="") as handle:
            review.to_csv(handle, index=False)
        report["coordinate_review_path"] = str(review_path)
    else:
        report["coordinate_review_path"] = None
        report["coordinate_review_notice"] = "Not generated: audit failed before coordinate review was available."
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(json_value(report), handle, ensure_ascii=False, indent=2)
    print(f"Audit status: {report['status']}")
    print(f"Audit JSON: {audit_path}")
    print(f"Coordinate review CSV: {report['coordinate_review_path']}")
    if error:
        print(error, file=sys.stderr)
    return 0 if success else 1
