"""[서연] TOPIS traffic-volume spatial mapping input audit and future mapping CLI."""

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


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:  # python src/traffic_mapping.py 로 바로 실행해도 src 를 찾게
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.io_utils import data_root as _data_root, derived_of  # noqa: E402

# [서연] Fixed, versioned source inputs; never read the large link-hourly Parquet.
# 원본 데이터 위치: 저장소 data/raw/ (io_utils.data_root)
DEFAULT_DATA_ROOT = _data_root()
STABLE_LINKS_RELATIVE = Path("outputs/tables/yang_seoyeon_stable_links_20261122_v02.csv")
VERTEX_FILENAME = "link_vertices_raw_20261123T213044_v01_(성공).csv"
LINES_FILENAME = "road_lines_20261123T223648_v01_(성공).geojson"


# [서연] JSON-friendly conversion for pandas and numpy scalar values.
def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, "item"):
        return json_value(value.item())
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


# [서연] Select only successful, latest matching train/eval outputs; pair by run stamp.
def select_volume_files(folder: Path) -> dict[str, Path]:
    import re

    pattern = re.compile(
        r"^traffic_volume_(train|eval)_(\d{8}T\d{6})_v(\d+)_\(성공\)\.parquet$"
    )
    groups: dict[tuple[str, int], dict[str, Path]] = {}
    for path in folder.glob("traffic_volume_*.parquet"):
        match = pattern.fullmatch(path.name)
        if match:
            split, stamp, version = match.groups()
            groups.setdefault((stamp, int(version)), {})[split] = path
    complete = [(key, files) for key, files in groups.items() if set(files) == {"train", "eval"}]
    if not complete:
        raise FileNotFoundError(f"No complete successful train/eval pair in {folder}")
    return max(complete, key=lambda item: item[0])[1]


# [서연] Summarize coordinate history without modifying or remapping source records.
def summarize_spot_coordinate_history(volume: pd.DataFrame, spot_id: str = "F-03") -> dict[str, Any]:
    required = {"date", "spot_id", "io_type", "latitude", "longitude"}
    missing = required - set(volume.columns)
    if missing:
        raise ValueError(f"Coordinate history missing columns: {sorted(missing)}")
    rows = volume.loc[volume["spot_id"].astype("string").eq(spot_id)].copy()
    if rows.empty:
        raise ValueError(f"No volume rows for spot_id={spot_id}")
    rows["parsed_date"] = pd.to_datetime(rows["date"], errors="coerce")
    if rows["parsed_date"].isna().any():
        raise ValueError(f"Unparseable date for spot_id={spot_id}")
    rows["latitude"] = pd.to_numeric(rows["latitude"], errors="coerce")
    rows["longitude"] = pd.to_numeric(rows["longitude"], errors="coerce")
    rows["day"] = rows["parsed_date"].dt.strftime("%Y-%m-%d")
    records = []
    for direction, group in rows.groupby("io_type", sort=True, dropna=False):
        # [서연] Group by calendar day first; preserve mixed coordinates on the same day.
        daily = (group.groupby(["day", "latitude", "longitude"], dropna=False)
                 .size().reset_index(name="rows")
                 .sort_values(["day", "latitude", "longitude"], na_position="last"))
        variants = []
        for (lat, lon), part in daily.groupby(["latitude", "longitude"], dropna=False, sort=False):
            variants.append({
                "latitude": json_value(lat), "longitude": json_value(lon),
                "rows": int(part["rows"].sum()),
                "first_date": str(part["day"].min()),
                "last_date": str(part["day"].max()),
                "days": int(part["day"].nunique()),
            })
        day_counts = daily.groupby("day").size()
        records.append({
            "io_type": str(direction), "rows": int(len(group)),
            "first_date": str(group["day"].min()), "last_date": str(group["day"].max()),
            "coordinate_variants": variants,
            "days_with_multiple_coordinates": [str(x) for x in day_counts[day_counts > 1].index],
            "daily_coordinate_rows": daily.to_dict("records"),
        })
    return {"spot_id": spot_id, "rows": int(len(rows)), "directions": records}


# [서연] Inspect only the latest complete successful train/eval pair; print evidence, save nothing.
def execute_coordinate_history(data_root: Path, spot_id: str = "F-03") -> int:
    paths = select_volume_files(derived_of(data_root) / "topis_traffic_volume")
    frames = [pd.read_parquet(paths[split]) for split in ("train", "eval")]
    history = summarize_spot_coordinate_history(pd.concat(frames, ignore_index=True), spot_id)
    print(f"Coordinate history (read-only): {spot_id}")
    for split in ("train", "eval"):
        print(f"Input {split}: {paths[split]}")
        print(f"SHA-256 {split}: {input_file_provenance(paths[split])['sha256']}")
    for direction in history["directions"]:
        print(f"\n{direction['io_type']}: {direction['rows']} rows, {direction['first_date']} ~ {direction['last_date']}")
        for item in direction["coordinate_variants"]:
            print(f"  ({item['latitude']}, {item['longitude']}): {item['rows']} rows, "
                  f"{item['days']} days, {item['first_date']} ~ {item['last_date']}")
        print(f"  Days with multiple coordinates: {direction['days_with_multiple_coordinates']}")
        print("  Daily breakdown (date, latitude, longitude, rows):")
        for item in direction["daily_coordinate_rows"]:
            print(f"    {item['day']}, {item['latitude']}, {item['longitude']}, {item['rows']}")
    print("Read-only inspection; no mapping, source edits or output files created.")
    return 0


# [서연] Compare original coordinate variants with actual stable-link geometries; no automatic decisions.
def compare_spot_link_locations(volume: pd.DataFrame, stable: pd.DataFrame, lines: Any,
                                spot_id: str = "F-03") -> list[dict[str, Any]]:
    import geopandas as gpd
    import shapely

    required = {"spot_id", "io_type", "road_name", "direction_detail", "latitude", "longitude"}
    if required - set(volume.columns):
        raise ValueError(f"Missing volume metadata: {sorted(required - set(volume.columns))}")
    spot = volume.loc[volume["spot_id"].astype("string").eq(spot_id)].copy()
    if spot.empty:
        raise ValueError(f"No volume rows for {spot_id}")
    spot["latitude"] = pd.to_numeric(spot["latitude"], errors="coerce")
    spot["longitude"] = pd.to_numeric(spot["longitude"], errors="coerce")
    variants = spot.groupby(["io_type", "road_name", "direction_detail", "latitude", "longitude"],
                            dropna=False, sort=True).size().reset_index(name="rows")
    for column in ("link_id", "road_id", "road_name", "direction"):
        if column not in stable.columns:
            raise ValueError(f"Stable links missing {column}")
    if "link_id" not in lines.columns or "geometry" not in lines.columns or lines.crs is None:
        raise ValueError("Line geometry missing link_id, geometry, or CRS")
    links = stable.copy()
    links["link_id"] = normalize_link_ids(links["link_id"], "stable links")
    geo = lines[["link_id", "geometry"]].copy()
    geo["link_id"] = normalize_link_ids(geo["link_id"], "lines")
    if links["link_id"].duplicated().any() or geo["link_id"].duplicated().any():
        raise ValueError("Duplicate link_id in spatial reference")
    merged = links.merge(geo, on="link_id", validate="one_to_one")
    if len(merged) != len(links) or merged["geometry"].isna().any() or merged["geometry"].map(lambda geom: geom.is_empty if geom is not None else True).any():
        raise ValueError("Incomplete stable link geometries")
    roads = gpd.GeoDataFrame(merged, geometry="geometry", crs=lines.crs).to_crs(epsg=5179)
    records = []
    for row in variants.to_dict("records"):
        lat, lon = row["latitude"], row["longitude"]
        if pd.isna(lat) or pd.isna(lon) or not (33 <= lat <= 39 and 124 <= lon <= 132):
            raise ValueError(f"Invalid coordinate in {spot_id}: {lat}, {lon}")
        point = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs="EPSG:4326").to_crs(epsg=5179).iloc[0]
        distances = shapely.distance(roads.geometry.to_numpy(), point)
        ranked = sorted(range(len(roads)), key=lambda i: (float(distances[i]), str(roads.iloc[i]["link_id"])))
        road_label = normalized_road_name(row["road_name"])
        matching = [i for i in ranked if normalized_road_name(roads.iloc[i]["road_name"]) == road_label]
        def describe(i: int) -> dict[str, Any]:
            item = roads.iloc[i]
            return {"link_id": str(item["link_id"]), "road_id": str(item["road_id"]),
                    "road_name": str(item["road_name"]), "link_direction": str(item["direction"]),
                    "distance_m": round(float(distances[i]), 3)}
        records.append({"spot_id": spot_id, "io_type": str(row["io_type"]),
                        "direction_detail": str(row["direction_detail"]), "source_road_name": str(row["road_name"]),
                        "latitude": float(lat), "longitude": float(lon), "rows": int(row["rows"]),
                        "nearest_link": describe(ranked[0]),
                        "nearest_same_name_link": describe(matching[0]) if matching else None,
                        "same_name_link_count": len(matching),
                        "direction_verdict": "unconfirmed: io_type and link_direction cannot be equated"})
    return records


# [서연] Read-only location comparison; never update mapping or manual review artifacts.
def execute_link_location_check(project_root: Path, data_root: Path, spot_id: str = "F-03") -> int:
    import geopandas as gpd

    volumes = select_volume_files(derived_of(data_root) / "topis_traffic_volume")
    stable_path = project_root / STABLE_LINKS_RELATIVE
    lines_path = derived_of(data_root) / "road_lines" / LINES_FILENAME
    source_paths = {"train": volumes["train"], "eval": volumes["eval"],
                    "stable_links": stable_path, "road_lines": lines_path}
    frame = pd.concat([pd.read_parquet(volumes[key]) for key in ("train", "eval")], ignore_index=True)
    stable = pd.read_csv(stable_path, dtype={"link_id": "string"}, encoding="utf-8-sig")
    lines = gpd.read_file(lines_path)
    results = compare_spot_link_locations(frame, stable, lines, spot_id)
    print(f"Link location comparison (read-only): {spot_id}")
    for key, path in source_paths.items():
        print(f"Source {key}: {path}")
        print(f"SHA-256 {key}: {input_file_provenance(path)['sha256']}")
    for result in results:
        print(json.dumps(result, ensure_ascii=False))
    print("No link_id, link_direction, or mapping_status was confirmed or changed. No output files created.")
    return 0


# [서연] Preserve reproducible F-03 evidence without changing any mapping decision.
def build_f03_evidence(volume: pd.DataFrame, stable: pd.DataFrame, lines: Any) -> dict[str, Any]:
    import geopandas as gpd
    from shapely.geometry import Point

    history = summarize_spot_coordinate_history(volume, "F-03")
    locations = compare_spot_link_locations(volume, stable, lines, "F-03")
    links = stable.copy()
    links["link_id"] = normalize_link_ids(links["link_id"], "stable links")
    relevant = links.loc[links["road_id"].astype(str).eq("R005")]
    if relevant.empty:
        raise ValueError("Missing R005 reference links")
    ids = ("1120002300", "1120002100", "1120001700")
    geo = lines.copy()
    geo["link_id"] = normalize_link_ids(geo["link_id"], "road lines")
    selected = geo.loc[geo["link_id"].isin(ids)]
    if len(selected) != len(ids) or selected["link_id"].duplicated().any():
        raise ValueError("Missing or duplicated F-03 connected link geometries")
    projected = selected.to_crs(epsg=5179).set_index("link_id")
    chain = []
    for left, right in zip(ids, ids[1:]):
        left_geom, right_geom = projected.loc[left, "geometry"], projected.loc[right, "geometry"]
        if left_geom.geom_type != "LineString" or right_geom.geom_type != "LineString":
            raise ValueError("Expected LineString for F-03 chain")
        gap = Point(left_geom.coords[-1]).distance(Point(right_geom.coords[0]))
        chain.append({"from_link_id": left, "to_link_id": right,
                      "geometry_end_to_start_gap_m": round(float(gap), 3)})
    selected_stable = relevant.set_index("link_id")
    if any(i not in selected_stable.index for i in ids):
        raise ValueError("Missing F-03 chain in R005 stable links")
    if any(str(selected_stable.loc[i, "direction"]) != "하행" for i in ids):
        raise ValueError("F-03 chain direction differs from expected 하행")
    if not any(r["io_type"] == "유출" and abs(r["latitude"] - 37.596363) < 1e-6
               and abs(r["longitude"] - 126.95411397) < 1e-6
               and r["nearest_same_name_link"] is not None
               and r["nearest_same_name_link"]["link_id"] == "1120002100"
               for r in locations):
        raise ValueError("F-03 changed coordinate B does not match the observed reference link")
    return {
        "spot_id": "F-03", "road_id_reference": "R005", "road_name_reference": "내부순환로",
        "coordinate_history": history, "link_location_comparisons": locations,
        "r005_link_direction_counts": {str(k): int(v) for k, v in relevant["direction"].value_counts().items()},
        "geometry_order_candidate": list(ids), "geometry_endpoint_gaps": chain,
        "confirmed": ["F-03 source road name is 내부순환로", "coordinate B is near R005 link 1120002100",
                      "three R005 links have 하행 attributes and nearly touching geometry endpoints"],
        "unconfirmed": ["하행 corresponds to actual 홍지문터널→홍은램프 driving direction",
                        "GeoJSON coordinate order corresponds to vehicle travel direction",
                        "exact link_id and link_direction mapping selection"],
        "review_status": "pending_review",
        "decision_notice": "Evidence only. No road_id, link_id, link_direction, or mapping_status selection changed.",
    }


# [서연] Save only a new, exclusive JSON evidence log; never rewrite old artifacts.
def execute_record_f03_evidence(project_root: Path, data_root: Path) -> int:
    import geopandas as gpd

    started = datetime.now().astimezone()
    paths: dict[str, Path] = {}
    evidence = None
    error = None
    try:
        pair = select_volume_files(derived_of(data_root) / "topis_traffic_volume")
        paths = {"train": pair["train"], "eval": pair["eval"],
                 "stable_links": project_root / STABLE_LINKS_RELATIVE,
                 "road_lines": derived_of(data_root) / "road_lines" / LINES_FILENAME}
        volume = pd.concat([pd.read_parquet(paths[key]) for key in ("train", "eval")], ignore_index=True)
        stable = pd.read_csv(paths["stable_links"], dtype={"link_id": "string"}, encoding="utf-8-sig")
        lines = gpd.read_file(paths["road_lines"])
        evidence = build_f03_evidence(volume, stable, lines)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    status = "성공" if error is None else "실패"
    logs = project_root / "logs" / "ysy" / "traffic_mapping"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%dT%H%M%S")
    report = {"run_kind": "f03_spatial_evidence_only", "status": "success" if error is None else "failure",
              "started_at": started.isoformat(), "finished_at": datetime.now().astimezone().isoformat(),
              "provenance": audit_provenance(project_root, paths), "evidence": evidence,
              "error": error, "mapping_modified": False}
    for version in range(1, 1000):
        suffix = "" if version == 1 else f"_{version:03d}"
        target = logs / f"f03_link_evidence_{stamp}{suffix}_({status})_v01.json"
        try:
            with target.open("x", encoding="utf-8") as handle:
                json.dump(json_value(report), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError("No available unique evidence log path")
    print(f"F-03 evidence status: {report['status']}")
    print(f"F-03 evidence JSON: {target}")
    print("Mapping decisions remain pending_review and unchanged.")
    if error:
        print(error, file=sys.stderr)
    return 0 if error is None else 1


# [서연] Normalize link IDs to strings while rejecting missing/blank IDs.
def normalize_link_ids(series: pd.Series, source: str) -> pd.Series:
    values = series.astype("string").str.strip()
    values = values.str.replace(r"\.0$", "", regex=True)
    if values.isna().any() or values.eq("").any():
        raise ValueError(f"Missing link_id in {source}")
    return values


# [서연] Calculate geodesic maximum distance among coordinate versions of one spot-direction.
def maximum_coordinate_distance(coords: pd.DataFrame) -> float:
    from pyproj import Geod

    points = coords[["latitude", "longitude"]].drop_duplicates().to_numpy()
    if len(points) < 2:
        return 0.0
    geod = Geod(ellps="WGS84")
    maximum = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            _, _, distance = geod.inv(
                float(points[i][1]), float(points[i][0]),
                float(points[j][1]), float(points[j][0]),
            )
            maximum = max(maximum, float(distance))
    return maximum


# [서연] Build coordinate review for all spot-direction pairs, including invalid coordinate counts.
def build_coordinate_review(volume: pd.DataFrame) -> pd.DataFrame:
    required = {"spot_id", "io_type", "latitude", "longitude"}
    missing = required - set(volume.columns)
    if missing:
        raise ValueError(f"Traffic volume missing columns: {sorted(missing)}")

    frame = volume.copy()
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    valid = frame["latitude"].between(33, 39) & frame["longitude"].between(124, 132)
    zero = frame["latitude"].eq(0) | frame["longitude"].eq(0)
    records = []
    for (spot, direction), group in frame.groupby(["spot_id", "io_type"], dropna=False):
        mask_valid = valid.loc[group.index]
        coords = group.loc[mask_valid, ["latitude", "longitude"]].drop_duplicates()
        count = len(coords)
        distance = maximum_coordinate_distance(coords)
        records.append({
            "spot_id": spot,
            "io_type": direction,
            "row_count": len(group),
            "valid_coordinate_count": count,
            "invalid_coordinate_rows": int((~mask_valid).sum()),
            "zero_coordinate_rows": int(zero.loc[group.index].sum()),
            "coordinate_changed": count > 1,
            "max_coordinate_distance_m": round(distance, 3),
            "coordinate_versions": json.dumps(
                coords.to_dict("records"), ensure_ascii=False
            ),
            "review_status": (
                "manual_review_needed" if count != 1 or (~mask_valid).any()
                else "single_valid_coordinate"
            ),
        })
    return pd.DataFrame.from_records(records, columns=[
        "spot_id", "io_type", "row_count", "valid_coordinate_count",
        "invalid_coordinate_rows", "zero_coordinate_rows", "coordinate_changed",
        "max_coordinate_distance_m", "coordinate_versions", "review_status",
    ])


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


# [서연] Record the exact Git revision and working-tree state without modifying the repository.
def git_provenance(project_root: Path) -> dict[str, Any]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(project_root), *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=False, timeout=15,
        )

    try:
        revision = git("rev-parse", "HEAD")
        status = git("status", "--porcelain", "--untracked-files=all")
        if revision.returncode or status.returncode:
            return {
                "commit_id": None,
                "has_uncommitted_changes": None,
                "git_error": (revision.stderr or status.stderr or "Git unavailable").strip(),
            }
        return {
            "commit_id": revision.stdout.strip(),
            "has_uncommitted_changes": bool(status.stdout.strip()),
            "git_error": None,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"commit_id": None, "has_uncommitted_changes": None, "git_error": str(exc)}


# [서연] Stream SHA-256 so large input files do not have to fit into memory.
def input_file_provenance(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "size_bytes": None, "sha256": None}
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = digest.hexdigest()
    except OSError as exc:
        record["error"] = str(exc)
    return record


# [서연] Capture environment metadata and source fingerprints for every fresh audit.
def audit_provenance(project_root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "git": git_provenance(project_root),
        "python": {"executable": sys.executable, "version": sys.version, "implementation": platform.python_implementation()},
        "input_files": {name: input_file_provenance(path) for name, path in paths.items()},
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
    logs = project_root / "logs" / "ysy" / "traffic_mapping"
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



# [서연] Conservative, explicit road-level matching thresholds (metres).
MAP_CANDIDATE_LIMIT = 3
MAP_AUTO_MAX_DISTANCE_M = 50.0
MAP_FAR_DISTANCE_M = 200.0
MAP_LARGE_COORDINATE_CHANGE_M = 100.0
MAP_AUTO_MAX_COORDINATE_CHANGE_M = 50.0


# [서연] Compare road labels after removing only whitespace; never infer a synonym.
def normalized_road_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    return "".join(str(value).split())


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
    logs = project_root / "logs" / "ysy" / "traffic_mapping"
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
    logs = project_root / "logs" / "ysy" / "traffic_mapping"
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


# [서연] Select the newest complete successful candidate/review pair from one map run.
def select_mapping_artifact_pair(project_root: Path) -> tuple[Path, Path]:
    import re

    tables = project_root / "outputs" / "tables"
    candidate_pattern = re.compile(
        r"^yang_seoyeon_mapping_candidates_(\d{8}T\d{6})_\(성공\)_v(\d+)\.csv$"
    )
    review_pattern = re.compile(
        r"^yang_seoyeon_mapping_final_review_(\d{8}T\d{6})_\(성공\)_v(\d+)\.csv$"
    )
    pairs: dict[tuple[str, int], dict[str, Path]] = {}
    for path in tables.glob("yang_seoyeon_mapping_*_(성공)_v*.csv"):
        candidate_match = candidate_pattern.fullmatch(path.name)
        review_match = review_pattern.fullmatch(path.name)
        match = candidate_match or review_match
        if match:
            key = (match.group(1), int(match.group(2)))
            kind = "candidates" if candidate_match else "review"
            pairs.setdefault(key, {})[kind] = path
    complete = [(key, paths) for key, paths in pairs.items() if set(paths) == {"candidates", "review"}]
    if not complete:
        raise FileNotFoundError(f"Complete successful mapping candidate/review pair not found: {tables}")
    _, latest = max(complete, key=lambda item: item[0])
    return latest["candidates"], latest["review"]


# [서연] Build a manual-review sheet without changing any existing mapping decision.
def review_alternative_candidates(
    candidates: pd.DataFrame, final_review: pd.DataFrame
) -> pd.DataFrame:
    candidate_required = {
        "coordinate_version_id", "candidate_rank", "link_id", "road_id",
        "link_road_name", "distance_m", "road_name_match",
    }
    review_required = {
        "coordinate_version_id", "spot_id", "io_type", "road_name",
        "mapping_status", "review_reasons", "selected_link_id", "selected_road_id",
        "nearest_candidate_link_id", "nearest_candidate_road_id",
        "nearest_distance_m", "nearest_road_name_match", "valid_coordinate",
        "max_coordinate_change_m", "invalid_rows_in_spot_direction",
    }
    if candidate_required - set(candidates.columns):
        raise ValueError(f"Candidate columns missing: {sorted(candidate_required - set(candidates.columns))}")
    if review_required - set(final_review.columns):
        raise ValueError(f"Final-review columns missing: {sorted(review_required - set(final_review.columns))}")
    if final_review["coordinate_version_id"].astype("string").duplicated().any():
        raise ValueError("Final-review coordinate_version_id is not unique")

    candidate_frame = candidates.copy()
    review_frame = final_review.copy()
    candidate_frame["candidate_rank"] = pd.to_numeric(candidate_frame["candidate_rank"], errors="coerce")
    candidate_frame["distance_m"] = pd.to_numeric(candidate_frame["distance_m"], errors="coerce")
    review_frame["nearest_distance_m"] = pd.to_numeric(review_frame["nearest_distance_m"], errors="coerce")
    review_frame["max_coordinate_change_m"] = pd.to_numeric(
        review_frame["max_coordinate_change_m"], errors="coerce"
    )
    review_frame["invalid_rows_in_spot_direction"] = pd.to_numeric(
        review_frame["invalid_rows_in_spot_direction"], errors="coerce"
    )
    if candidate_frame[["candidate_rank", "distance_m"]].isna().any().any():
        raise ValueError("Candidate rank or distance contains non-numeric values")

    candidate_frame["road_name_match"] = (
        candidate_frame["road_name_match"].astype("string").str.strip().str.lower().eq("true")
    )
    review_frame["nearest_road_name_match"] = (
        review_frame["nearest_road_name_match"].astype("string").str.strip().str.lower().eq("true")
    )
    review_frame["valid_coordinate"] = (
        review_frame["valid_coordinate"].astype("string").str.strip().str.lower().eq("true")
    )

    # [서연] Alternatives are evidence for human review, never automatic replacements.
    alternatives = candidate_frame.loc[
        candidate_frame["candidate_rank"].gt(1)
        & candidate_frame["road_name_match"]
        & candidate_frame["distance_m"].le(MAP_AUTO_MAX_DISTANCE_M)
    ].copy()
    review_subset = review_frame.loc[
        review_frame["mapping_status"].eq("manual_review_needed")
        & ~review_frame["nearest_road_name_match"]
    ].copy()
    merged = review_subset.merge(
        alternatives[
            ["coordinate_version_id", "candidate_rank", "link_id", "road_id",
             "link_road_name", "distance_m", "road_name_match"]
        ],
        on="coordinate_version_id",
        how="inner",
        validate="one_to_many",
    )
    merged = merged.rename(columns={
        "candidate_rank": "alternative_candidate_rank",
        "link_id": "alternative_link_id",
        "road_id": "alternative_road_id",
        "link_road_name": "alternative_link_road_name",
        "distance_m": "alternative_distance_m",
        "road_name_match": "alternative_road_name_match",
    })
    merged["distance_increase_from_nearest_m"] = (
        merged["alternative_distance_m"] - merged["nearest_distance_m"]
    ).round(3)
    merged["coordinate_stable_for_auto_rule"] = (
        merged["valid_coordinate"]
        & merged["max_coordinate_change_m"].le(MAP_AUTO_MAX_COORDINATE_CHANGE_M)
        & merged["invalid_rows_in_spot_direction"].fillna(0).eq(0)
    )
    merged["review_recommendation"] = "manual_compare_alternative_link"
    merged["manual_decision"] = ""
    merged["manual_review_notes"] = ""
    ordered = [
        "coordinate_version_id", "spot_id", "io_type", "road_name",
        "mapping_status", "review_reasons", "selected_link_id", "selected_road_id",
        "nearest_candidate_link_id", "nearest_candidate_road_id", "nearest_distance_m",
        "nearest_road_name_match", "alternative_candidate_rank", "alternative_link_id",
        "alternative_road_id", "alternative_link_road_name", "alternative_distance_m",
        "alternative_road_name_match", "distance_increase_from_nearest_m",
        "valid_coordinate", "max_coordinate_change_m", "invalid_rows_in_spot_direction",
        "coordinate_stable_for_auto_rule", "review_recommendation",
        "manual_decision", "manual_review_notes",
    ]
    return merged[ordered].sort_values(
        ["coordinate_version_id", "alternative_candidate_rank", "alternative_distance_m"]
    ).reset_index(drop=True)


# [서연] Verify that an alternative sheet is conservative and traceable.
def validate_alternative_review(
    alternatives: pd.DataFrame, candidates: pd.DataFrame, final_review: pd.DataFrame
) -> dict[str, bool]:
    review_status = final_review.set_index("coordinate_version_id")["mapping_status"]
    candidate_keys = set(
        zip(
            candidates["coordinate_version_id"].astype(str),
            pd.to_numeric(candidates["candidate_rank"], errors="coerce"),
            normalize_link_ids(candidates["link_id"], "candidate alternatives"),
        )
    )
    alternative_keys = set(
        zip(
            alternatives["coordinate_version_id"].astype(str),
            pd.to_numeric(alternatives["alternative_candidate_rank"], errors="coerce"),
            normalize_link_ids(alternatives["alternative_link_id"], "alternative review"),
        )
    ) if len(alternatives) else set()
    expected_status = alternatives["coordinate_version_id"].map(review_status) if len(alternatives) else pd.Series(dtype="string")
    checks = {
        "rows_reference_existing_candidates": alternative_keys.issubset(candidate_keys),
        "only_non_nearest_candidates": bool(not len(alternatives) or alternatives["alternative_candidate_rank"].gt(1).all()),
        "only_name_matches": bool(not len(alternatives) or alternatives["alternative_road_name_match"].eq(True).all()),
        "only_within_50m": bool(not len(alternatives) or alternatives["alternative_distance_m"].le(MAP_AUTO_MAX_DISTANCE_M).all()),
        "only_manual_review_versions": bool(not len(alternatives) or expected_status.eq("manual_review_needed").all()),
        "mapping_status_preserved": bool(not len(alternatives) or alternatives["mapping_status"].eq(expected_status).all()),
        "selected_links_not_filled": bool(not len(alternatives) or alternatives[["selected_link_id", "selected_road_id"]].isna().all().all()),
        "manual_fields_initially_blank": bool(
            not len(alternatives)
            or alternatives[["manual_decision", "manual_review_notes"]].fillna("").eq("").all().all()
        ),
    }
    return checks


# [서연] Allocate versioned alternative-review evidence without overwriting prior work.
def allocate_alternative_review_paths(project_root: Path, stamp: str, success: bool) -> dict[str, Path]:
    logs = project_root / "logs" / "ysy" / "traffic_mapping"
    tables = project_root / "outputs" / "tables"
    logs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    status = "성공" if success else "실패"
    for index in range(1, 1000):
        unique = stamp if index == 1 else f"{stamp}_{index:03d}"
        paths = {
            "json": logs / f"alternative_review_{unique}_({status}).json",
            "review": tables / f"yang_seoyeon_alternative_link_review_{unique}_({status})_v01.csv",
        }
        if not any(path.exists() for path in paths.values()):
            return paths
    raise FileExistsError("Cannot allocate unique alternative-review paths")


# [서연] Explicit road-only decisions; no link or link-direction selection.
ROAD_REVIEW_NOTES = {
    "F-09": "원본 지점명·road_name·direction_detail 일관성 확인. road_id=R009(서부간선지하도로)만 수동 확정. link_id 및 link_direction 미확정. 기존 매핑 상태 유지.",
    "D-08": "원본 지점명·road_name·direction_detail 일관성 확인. road_id=R090(화랑로)만 수동 확정. link_id 및 link_direction 미확정. 기존 매핑 상태 유지.",
    "D-39": "원본 지점명·road_name·direction_detail 일관성 확인. road_id=R022(남부순환로)만 수동 확정. 대체 링크 2개 중 특정 link_id 및 link_direction 미확정. 기존 매핑 상태 유지.",
    "F-03": "좌표 변화 4182.539m의 원인 미확인. road_id, link_id 및 link_direction 판정 보류. 기존 매핑 상태 유지.",
}
ROAD_REVIEW_SPEC = {
    "V00275": ("F-09", "R009", "road_only_confirmed", 1),
    "V00276": ("F-09", "R009", "road_only_confirmed", 1),
    "V00587": ("F-09", "R009", "road_only_confirmed", 1),
    "V00588": ("F-09", "R009", "road_only_confirmed", 1),
    "V00493": ("D-08", "R090", "road_only_confirmed", 1),
    "V00555": ("D-39", "R022", "road_only_confirmed", 2),
    "V00576": ("F-03", None, "pending_review", 2),
}
ROAD_REVIEW_EDITABLE = ("manual_decision", "manual_review_notes")
ROAD_REVIEW_REQUIRED = {
    "coordinate_version_id", "spot_id", "road_name", "mapping_status",
    "selected_link_id", "selected_road_id", "alternative_link_id",
    "alternative_road_id", "alternative_distance_m", "alternative_road_name_match",
    *ROAD_REVIEW_EDITABLE,
}


# [서연] Select the latest successful v01 review by run timestamp, never a v02 or failed artifact.
def select_alternative_review_v01(project_root: Path) -> Path:
    folder = project_root / "outputs" / "tables"
    pattern = re.compile(r"yang_seoyeon_alternative_link_review_(\d{8}T\d{6})(?:_(\d{3}))?_\(성공\)_v01\.csv")
    matches = []
    for path in folder.glob("yang_seoyeon_alternative_link_review_*_(성공)_v01.csv"):
        match = pattern.fullmatch(path.name)
        if match:
            matches.append(((match.group(1), int(match.group(2) or 0)), path))
    if not matches:
        raise FileNotFoundError(f"No successful alternative-link-review v01 in {folder}")
    return max(matches, key=lambda pair: pair[0])[1]


# [서연] Validate the exact source sheet before assigning any road-only decisions.
def validate_road_review_source(source: pd.DataFrame) -> dict[str, bool]:
    checks = {
        "required_columns": ROAD_REVIEW_REQUIRED <= set(source.columns),
        "unique_columns": source.columns.is_unique,
        "exactly_9_rows": len(source) == 9,
        "exactly_7_versions": "coordinate_version_id" in source and source["coordinate_version_id"].nunique() == 7,
        "exact_versions": "coordinate_version_id" in source and set(source["coordinate_version_id"]) == set(ROAD_REVIEW_SPEC),
        "manual_fields_initially_blank": all(col in source and source[col].eq("").all() for col in ROAD_REVIEW_EDITABLE),
        "selected_fields_blank": all(col in source and source[col].eq("").all() for col in ("selected_link_id", "selected_road_id")),
        "mapping_status_manual_review": "mapping_status" in source and source["mapping_status"].eq("manual_review_needed").all(),
    }
    if checks["required_columns"] and checks["exact_versions"]:
        checks["version_row_counts"] = all(
            len(source.loc[source["coordinate_version_id"].eq(version)]) == spec[3]
            for version, spec in ROAD_REVIEW_SPEC.items()
        )
        checks["spot_version_relationship"] = all(
            source.loc[source["coordinate_version_id"].eq(version), "spot_id"].eq(spec[0]).all()
            for version, spec in ROAD_REVIEW_SPEC.items()
        )
        checks["candidate_road_relationship"] = all(
            source.loc[source["coordinate_version_id"].eq(version), "alternative_road_id"].eq(spec[1]).all()
            for version, spec in ROAD_REVIEW_SPEC.items() if spec[1] is not None
        )
        checks["candidate_link_unique"] = not source.duplicated(["coordinate_version_id", "alternative_link_id"]).any()
    else:
        checks.update(version_row_counts=False, spot_version_relationship=False,
                      candidate_road_relationship=False, candidate_link_unique=False)
    return {key: bool(value) for key, value in checks.items()}


# [서연] Change only the two approved review columns, preserving all candidate rows and order.
def apply_road_review(source: pd.DataFrame) -> pd.DataFrame:
    checks = validate_road_review_source(source)
    if not all(checks.values()):
        raise ValueError(f"Invalid alternative review v01: {checks}")
    result = source.copy(deep=True)
    for version, (spot, _road_id, decision, _count) in ROAD_REVIEW_SPEC.items():
        mask = result["coordinate_version_id"].eq(version)
        result.loc[mask, "manual_decision"] = decision
        result.loc[mask, "manual_review_notes"] = ROAD_REVIEW_NOTES[spot]
    return result


# [서연] Verify exact column order, row order, candidate identity and every immutable value.
def validate_road_review(source: pd.DataFrame, reviewed: pd.DataFrame) -> dict[str, bool]:
    source_checks = validate_road_review_source(source)
    checks = {f"source_{name}": value for name, value in source_checks.items()}
    checks["same_column_order"] = list(source.columns) == list(reviewed.columns)
    checks["same_row_count"] = len(source) == len(reviewed)
    checks["all_other_columns_unchanged"] = False
    if checks["same_column_order"] and checks["same_row_count"]:
        fixed = [col for col in source.columns if col not in ROAD_REVIEW_EDITABLE]
        checks["all_other_columns_unchanged"] = source[fixed].reset_index(drop=True).equals(reviewed[fixed].reset_index(drop=True))
    checks["exact_decisions_and_notes"] = False
    checks["duplicate_version_consistency"] = False
    checks["decision_counts"] = False
    checks["version_decision_counts"] = False
    if ROAD_REVIEW_REQUIRED <= set(reviewed.columns) and checks["same_row_count"]:
        checks["exact_decisions_and_notes"] = all(
            (lambda rows: len(rows) == spec[3]
             and rows["manual_decision"].eq(spec[2]).all()
             and rows["manual_review_notes"].eq(ROAD_REVIEW_NOTES[spec[0]]).all())(
                reviewed.loc[reviewed["coordinate_version_id"].eq(version)])
            for version, spec in ROAD_REVIEW_SPEC.items()
        )
        checks["duplicate_version_consistency"] = all(
            reviewed.groupby("coordinate_version_id", sort=False)[col].nunique(dropna=False).eq(1).all()
            for col in ROAD_REVIEW_EDITABLE
        )
        counts = reviewed["manual_decision"].value_counts()
        checks["decision_counts"] = counts.get("road_only_confirmed", 0) == 7 and counts.get("pending_review", 0) == 2 and len(counts) == 2
        versions = reviewed.groupby("manual_decision")["coordinate_version_id"].nunique()
        checks["version_decision_counts"] = versions.get("road_only_confirmed", 0) == 6 and versions.get("pending_review", 0) == 1
    return {key: bool(value) for key, value in checks.items()}


# [서연] Build the exact v02 filename from its v01 source; never allocate a replacement v02.
def road_review_output_paths(project_root: Path, source_path: Path) -> dict[str, Path]:
    output = source_path.with_name(source_path.name.replace("_v01.csv", "_v02.csv"))
    log = project_root / "logs" / "ysy" / "traffic_mapping" / (
        source_path.stem.replace("yang_seoyeon_alternative_link_review_", "road_review_validation_") + ".json"
    )
    return {"review": output, "json": log}


# [서연] Validate entirely in memory before exclusive creation; never execute mapping.
def execute_record_road_review(project_root: Path) -> int:
    started = datetime.now().astimezone()
    source_path = select_alternative_review_v01(project_root)
    paths = road_review_output_paths(project_root, source_path)
    # [서연] Reject existing outputs before reading or creating any output artifact.
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Road-review v02 or audit JSON already exists: {paths}")
    source = pd.read_csv(source_path, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    reviewed = apply_road_review(source)
    checks = validate_road_review(source, reviewed)
    if not all(checks.values()):
        raise ValueError(f"Road-review validation failed: {checks}")
    # [서연] Round-trip through a memory buffer before writing any successful v02.
    buffer = io.StringIO(newline="")
    reviewed.to_csv(buffer, index=False, lineterminator="\n")
    csv_bytes = ("\ufeff" + buffer.getvalue()).encode("utf-8")
    roundtrip = pd.read_csv(io.BytesIO(csv_bytes), dtype="string", keep_default_na=False, encoding="utf-8-sig")
    roundtrip_checks = validate_road_review(source, roundtrip)
    if not all(roundtrip_checks.values()):
        raise ValueError(f"Road-review CSV round-trip validation failed: {roundtrip_checks}")
    source_sha = input_file_provenance(source_path)["sha256"]
    if not source_sha:
        raise ValueError("Could not fingerprint source v01")
    git = git_provenance(project_root)
    output_sha = hashlib.sha256(csv_bytes).hexdigest()
    report = {
        "run_kind": "manual_road_only_alternative_review_v02",
        "status": "success",
        "started_at": started.isoformat(),
        "finished_at": None,
        "git": git,
        "input_v01": {"path": str(source_path), "sha256": source_sha},
        "output_v02": {"path": str(paths["review"]), "sha256": output_sha},
        "counts": {"rows": 9, "coordinate_versions": 7,
                   "road_only_confirmed": {"rows": 7, "coordinate_versions": 6},
                   "pending_review": {"rows": 2, "coordinate_versions": 1}},
        "validation_checks": checks,
        "roundtrip_validation_checks": roundtrip_checks,
        "record_notice": "Road-only manual decision only; exact link_id and link_direction remain unconfirmed. Original mapping_status and selected IDs unchanged.",
    }
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    # [서연] Reserve JSON first so an existing log cannot leave a misleading success CSV.
    try:
        with paths["json"].open("x", encoding="utf-8") as log_handle:
            with paths["review"].open("xb") as csv_handle:
                csv_handle.write(csv_bytes)
                csv_handle.flush()
            # [서연] Verify written bytes and original fingerprint before declaring success.
            if input_file_provenance(paths["review"])["sha256"] != output_sha:
                raise IOError("Output v02 SHA-256 mismatch")
            if input_file_provenance(source_path)["sha256"] != source_sha:
                raise IOError("Input v01 changed during recording")
            report["finished_at"] = datetime.now().astimezone().isoformat()
            json.dump(json_value(report), log_handle, ensure_ascii=False, indent=2)
    except Exception:
        # [서연] Exclusive creation means these files belong to this invocation only.
        paths["review"].unlink(missing_ok=True)
        paths["json"].unlink(missing_ok=True)
        raise
    print(f"Road-only review v02: {paths['review']}")
    print(f"Validation JSON: {paths['json']}")
    return 0


# [서연] Create a new review artifact while leaving source decisions untouched.
def execute_review_alternatives(
    project_root: Path,
    candidates_path: Path | None = None,
    final_review_path: Path | None = None,
) -> int:
    started = datetime.now().astimezone()
    error = None
    output = None
    checks = None
    if (candidates_path is None) != (final_review_path is None):
        error = "ValueError: --candidates and --final-review must be supplied together"
        sources = {"candidates": candidates_path, "final_review": final_review_path}
    else:
        try:
            if candidates_path is None:
                candidates_path, final_review_path = select_mapping_artifact_pair(project_root)
            sources = {"candidates": candidates_path, "final_review": final_review_path}
            candidates = pd.read_csv(candidates_path, dtype={"link_id": "string", "road_id": "string"})
            final_review = pd.read_csv(
                final_review_path,
                dtype={"selected_link_id": "string", "nearest_candidate_link_id": "string"},
            )
            output = review_alternative_candidates(candidates, final_review)
            checks = validate_alternative_review(output, candidates, final_review)
            if not all(checks.values()):
                error = f"AlternativeReviewValidationError: {checks}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            sources = {"candidates": candidates_path, "final_review": final_review_path}
    success = error is None
    paths = allocate_alternative_review_paths(
        project_root, started.strftime("%Y%m%dT%H%M%S"), success
    )
    if output is not None:
        with paths["review"].open("x", encoding="utf-8-sig", newline="") as handle:
            output.to_csv(handle, index=False)
    report = {
        "run_kind": "fresh_read_only_alternative_candidate_review",
        "status": "success" if success else "failure",
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "record_notice": "This output suggests manual comparisons and never changes existing mapping decisions.",
        "error": error,
        "thresholds": {"name_matching_alternative_max_distance_m": MAP_AUTO_MAX_DISTANCE_M},
        "counts": {
            "alternative_rows": int(len(output)) if output is not None else None,
            "coordinate_versions": int(output["coordinate_version_id"].nunique()) if output is not None else None,
        },
        "validation_checks": checks,
        "provenance": audit_provenance(
            project_root,
            {name: path for name, path in sources.items() if isinstance(path, Path)},
        ),
        "requested_inputs": {
            name: str(path) if path is not None else None for name, path in sources.items()
        },
        "output_files": {
            "json": str(paths["json"]),
            "alternative_review": str(paths["review"]) if paths["review"].exists() else None,
        },
    }
    with paths["json"].open("x", encoding="utf-8") as handle:
        json.dump(json_value(report), handle, ensure_ascii=False, indent=2)
    print(f"Alternative review status: {report['status']}")
    print(f"Alternative review JSON: {paths['json']}")
    print(f"Alternative review CSV: {report['output_files']['alternative_review']}")
    if error:
        print(error, file=sys.stderr)
    return 0 if success else 1


# [서연] Classify manual-review reasons without asserting new road or link decisions.
def classify_manual_review_frame(review: pd.DataFrame, reviewed: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"coordinate_version_id", "mapping_status", "review_reasons"}
    if required - set(review.columns):
        raise ValueError(f"Missing final-review columns: {sorted(required - set(review.columns))}")
    if not review["coordinate_version_id"].astype(str).is_unique:
        raise ValueError("Final review contains duplicate coordinate versions")
    if {"coordinate_version_id", "manual_decision"} - set(reviewed.columns):
        raise ValueError("Missing alternative-review columns")
    versions = reviewed[["coordinate_version_id", "manual_decision"]].drop_duplicates()
    if versions["coordinate_version_id"].duplicated().any():
        raise ValueError("Conflicting reviewed decisions for one coordinate version")
    manual = review.loc[review["mapping_status"].eq("manual_review_needed")].copy()
    manual["coordinate_version_id"] = manual["coordinate_version_id"].astype(str)
    versions["coordinate_version_id"] = versions["coordinate_version_id"].astype(str)
    if not set(versions["coordinate_version_id"]).issubset(set(manual["coordinate_version_id"])):
        raise ValueError("Reviewed versions are absent from manual-review population")
    manual = manual.merge(versions.rename(columns={"manual_decision": "previous_review_decision"}),
                          on="coordinate_version_id", how="left", validate="one_to_one", sort=False)
    manual["previous_review_decision"] = manual["previous_review_decision"].fillna("")
    def normalize(value: Any) -> str:
        return ";".join(sorted({part.strip() for part in str(value).split(";") if part.strip()}))
    manual["reason_combination"] = manual["review_reasons"].fillna("").map(normalize)
    if manual["reason_combination"].eq("").any():
        raise ValueError("Manual-review version has no reason")
    # [서연] This is triage only: no unreviewed case becomes confirmed automatically.
    manual["triage_group"] = manual["previous_review_decision"].map(
        {"road_only_confirmed": "previously_road_only_confirmed", "pending_review": "previously_pending_review"}
    ).fillna("unreviewed_needs_rule_assessment")
    counts = manual["reason_combination"].value_counts().sort_index()
    report = {
        "manual_versions": int(len(manual)),
        "previously_reviewed_versions": int(manual["previous_review_decision"].ne("").sum()),
        "unreviewed_versions": int(manual["previous_review_decision"].eq("").sum()),
        "triage_group_counts": {str(k): int(v) for k, v in manual["triage_group"].value_counts().sort_index().items()},
        "manual_reason_combinations": {str(k): int(v) for k, v in counts.items()},
        "validation_checks": {
            "unique_versions": bool(manual["coordinate_version_id"].is_unique),
            "all_manual": bool(manual["mapping_status"].eq("manual_review_needed").all()),
            "no_mapping_decisions_changed": True,
            "counts_partition": int(len(manual)) == int(manual["triage_group"].value_counts().sum()),
        },
    }
    return manual, report


# [서연] Choose latest successful final-review CSV; exclude previous/failed files.
def select_final_review_for_classification(project_root: Path) -> Path:
    folder = project_root / "outputs" / "tables"
    pattern = re.compile(r"yang_seoyeon_mapping_final_review_(\d{8}T\d{6})_\(성공\)_v01\.csv")
    matches = [(m.group(1), p) for p in folder.glob("yang_seoyeon_mapping_final_review_*_(성공)_v01.csv")
               if (m := pattern.fullmatch(p.name))]
    if not matches:
        raise FileNotFoundError(f"No successful final-review v01 in {folder}")
    return max(matches, key=lambda item: item[0])[1]


# [서연] Create new timestamped CSV and audit JSON exclusively; preserve all source files.
def execute_classify_manual_reviews(project_root: Path, final_review: Path | None = None,
                                    alternative_review: Path | None = None) -> int:
    source = final_review or select_final_review_for_classification(project_root)
    alternative = alternative_review or (project_root / "outputs" / "tables" /
        "yang_seoyeon_alternative_link_review_20261123T234035_(성공)_v02.csv")
    review = pd.read_csv(source, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    reviewed = pd.read_csv(alternative, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    classified, report = classify_manual_review_frame(review, reviewed)
    if not all(report["validation_checks"].values()):
        raise ValueError(f"Manual classification checks failed: {report['validation_checks']}")
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    folder = project_root / "outputs" / "tables"
    logs = project_root / "logs" / "ysy" / "traffic_mapping"
    folder.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        suffix = f"{stamp}_(성공)_v{version:02d}"
        csv_path = folder / f"yang_seoyeon_manual_reason_classification_{suffix}.csv"
        json_path = logs / f"manual_reason_classification_{suffix}.json"
        if not csv_path.exists() and not json_path.exists():
            break
        version += 1
    report.update({"status": "success", "source_final_review": str(source),
                   "source_alternative_review": str(alternative),
                   "source_hashes": {"final_review": input_file_provenance(source)["sha256"],
                                     "alternative_review": input_file_provenance(alternative)["sha256"]},
                   "output_csv": str(csv_path), "output_json": str(json_path),
                   "notice": "Classification only. No new road/link confirmation or mapping status changes."})
    with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        classified.to_csv(handle, index=False)
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Manual versions: {report['manual_versions']}; previously reviewed: {report['previously_reviewed_versions']}; unreviewed: {report['unreviewed_versions']}")
    print(f"Classification CSV: {csv_path}")
    print(f"Validation JSON: {json_path}")
    return 0


# [서연] Assess same-name road evidence for multiple-nearby-roads cases; never auto-confirm.
def assess_nearby_road_frame(classified: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_review = {"coordinate_version_id", "review_reasons", "mapping_status", "previous_review_decision", "road_name"}
    required_candidates = {"coordinate_version_id", "candidate_rank", "road_id", "link_id", "distance_m", "road_name_match", "direction_comparison"}
    for label, frame, required in (("classification", classified, required_review), ("candidates", candidates, required_candidates)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing {label} columns: {sorted(missing)}")
    if classified["coordinate_version_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate classification coordinate versions")
    if candidates.duplicated(["coordinate_version_id", "candidate_rank"]).any():
        raise ValueError("Duplicate candidate ranks per version")
    selected = classified.loc[classified["mapping_status"].eq("manual_review_needed") &
        classified["review_reasons"].eq("multiple_nearby_roads")].copy()
    if selected.empty:
        return pd.DataFrame(columns=["coordinate_version_id", "assessment"]), {"target_versions": 0, "assessment_counts": {}}
    chosen_ids = set(selected["coordinate_version_id"].astype(str))
    relevant = candidates.loc[candidates["coordinate_version_id"].astype(str).isin(chosen_ids)].copy()
    relevant["distance_m"] = pd.to_numeric(relevant["distance_m"], errors="coerce")
    if relevant["distance_m"].isna().any() or relevant["distance_m"].lt(0).any():
        raise ValueError("Invalid candidate distances")
    # [서연] Text cues are review flags only; no road or link is confirmed from keywords.
    structure_patterns = (
        ("underground_or_underpass", r"지하|터널|지하차도"),
        ("expressway_or_arterial", r"고속|간선|올림픽대로|자동차전용"),
        ("ramp_or_interchange", r"램프|진출|진입|IC|JC|분기점|나들목"),
        ("bridge_or_elevated", r"고가|교량|대교|철교"),
    )
    records = []
    for item in selected.to_dict("records"):
        version = str(item["coordinate_version_id"])
        group = relevant.loc[relevant["coordinate_version_id"].astype(str).eq(version)].sort_values(["distance_m", "candidate_rank"])
        same = group.loc[group["road_name_match"].astype(str).str.lower().eq("true")]
        other = group.loc[~group["road_name_match"].astype(str).str.lower().eq("true")]
        road_ids = sorted(set(same["road_id"].dropna().astype(str)) - {""})
        nearest_same = float(same.iloc[0]["distance_m"]) if len(same) else None
        nearest_other = float(other.iloc[0]["distance_m"]) if len(other) else None
        gap = round(nearest_other - nearest_same, 3) if nearest_same is not None and nearest_other is not None else None
        previous = str(item.get("previous_review_decision") or "")
        # [서연] 20 m gap is exploratory, not a verified correctness threshold.
        if previous:
            assessment = "previously_reviewed_excluded"
        elif not len(group) or not len(same) or len(road_ids) != 1 or nearest_other is None:
            assessment = "insufficient_or_ambiguous_evidence"
        elif gap >= 20:
            assessment = "road_only_priority_for_validation"
        else:
            assessment = "additional_manual_review"
        # [서연] Assess original spot text and nearest same-name link independently of distance triage.
        original_text = " ".join(str(item.get(field) or "") for field in
                                 ("road_name", "spot_name", "direction_detail"))
        structure_flags = [label for label, pattern in structure_patterns
                           if re.search(pattern, original_text, flags=re.IGNORECASE)]
        # [서연] Source direction cannot be crosswalked to TOPIS up/down without verified evidence.
        direction_unverified = bool(len(same)) and str(same.iloc[0]["direction_comparison"]) == "not_comparable_without_verified_crosswalk"
        structure_review = ("structure_cue_requires_review" if structure_flags else
                            "no_text_structure_cue_detected")
        records.append({
            "coordinate_version_id": version, "spot_id": item.get("spot_id", ""),
            "io_type": item.get("io_type", ""), "road_name": item["road_name"],
            "previous_review_decision": previous, "candidate_count_actual": int(len(group)),
            "distinct_candidate_roads": int(group["road_id"].nunique()),
            "same_name_road_count": len(road_ids),
            "same_name_road_id": road_ids[0] if len(road_ids) == 1 else "",
            "nearest_same_name_m": nearest_same, "nearest_other_name_m": nearest_other,
            "other_name_distance_gap_m": gap,
            "nearest_same_direction_comparison": str(same.iloc[0]["direction_comparison"]) if len(same) else "",
            "assessment": assessment,
            "spot_name": item.get("spot_name", ""),
            "direction_detail": item.get("direction_detail", ""),
            "structure_risk_flags": ";".join(structure_flags),
            "structure_review": structure_review,
            "direction_crosswalk_unverified": direction_unverified,
            "geometry_overlap_verified": False,
            "validation_readiness": ("requires_structure_and_direction_validation"
                                     if structure_flags else "requires_direction_and_geometry_validation"),
            "mapping_decision": "unchanged",
            "assessment_notice": "Priority for further validation only; not an automatic road/link confirmation",
        })
    result = pd.DataFrame.from_records(records)
    counts = result["assessment"].value_counts().sort_index()
    structure_counts = result["structure_review"].value_counts().sort_index()
    priority = result.loc[result["assessment"].eq("road_only_priority_for_validation")]
    report = {"target_versions": len(result),
              "assessment_counts": {str(k): int(v) for k, v in counts.items()},
              "structure_review_counts": {str(k): int(v) for k, v in structure_counts.items()},
              "priority_structure_review_counts": {
                  str(k): int(v) for k, v in priority["structure_review"].value_counts().sort_index().items()
              },
              "validation_checks": {
                  "unique_target_versions": bool(result["coordinate_version_id"].is_unique),
                  "target_population_matches": set(result["coordinate_version_id"]) == chosen_ids,
                  "assessment_partition": int(counts.sum()) == len(result),
                  "no_mapping_decisions_changed": bool(result["mapping_decision"].eq("unchanged").all()),
                  "structure_partition": int(structure_counts.sum()) == len(result),
                  "no_geometry_claims": bool(result["geometry_overlap_verified"].eq(False).all()),
              }}
    return result, report


# [서연] Write a separate, versioned assessment; preserve mapping and classification inputs.
def execute_assess_nearby_roads(project_root: Path, classification: Path | None = None,
                              candidates: Path | None = None) -> int:
    folder = project_root / "outputs" / "tables"
    if classification is None:
        paths = sorted(folder.glob("yang_seoyeon_manual_reason_classification_*_(성공)_v*.csv"))
        if not paths:
            raise FileNotFoundError("No successful manual reason classification CSV")
        classification = paths[-1]
    candidates = candidates or select_candidate_file(project_root)
    classified = pd.read_csv(classification, dtype="string", encoding="utf-8-sig", keep_default_na=False)
    candidate_frame = pd.read_csv(candidates, dtype="string", encoding="utf-8-sig", keep_default_na=False)
    result, report = assess_nearby_road_frame(classified, candidate_frame)
    if not all(report.get("validation_checks", {}).values()):
        raise ValueError(f"Assessment validation failed: {report.get('validation_checks')}")
    logs = project_root / "logs" / "ysy" / "traffic_mapping"
    folder.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    version = 1
    while True:
        suffix = f"{stamp}_(성공)_v{version:02d}"
        output_csv = folder / f"yang_seoyeon_nearby_road_assessment_{suffix}.csv"
        output_json = logs / f"nearby_road_assessment_{suffix}.json"
        if not output_csv.exists() and not output_json.exists():
            break
        version += 1
    report.update({"status": "success", "source_classification": str(classification),
                   "source_candidates": str(candidates),
                   "source_hashes": {"classification": input_file_provenance(classification)["sha256"],
                                     "candidates": input_file_provenance(candidates)["sha256"]},
                   "output_csv": str(output_csv), "output_json": str(output_json),
                   "notice": "Read-only assessment; no mapping decisions are changed. 20m gap is an exploratory triage threshold."})
    with output_csv.open("x", encoding="utf-8-sig", newline="") as handle:
        result.to_csv(handle, index=False)
    with output_json.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Assessed versions: {report['target_versions']}; groups: {report['assessment_counts']}")
    print(f"Priority structure flags: {report['priority_structure_review_counts']}")
    print(f"Assessment CSV: {output_csv}")
    print(f"Validation JSON: {output_json}")
    return 0


# [서연] Assess every unreviewed coordinate version in one read-only pass.
def assess_all_manual_frame(classified: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"coordinate_version_id", "mapping_status", "review_reasons", "previous_review_decision", "road_name"}
    candidate_required = {"coordinate_version_id", "candidate_rank", "road_id", "link_id", "distance_m", "road_name_match", "direction_comparison"}
    if required - set(classified.columns) or candidate_required - set(candidates.columns):
        raise ValueError("Missing classification or candidate columns")
    if classified["coordinate_version_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate classification coordinate versions")
    if candidates.duplicated(["coordinate_version_id", "candidate_rank"]).any():
        raise ValueError("Duplicate candidate ranks per version")
    unreviewed = classified.loc[classified["mapping_status"].eq("manual_review_needed") &
                               classified["previous_review_decision"].fillna("").eq("")].copy()
    candidate_copy = candidates.copy()
    candidate_copy["coordinate_version_id"] = candidate_copy["coordinate_version_id"].astype(str)
    candidate_copy["distance_m"] = pd.to_numeric(candidate_copy["distance_m"], errors="coerce")
    if candidate_copy["distance_m"].isna().any() or candidate_copy["distance_m"].lt(0).any():
        raise ValueError("Invalid candidate distances")
    by_version = {str(key): group.sort_values(["distance_m", "candidate_rank"])
                  for key, group in candidate_copy.groupby("coordinate_version_id", sort=False)}
    patterns = (
        ("underground_or_underpass", r"지하|터널|지하차도"),
        ("expressway_or_arterial", r"고속|간선|올림픽대로|자동차전용"),
        ("ramp_or_interchange", r"램프|진출|진입|IC|JC|분기점|나들목"),
        ("bridge_or_elevated", r"고가|교량|대교|철교"),
    )
    records = []
    for item in unreviewed.to_dict("records"):
        version = str(item["coordinate_version_id"])
        group = by_version.get(version, candidate_copy.iloc[0:0])
        matched = group.loc[group["road_name_match"].astype(str).str.lower().eq("true")]
        others = group.loc[~group["road_name_match"].astype(str).str.lower().eq("true")]
        same_ids = sorted(set(matched["road_id"].dropna().astype(str)) - {""})
        nearest_same = float(matched.iloc[0]["distance_m"]) if len(matched) else None
        nearest_other = float(others.iloc[0]["distance_m"]) if len(others) else None
        gap = round(nearest_other - nearest_same, 3) if nearest_same is not None and nearest_other is not None else None
        reasons = sorted(set(str(item.get("review_reasons") or "").split(";")) - {""})
        text_fields = " ".join(str(item.get(field) or "") for field in ("road_name", "spot_name", "direction_detail"))
        flags = [name for name, pattern in patterns if re.search(pattern, text_fields, flags=re.IGNORECASE)]
        # [서연] Conservative triage: coordinate shifts and distant candidates cannot be cleared by name/distance alone.
        coordinate_risk = any("coordinate_change" in r or "invalid_coordinate" in r for r in reasons)
        distant_risk = "far_from_analysis_links" in reasons
        name_mismatch = "nearest_road_name_mismatch" in reasons
        if not len(group) or not len(matched) or len(same_ids) != 1:
            group_label = "manual_review_required"
            rationale = "missing_or_ambiguous_same_name_road"
        elif coordinate_risk or distant_risk or name_mismatch:
            group_label = "manual_review_required"
            rationale = "coordinate_distance_or_name_conflict"
        elif gap is None:
            group_label = "additional_validation"
            rationale = "no_other_name_comparator_in_saved_candidates"
        elif gap < 20:
            group_label = "additional_validation"
            rationale = "other_road_distance_gap_below_exploratory_threshold"
        else:
            group_label = "road_only_priority_for_validation"
            rationale = "same_name_unique_road_and_exploratory_gap"
        # [서연] Subtype manual cases without clearing any original review reason.
        if group_label == "manual_review_required":
            if coordinate_risk:
                next_check = "coordinate_history_consistency"
            elif distant_risk:
                next_check = "analysis_road_coverage_and_geometry"
            elif name_mismatch:
                next_check = "road_name_alias_or_candidate_coverage"
            else:
                next_check = "same_name_road_ambiguity"
        elif group_label == "additional_validation":
            next_check = "candidate_distance_and_geometry"
        else:
            next_check = "road_geometry_and_direction_crosswalk"
        # [서연] A matching road-name candidate is evidence for review, not a resolved alias.
        name_alias_candidate = bool(name_mismatch and len(matched) and len(same_ids) == 1)
        coordinate_history_candidate = bool(coordinate_risk and len(matched) and len(same_ids) == 1)
        records.append({
            "coordinate_version_id": version, "spot_id": item.get("spot_id", ""),
            "io_type": item.get("io_type", ""), "road_name": item.get("road_name", ""),
            "spot_name": item.get("spot_name", ""), "direction_detail": item.get("direction_detail", ""),
            "review_reasons": item.get("review_reasons", ""), "reason_combination": item.get("reason_combination", ""),
            "candidate_count_actual": int(len(group)), "distinct_candidate_roads": int(group["road_id"].nunique()),
            "same_name_road_count": len(same_ids), "same_name_road_id": same_ids[0] if len(same_ids) == 1 else "",
            "nearest_same_name_m": nearest_same, "nearest_other_name_m": nearest_other,
            "other_name_distance_gap_m": gap,
            "nearest_same_direction_comparison": str(matched.iloc[0]["direction_comparison"]) if len(matched) else "",
            "structure_risk_flags": ";".join(flags), "structure_review": "structure_cue_requires_review" if flags else "no_text_structure_cue_detected",
            "coordinate_change_risk": coordinate_risk, "distant_link_risk": distant_risk,
            "direction_crosswalk_verified": False, "geometry_overlap_verified": False,
            "assessment": group_label, "assessment_reason": rationale,
            "next_check": next_check,
            "name_alias_candidate_for_review": name_alias_candidate,
            "coordinate_history_candidate_for_review": coordinate_history_candidate,
            "mapping_decision": "unchanged",
            "assessment_notice": "Triage only; no road/link is confirmed, including priority candidates",
        })
    result = pd.DataFrame.from_records(records)
    counts = result["assessment"].value_counts().sort_index() if len(result) else pd.Series(dtype="int64")
    report = {
        "unreviewed_versions": len(result),
        "assessment_counts": {str(k): int(v) for k, v in counts.items()},
        "reason_combinations": {str(k): int(v) for k, v in result["reason_combination"].value_counts().sort_index().items()} if len(result) else {},
        "structure_review_counts": {str(k): int(v) for k, v in result["structure_review"].value_counts().sort_index().items()} if len(result) else {},
        "next_check_counts": {str(k): int(v) for k, v in result["next_check"].value_counts().sort_index().items()} if len(result) else {},
        "manual_reason_combinations": {str(k): int(v) for k, v in
            result.loc[result["assessment"].eq("manual_review_required"), "review_reasons"].value_counts().items()} if len(result) else {},
        "name_alias_candidates_for_review": int(result["name_alias_candidate_for_review"].sum()) if len(result) else 0,
        "coordinate_history_candidates_for_review": int(result["coordinate_history_candidate_for_review"].sum()) if len(result) else 0,
        "validation_checks": {
            "unique_unreviewed_versions": bool(result["coordinate_version_id"].is_unique) if len(result) else True,
            "population_matches": set(result["coordinate_version_id"]) == set(unreviewed["coordinate_version_id"].astype(str)),
            "assessment_partition": int(counts.sum()) == len(result),
            "no_mapping_decisions_changed": bool(result["mapping_decision"].eq("unchanged").all()) if len(result) else True,
            "no_direction_or_geometry_claims": bool((~result["direction_crosswalk_verified"]).all() and (~result["geometry_overlap_verified"]).all()) if len(result) else True,
        },
    }
    return result, report


# [서연] Preserve all input files and save a new timestamped bulk triage CSV/JSON.
def execute_assess_all_manual(project_root: Path, classification: Path | None = None,
                              candidates: Path | None = None) -> int:
    folder = project_root / "outputs" / "tables"
    if classification is None:
        options = sorted(folder.glob("yang_seoyeon_manual_reason_classification_*_(성공)_v*.csv"))
        if not options:
            raise FileNotFoundError("No successful manual classification CSV")
        classification = options[-1]
    candidates = candidates or select_candidate_file(project_root)
    classified = pd.read_csv(classification, dtype="string", encoding="utf-8-sig", keep_default_na=False)
    candidate_frame = pd.read_csv(candidates, dtype="string", encoding="utf-8-sig", keep_default_na=False)
    result, report = assess_all_manual_frame(classified, candidate_frame)
    if not all(report["validation_checks"].values()):
        raise ValueError(f"Bulk validation failed: {report['validation_checks']}")
    logs = project_root / "logs" / "ysy" / "traffic_mapping"
    folder.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    version = 1
    while True:
        suffix = f"{stamp}_(성공)_v{version:02d}"
        csv_path = folder / f"yang_seoyeon_all_manual_assessment_{suffix}.csv"
        json_path = logs / f"all_manual_assessment_{suffix}.json"
        if not csv_path.exists() and not json_path.exists():
            break
        version += 1
    report.update({
        "status": "success", "source_classification": str(classification), "source_candidates": str(candidates),
        "source_hashes": {"classification": input_file_provenance(classification)["sha256"],
                          "candidates": input_file_provenance(candidates)["sha256"]},
        "output_csv": str(csv_path), "output_json": str(json_path),
        "notice": "Read-only bulk triage; no original mappings or prior results are modified. 20m is exploratory only.",
    })
    with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        result.to_csv(handle, index=False)
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(json_value(report), handle, ensure_ascii=False, indent=2)
    print(f"Unreviewed versions: {report['unreviewed_versions']}; groups: {report['assessment_counts']}")
    print(f"Structure flags: {report['structure_review_counts']}")
    print(f"Next checks: {report['next_check_counts']}")
    print(f"Name alias review candidates: {report['name_alias_candidates_for_review']}; coordinate history review candidates: {report['coordinate_history_candidates_for_review']}")
    print(f"Assessment CSV: {csv_path}")
    print(f"Validation JSON: {json_path}")
    return 0


# [서연] Compare coordinate versions in each spot/direction without asserting driving direction.
def assess_coordinate_history_frame(classified: pd.DataFrame, candidates: pd.DataFrame,
                                    bulk: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"coordinate_version_id", "spot_id", "io_type", "latitude", "longitude",
                "road_name", "mapping_status", "previous_review_decision"}
    candidate_required = {"coordinate_version_id", "road_id", "distance_m", "road_name_match"}
    bulk_required = {"coordinate_version_id", "next_check", "coordinate_history_candidate_for_review"}
    if required - set(classified.columns) or candidate_required - set(candidates.columns) or bulk_required - set(bulk.columns):
        raise ValueError("Missing required columns for coordinate history assessment")
    for name, df in (("classification", classified), ("bulk", bulk)):
        if df["coordinate_version_id"].astype(str).duplicated().any():
            raise ValueError(f"Duplicate coordinate_version_id in {name}")
    target = bulk.loc[bulk["next_check"].eq("coordinate_history_consistency")].copy()
    classified = classified.copy()
    classified["coordinate_version_id"] = classified["coordinate_version_id"].astype(str)
    classified["latitude"] = pd.to_numeric(classified["latitude"], errors="coerce")
    classified["longitude"] = pd.to_numeric(classified["longitude"], errors="coerce")
    candidates = candidates.copy()
    candidates["coordinate_version_id"] = candidates["coordinate_version_id"].astype(str)
    candidates["distance_m"] = pd.to_numeric(candidates["distance_m"], errors="coerce")
    if candidates["distance_m"].isna().any() or candidates["distance_m"].lt(0).any():
        raise ValueError("Invalid candidate distances")
    by_candidate = {str(key): group.sort_values("distance_m")
                    for key, group in candidates.groupby("coordinate_version_id", sort=False)}
    by_version = classified.set_index("coordinate_version_id")
    by_spot = {(str(k[0]), str(k[1])): g for k, g in
               classified.groupby(["spot_id", "io_type"], dropna=False)}
    records = []
    for row in target.to_dict("records"):
        vid = str(row["coordinate_version_id"])
        if vid not in by_version.index:
            raise ValueError(f"Missing classification version: {vid}")
        item = by_version.loc[vid]
        spot = str(item["spot_id"])
        io = str(item["io_type"])
        peers = by_spot[(spot, io)]
        # [서연] Saved candidates are limited to three; missing roads are not proof of absence.
        own_candidates = by_candidate.get(vid, candidates.iloc[:0])
        own_match = own_candidates.loc[own_candidates["road_name_match"].astype(str).str.lower().eq("true")]
        own_roads = set(own_match["road_id"].dropna().astype(str)) - {""}
        peer_roads = []
        peer_versions = []
        peer_nearest_m = []
        for peer in peers.to_dict("records"):
            pvid = str(peer["coordinate_version_id"])
            if pvid == vid:
                continue
            pc = by_candidate.get(pvid, candidates.iloc[:0])
            pm = pc.loc[pc["road_name_match"].astype(str).str.lower().eq("true")]
            ids = set(pm["road_id"].dropna().astype(str)) - {""}
            peer_roads.append(ids)
            peer_versions.append(pvid)
            peer_nearest_m.append(float(pm.iloc[0]["distance_m"]) if len(pm) else None)
        own_single = len(own_roads) == 1
        peers_single_same = bool(peer_roads) and all(ids == own_roads for ids in peer_roads)
        coords = peers[["latitude", "longitude"]].dropna()
        coords = coords.loc[coords["latitude"].between(-90, 90) & coords["longitude"].between(-180, 180)]
        max_shift_m = round(float(maximum_coordinate_distance(coords)), 3) if len(coords) > 1 else None
        if not peer_versions:
            status = "no_peer_in_classification"
        elif not own_single:
            status = "target_road_ambiguous"
        elif not peers_single_same:
            status = "road_candidate_inconsistent_or_missing"
        else:
            status = "same_road_candidate_across_versions"
        records.append({
            "coordinate_version_id": vid, "spot_id": spot, "io_type": io,
            "road_name": item["road_name"], "peer_versions_in_classification": len(peer_versions),
            "peer_coordinate_version_ids": ";".join(peer_versions),
            "target_same_name_road_id": next(iter(own_roads)) if own_single else "",
            "peer_same_name_road_ids": ";".join(",".join(sorted(ids)) for ids in peer_roads),
            "max_coordinate_shift_m": max_shift_m,
            "target_nearest_same_name_m": float(own_match.iloc[0]["distance_m"]) if len(own_match) else None,
            "peer_nearest_same_name_m": ";".join("" if d is None else str(d) for d in peer_nearest_m),
            "history_evidence_status": status,
            "direction_crosswalk_verified": False, "geometry_overlap_verified": False,
            "mapping_decision": "unchanged",
            "assessment_notice": "Candidate-level history only; no road or link confirmed",
        })
    result = pd.DataFrame.from_records(records)
    counts = result["history_evidence_status"].value_counts().sort_index() if len(result) else pd.Series(dtype="int64")
    report = {
        "target_versions": len(target),
        "history_evidence_counts": {str(k): int(v) for k, v in counts.items()},
        "validation_checks": {
            "target_population_matches": set(result["coordinate_version_id"]) == set(target["coordinate_version_id"].astype(str)),
            "unique_target_versions": bool(result["coordinate_version_id"].is_unique) if len(result) else True,
            "status_partition": int(counts.sum()) == len(target),
            "no_mapping_changes": bool(result["mapping_decision"].eq("unchanged").all()) if len(result) else True,
            "no_unverified_geometry_or_direction_claims": bool((~result["direction_crosswalk_verified"]).all() and
                                                               (~result["geometry_overlap_verified"]).all()) if len(result) else True,
        },
    }
    return result, report


# [서연] Produce new immutable CSV/JSON evidence for coordinate-history review.
def execute_assess_coordinate_history(project_root: Path, classification: Path | None = None,
                                      candidates: Path | None = None, bulk_assessment: Path | None = None) -> int:
    folder = project_root / "outputs" / "tables"
    logs = project_root / "logs" / "ysy" / "traffic_mapping"
    def latest(pattern: str) -> Path:
        options = sorted(folder.glob(pattern))
        if not options:
            raise FileNotFoundError(f"No successful input: {pattern}")
        return options[-1]
    classification = classification or latest("yang_seoyeon_manual_reason_classification_*_(성공)_v*.csv")
    candidates = candidates or select_candidate_file(project_root)
    bulk_assessment = bulk_assessment or latest("yang_seoyeon_all_manual_assessment_*_(성공)_v*.csv")
    classified = pd.read_csv(classification, dtype="string", encoding="utf-8-sig", keep_default_na=False)
    candidate_frame = pd.read_csv(candidates, dtype="string", encoding="utf-8-sig", keep_default_na=False)
    bulk = pd.read_csv(bulk_assessment, dtype="string", encoding="utf-8-sig", keep_default_na=False)
    result, report = assess_coordinate_history_frame(classified, candidate_frame, bulk)
    if not all(report["validation_checks"].values()):
        raise ValueError(f"Coordinate history validation failed: {report['validation_checks']}")
    folder.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    version = 1
    while True:
        suffix = f"{stamp}_(성공)_v{version:02d}"
        csv_path = folder / f"yang_seoyeon_coordinate_history_assessment_{suffix}.csv"
        json_path = logs / f"coordinate_history_assessment_{suffix}.json"
        if not csv_path.exists() and not json_path.exists():
            break
        version += 1
    report.update({
        "status": "success",
        "source_files": {"classification": str(classification), "candidates": str(candidates),
                         "bulk_assessment": str(bulk_assessment)},
        "source_hashes": {"classification": input_file_provenance(classification)["sha256"],
                          "candidates": input_file_provenance(candidates)["sha256"],
                          "bulk_assessment": input_file_provenance(bulk_assessment)["sha256"]},
        "output_csv": str(csv_path), "output_json": str(json_path),
        "notice": "Read-only candidate history; classification may omit auto-confirmed peer versions.",
    })
    with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        result.to_csv(handle, index=False)
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(json_value(report), handle, ensure_ascii=False, indent=2)
    print(f"Coordinate history versions: {report['target_versions']}; groups: {report['history_evidence_counts']}")
    print(f"Assessment CSV: {csv_path}")
    print(f"Validation JSON: {json_path}")
    return 0


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
    log_dir = project_root / "logs" / "ysy" / "traffic_mapping"
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
    log_dir = project_root / "logs" / "ysy" / "traffic_mapping"
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
    log_dir = project_root / "logs" / "ysy" / "traffic_mapping"
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
    log_dir = project_root / "logs" / "ysy" / "traffic_mapping"
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
    log_dir = project_root / "logs" / "ysy" / "traffic_mapping"
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


# [서연] Reusable CLI entry point for audit, mapping and read-only diagnostics.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="[서연] TOPIS traffic mapping")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-inputs", help="Audit mapping inputs and save versioned evidence")
    audit.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    audit.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    mapping_cmd = sub.add_parser("map", help="Generate conservative version-level link candidates")
    mapping_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    mapping_cmd.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    analysis_cmd = sub.add_parser("analyze-candidates", help="Read-only diagnostics for a saved candidate CSV")
    analysis_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    analysis_cmd.add_argument("--candidates", type=Path, default=None)
    alternative_cmd = sub.add_parser(
        "review-alternatives", help="Create a read-only manual review sheet for alternative links"
    )
    alternative_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    alternative_cmd.add_argument("--candidates", type=Path, default=None)
    alternative_cmd.add_argument("--final-review", type=Path, default=None)
    road_review_cmd = sub.add_parser(
        "record-road-review", help="Record validated road-only decisions in alternative review v02"
    )
    road_review_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    history_cmd = sub.add_parser(
        "coordinate-history", help="Read-only date-by-date coordinate history for one spot"
    )
    history_cmd.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    history_cmd.add_argument("--spot-id", default="F-03")
    location_cmd = sub.add_parser(
        "link-location-check", help="Read-only compare spot coordinates with stable road link geometries"
    )
    location_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    location_cmd.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    location_cmd.add_argument("--spot-id", default="F-03")
    evidence_cmd = sub.add_parser(
        "record-f03-evidence", help="Save new F-03 spatial evidence JSON without modifying mapping decisions"
    )
    evidence_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    evidence_cmd.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    classify_cmd = sub.add_parser("classify-manual-reviews", help="Group manual review reasons without changing mapping")
    classify_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    classify_cmd.add_argument("--final-review", type=Path, default=None)
    classify_cmd.add_argument("--alternative-review", type=Path, default=None)
    nearby_cmd = sub.add_parser("assess-nearby-roads", help="Read-only evidence assessment for multiple-nearby-roads cases")
    nearby_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    nearby_cmd.add_argument("--classification", type=Path, default=None)
    nearby_cmd.add_argument("--candidates", type=Path, default=None)
    bulk_cmd = sub.add_parser("assess-all-manual", help="Read-only one-pass triage for all unreviewed versions")
    bulk_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    bulk_cmd.add_argument("--classification", type=Path, default=None)
    bulk_cmd.add_argument("--candidates", type=Path, default=None)
    history_assess_cmd = sub.add_parser("assess-coordinate-history", help="Read-only batch comparison of coordinate versions")
    history_assess_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    history_assess_cmd.add_argument("--classification", type=Path, default=None)
    history_assess_cmd.add_argument("--candidates", type=Path, default=None)
    history_assess_cmd.add_argument("--bulk-assessment", type=Path, default=None)
    full_cmd = sub.add_parser("evaluate-full-roads", help="Compare all analysis links by road, read-only")
    full_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    full_cmd.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    official_cmd = sub.add_parser("build-official-road-tables", help="Official road-name mapping and spot coverage")
    official_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    official_cmd.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    observation_cmd = sub.add_parser("join-official-observations", help="Row-preserving official road-name join for traffic observations")
    observation_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    observation_cmd.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    baseline_cmd = sub.add_parser("audit-baseline", help="Read-only weekday/hour baseline feasibility")
    baseline_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    baseline_cmd.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    date_cmd = sub.add_parser("audit-sampled-dates", help="Read-only year/month and weather-label readiness audit")
    date_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    date_cmd.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)
    if args.command == "audit-sampled-dates":
        return execute_date_audit(args.project_root, args.data_root)
    if args.command == "audit-baseline":
        return execute_baseline_audit(args.project_root, args.data_root)
    if args.command == "evaluate-full-roads":
        return execute_full_road_evaluation(args.project_root, args.data_root)
    if args.command == "join-official-observations":
        return execute_official_observation_join(args.project_root, args.data_root)
    if args.command == "build-official-road-tables":
        return execute_official_road_mapping(args.project_root, args.data_root)
    if args.command == "assess-coordinate-history":
        return execute_assess_coordinate_history(args.project_root, args.classification,
                                                 args.candidates, args.bulk_assessment)
    if args.command == "assess-all-manual":
        return execute_assess_all_manual(args.project_root, args.classification, args.candidates)
    if args.command == "assess-nearby-roads":
        return execute_assess_nearby_roads(args.project_root, args.classification, args.candidates)
    if args.command == "classify-manual-reviews":
        return execute_classify_manual_reviews(args.project_root, args.final_review, args.alternative_review)
    if args.command == "record-f03-evidence":
        return execute_record_f03_evidence(args.project_root, args.data_root)
    if args.command == "link-location-check":
        return execute_link_location_check(args.project_root, args.data_root, args.spot_id)
    if args.command == "coordinate-history":
        return execute_coordinate_history(args.data_root, args.spot_id)
    if args.command == "map":
        return execute_map(args.project_root, args.data_root)
    if args.command == "audit-inputs":
        return execute_audit(args.project_root, args.data_root)
    if args.command == "analyze-candidates":
        return execute_candidate_analysis(args.project_root, args.candidates)
    if args.command == "review-alternatives":
        return execute_review_alternatives(
            args.project_root, args.candidates, args.final_review
        )
    if args.command == "record-road-review":
        return execute_record_road_review(args.project_root)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
