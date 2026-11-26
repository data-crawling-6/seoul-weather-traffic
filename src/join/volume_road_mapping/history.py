"""[서연] 교통량 지점 좌표 이력 · 지점과 링크 위치 비교 · F-03 근거 기록.

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
from .common import LINES_FILENAME, STABLE_LINKS_RELATIVE, audit_provenance, input_file_provenance, json_value, normalize_link_ids, normalized_road_name, select_volume_files


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
    logs = project_root / "logs" / "ysy" / "volume_road_mapping"
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
