"""[서연] 공통: 데이터 위치 · 파일 고르기 · JSON 변환 · git 출처 기록.

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

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:  # python -m 없이 바로 실행해도 src 를 찾게
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


# [서연] Compare road labels after removing only whitespace; never infer a synonym.
def normalized_road_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    return "".join(str(value).split())
