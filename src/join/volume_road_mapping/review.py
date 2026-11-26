"""[서연] ③ 대체 후보 비교 · 수동 검토 분류 · 근처 도로와 좌표 이력 확인.

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

from .candidates import MAP_AUTO_MAX_COORDINATE_CHANGE_M, MAP_AUTO_MAX_DISTANCE_M, select_candidate_file
from .common import audit_provenance, git_provenance, input_file_provenance, json_value, maximum_coordinate_distance, normalize_link_ids


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
    logs = project_root / "logs" / "ysy" / "volume_road_mapping"
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
    log = project_root / "logs" / "ysy" / "volume_road_mapping" / (
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
    logs = project_root / "logs" / "ysy" / "volume_road_mapping"
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
    logs = project_root / "logs" / "ysy" / "volume_road_mapping"
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
    logs = project_root / "logs" / "ysy" / "volume_road_mapping"
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
    logs = project_root / "logs" / "ysy" / "volume_road_mapping"
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
