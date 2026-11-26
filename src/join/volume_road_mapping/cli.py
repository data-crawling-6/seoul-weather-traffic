"""[서연] 실행 명령 (python -m src.join.volume_road_mapping <명령>).

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

from .audit import execute_audit
from .candidates import execute_candidate_analysis, execute_map
from .checks import execute_baseline_audit, execute_date_audit
from .common import DEFAULT_DATA_ROOT
from .history import execute_coordinate_history, execute_link_location_check, execute_record_f03_evidence
from .official import execute_full_road_evaluation, execute_official_observation_join, execute_official_road_mapping
from .review import execute_assess_all_manual, execute_assess_coordinate_history, execute_assess_nearby_roads, execute_classify_manual_reviews, execute_record_road_review, execute_review_alternatives


# [서연] Reusable CLI entry point for audit, mapping and read-only diagnostics.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="[서연] TOPIS traffic mapping")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-inputs", help="Audit mapping inputs and save versioned evidence")
    audit.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    audit.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    mapping_cmd = sub.add_parser("map", help="Generate conservative version-level link candidates")
    mapping_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    mapping_cmd.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    analysis_cmd = sub.add_parser("analyze-candidates", help="Read-only diagnostics for a saved candidate CSV")
    analysis_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    analysis_cmd.add_argument("--candidates", type=Path, default=None)
    alternative_cmd = sub.add_parser(
        "review-alternatives", help="Create a read-only manual review sheet for alternative links"
    )
    alternative_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    alternative_cmd.add_argument("--candidates", type=Path, default=None)
    alternative_cmd.add_argument("--final-review", type=Path, default=None)
    road_review_cmd = sub.add_parser(
        "record-road-review", help="Record validated road-only decisions in alternative review v02"
    )
    road_review_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    history_cmd = sub.add_parser(
        "coordinate-history", help="Read-only date-by-date coordinate history for one spot"
    )
    history_cmd.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    history_cmd.add_argument("--spot-id", default="F-03")
    location_cmd = sub.add_parser(
        "link-location-check", help="Read-only compare spot coordinates with stable road link geometries"
    )
    location_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    location_cmd.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    location_cmd.add_argument("--spot-id", default="F-03")
    evidence_cmd = sub.add_parser(
        "record-f03-evidence", help="Save new F-03 spatial evidence JSON without modifying mapping decisions"
    )
    evidence_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    evidence_cmd.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    classify_cmd = sub.add_parser("classify-manual-reviews", help="Group manual review reasons without changing mapping")
    classify_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    classify_cmd.add_argument("--final-review", type=Path, default=None)
    classify_cmd.add_argument("--alternative-review", type=Path, default=None)
    nearby_cmd = sub.add_parser("assess-nearby-roads", help="Read-only evidence assessment for multiple-nearby-roads cases")
    nearby_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    nearby_cmd.add_argument("--classification", type=Path, default=None)
    nearby_cmd.add_argument("--candidates", type=Path, default=None)
    bulk_cmd = sub.add_parser("assess-all-manual", help="Read-only one-pass triage for all unreviewed versions")
    bulk_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    bulk_cmd.add_argument("--classification", type=Path, default=None)
    bulk_cmd.add_argument("--candidates", type=Path, default=None)
    history_assess_cmd = sub.add_parser("assess-coordinate-history", help="Read-only batch comparison of coordinate versions")
    history_assess_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    history_assess_cmd.add_argument("--classification", type=Path, default=None)
    history_assess_cmd.add_argument("--candidates", type=Path, default=None)
    history_assess_cmd.add_argument("--bulk-assessment", type=Path, default=None)
    full_cmd = sub.add_parser("evaluate-full-roads", help="Compare all analysis links by road, read-only")
    full_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    full_cmd.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    official_cmd = sub.add_parser("build-official-road-tables", help="Official road-name mapping and spot coverage")
    official_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    official_cmd.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    observation_cmd = sub.add_parser("join-official-observations", help="Row-preserving official road-name join for traffic observations")
    observation_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    observation_cmd.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    baseline_cmd = sub.add_parser("audit-baseline", help="Read-only weekday/hour baseline feasibility")
    baseline_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    baseline_cmd.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    date_cmd = sub.add_parser("audit-sampled-dates", help="Read-only year/month and weather-label readiness audit")
    date_cmd.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
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
