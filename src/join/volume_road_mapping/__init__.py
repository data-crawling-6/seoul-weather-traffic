"""[서연] 교통량 조사 지점 ↔ TOPIS 도로 · 링크 매핑 (단계별 모듈).

common 공통 · audit ① 입력 점검 · candidates ② 후보 · review ③ 검토 · official ④ 공식 매핑 · checks ⑤ 사후 점검
history 좌표 이력 · F-03 근거 · cli 실행 명령. 실행: python -m src.join.volume_road_mapping <명령>
"""
from __future__ import annotations

import pandas as pd  # noqa: F401  (테스트가 mapping.pd 로 씀)

from .common import DEFAULT_DATA_ROOT, STABLE_LINKS_RELATIVE, VERTEX_FILENAME, LINES_FILENAME, json_value, select_volume_files, normalize_link_ids, maximum_coordinate_distance, build_coordinate_review, git_provenance, input_file_provenance, audit_provenance, normalized_road_name  # noqa: F401
from .history import summarize_spot_coordinate_history, execute_coordinate_history, compare_spot_link_locations, execute_link_location_check, build_f03_evidence, execute_record_f03_evidence  # noqa: F401
from .audit import audit_vertices, audit_link_coverage, run_input_audit, AuditValidationError, allocate_output_paths, execute_audit  # noqa: F401
from .candidates import MAP_CANDIDATE_LIMIT, MAP_AUTO_MAX_DISTANCE_M, MAP_FAR_DISTANCE_M, MAP_LARGE_COORDINATE_CHANGE_M, MAP_AUTO_MAX_COORDINATE_CHANGE_M, build_mapping_versions, prepare_mapping_links, generate_mapping_candidates, allocate_map_paths, run_mapping, summarize_mapping_evidence, mapping_processing_steps, execute_map, analyze_candidate_frame, select_candidate_file, allocate_candidate_analysis_paths, execute_candidate_analysis  # noqa: F401
from .review import select_mapping_artifact_pair, review_alternative_candidates, validate_alternative_review, allocate_alternative_review_paths, ROAD_REVIEW_NOTES, ROAD_REVIEW_SPEC, ROAD_REVIEW_EDITABLE, ROAD_REVIEW_REQUIRED, select_alternative_review_v01, validate_road_review_source, apply_road_review, validate_road_review, road_review_output_paths, execute_record_road_review, execute_review_alternatives, classify_manual_review_frame, select_final_review_for_classification, execute_classify_manual_reviews, assess_nearby_road_frame, execute_assess_nearby_roads, assess_all_manual_frame, execute_assess_all_manual, assess_coordinate_history_frame, execute_assess_coordinate_history  # noqa: F401
from .official import evaluate_full_road_candidates, execute_full_road_evaluation, build_official_road_mapping, execute_official_road_mapping, build_official_observation_tables, execute_official_observation_join  # noqa: F401
from .checks import audit_traffic_baseline, execute_baseline_audit, audit_traffic_dates, execute_date_audit  # noqa: F401
from .cli import main  # noqa: F401
from . import audit, candidates, checks, cli, common, history, official, review  # noqa: F401,E402
