"""[서연] Unit tests for the reusable mapping input audit; no real input files required."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch


# [서연] Load the implementation directly without requiring src to be a package.
MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "traffic_mapping.py"
spec = importlib.util.spec_from_file_location("traffic_mapping", MODULE_PATH)
mapping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapping)


# [서연] unittest tests use synthetic data only; never run audit-inputs on actual files.
class TrafficMappingAuditTests(unittest.TestCase):
    # [서연] Choose the latest complete successful pair, not a partial/failed run.
    def test_select_volume_files_only_complete_success_pairs(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            for name in (
                "yang_seoyeon_traffic_volume_train_20261123T100000_v01_(성공).parquet",
                "yang_seoyeon_traffic_volume_eval_20261123T100000_v01_(성공).parquet",
                "yang_seoyeon_traffic_volume_train_20261123T110000_v01_(성공).parquet",
                "yang_seoyeon_traffic_volume_train_20261123T120000_v01_(실패).parquet",
            ):
                (folder / name).touch()
            chosen = mapping.select_volume_files(folder)
            self.assertIn("20261123T100000", chosen["train"].name)
            self.assertIn("20261123T100000", chosen["eval"].name)

    # [서연] Invalid coordinates are excluded from coordinate-version counts.
    def test_coordinate_review_preserves_invalid_and_changed_versions(self):
        df = pd.DataFrame({
            "spot_id": ["A-01", "A-01", "A-01", "B-01"],
            "io_type": ["in", "in", "in", "out"],
            "latitude": [0, 37.568588, 37.568808, 37.5],
            "longitude": [126.948436, 126.948436, 126.948774, 127.0],
        })
        review = mapping.build_coordinate_review(df)
        a01 = review.loc[review["spot_id"].eq("A-01")].iloc[0]
        self.assertEqual(a01["valid_coordinate_count"], 2)
        self.assertEqual(a01["zero_coordinate_rows"], 1)
        self.assertTrue(a01["coordinate_changed"])
        self.assertGreater(a01["max_coordinate_distance_m"], 0)
        self.assertEqual(a01["review_status"], "manual_review_needed")
        self.assertEqual(len(json.loads(a01["coordinate_versions"])), 2)

    # [서연] Check real Vertex coordinate-column variant and per-link vertex counts.
    def test_vertex_audit_grs80tm_and_minimum_vertices(self):
        vertices = pd.DataFrame({
            "link_id": ["1", "1", "2"],
            "grs80tm_x": [200, 201, None],
            "grs80tm_y": [300, 301, 302],
            "vertex_order": [1, 2, 1],
        })
        result = mapping.audit_vertices(vertices)
        self.assertEqual(result["coordinate_columns"], ["grs80tm_x", "grs80tm_y"])
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["unique_links"], 2)
        self.assertEqual(result["missing_coordinate_rows"], 1)
        self.assertEqual(result["minimum_vertices_per_link"], 1)
        self.assertEqual(result["links_with_fewer_than_two_vertices"], 1)
        self.assertEqual(result["invalid_vertex_order_rows"], 0)

    # [서연] Missing/duplicate/non-integer vertex_order must be flagged.
    def test_vertex_order_validation(self):
        vertices = pd.DataFrame({
            "link_id": ["1", "1", "1", "2"],
            "grs80tm_x": [1, 2, 3, 4],
            "grs80tm_y": [1, 2, 3, 4],
            "vertex_order": [1, 1, None, 1.5],
        })
        result = mapping.audit_vertices(vertices)
        self.assertEqual(result["missing_vertex_order_rows"], 1)
        self.assertEqual(result["invalid_vertex_order_rows"], 2)
        self.assertEqual(result["duplicate_vertex_order_within_link_rows"], 1)
        with self.assertRaisesRegex(ValueError, "vertex_order"):
            mapping.audit_vertices(vertices.drop(columns=["vertex_order"]))

    # [서연] Verify duplicate link assignments against synthetic LineStrings.
    def test_link_coverage_detects_duplicate_assignments(self):
        import geopandas as gpd
        from shapely.geometry import LineString
        stable = pd.DataFrame({
            "link_id": ["1", "1", "2"],
            "road_id": ["R001", "R002", "R002"],
            "road_name": ["a", "b", "b"],
            "direction": ["up", "down", "down"],
        })
        lines = gpd.GeoDataFrame({
            "link_id": ["1", "2"],
            "geometry": [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        }, geometry="geometry", crs="EPSG:4326")
        result = mapping.audit_link_coverage(stable, lines)
        self.assertEqual(result["matched_analysis_links"], 2)
        self.assertEqual(result["unmatched_analysis_links"], 0)
        self.assertEqual(result["duplicate_stable_link_rows"], 1)
        self.assertEqual(result["links_assigned_to_multiple_roads"], 1)

    # [서연] Fingerprints are calculated from synthetic files, not production data.
    def test_input_file_provenance_sha256_and_size(self):
        import hashlib
        with TemporaryDirectory() as temp:
            path = Path(temp) / "sample.csv"
            payload = b"link_id,road_id\n1,R001\n"
            path.write_bytes(payload)
            result = mapping.input_file_provenance(path)
            self.assertEqual(result["size_bytes"], len(payload))
            self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(result["path"], str(path))

    # [서연] Git status and Python runtime are recorded without changing the repository.
    def test_audit_provenance_records_git_and_python(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.csv"
            source.write_text("sample", encoding="utf-8")
            with patch.object(mapping, "git_provenance", return_value={
                "commit_id": "a" * 40,
                "has_uncommitted_changes": True,
                "git_error": None,
            }):
                result = mapping.audit_provenance(root, {"sample": source})
            self.assertEqual(result["git"]["commit_id"], "a" * 40)
            self.assertTrue(result["git"]["has_uncommitted_changes"])
            self.assertTrue(result["python"]["executable"])
            self.assertTrue(result["python"]["version"])
            self.assertEqual(result["input_files"]["sample"]["size_bytes"], 6)

    # [서연] Existing timestamped output names must not be overwritten.
    def test_output_paths_are_unique(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            first_json, _ = mapping.allocate_output_paths(root, "20261123T220000", True)
            first_json.touch()
            second_json, _ = mapping.allocate_output_paths(root, "20261123T220000", True)
            self.assertNotEqual(first_json, second_json)
            self.assertIn("_002_", second_json.name)

    # [서연] Simulated failures record a fresh failure JSON without fabricated review data.
    def test_execute_audit_failure_writes_real_timestamped_json(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(mapping, "run_input_audit", side_effect=FileNotFoundError("test missing input")):
                status = mapping.execute_audit(root, root)
            self.assertEqual(status, 1)
            files = list((root / "logs" / "yang_seoyeon" / "traffic_mapping").glob("*_(실패).json"))
            self.assertEqual(len(files), 1)
            payload = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failure")
            self.assertIsNone(payload["coordinate_review_path"])
            self.assertIn("Fresh audit", payload["record_notice"])


    # [서연] Map every coordinate version and retain invalid coordinates as unmapped.
    def test_map_coordinate_versions_and_conservative_review(self):
        import geopandas as gpd
        from shapely.geometry import LineString
        import numpy as np

        volume = pd.DataFrame({
            "spot_id": ["A", "A", "A", "B"],
            "io_type": ["in", "in", "in", "out"],
            "road_name": ["Road A", "Road A", "Road A", "Road B"],
            "latitude": [37.50, 37.51, 0, 37.55],
            "longitude": [127.00, 127.01, 127.00, 127.05],
        })
        # [서연] Patch only the link validator's expected production counts for synthetic inputs.
        stable = pd.DataFrame({
            "link_id": ["1", "2", "3", "4"],
            "road_id": ["R1", "R1", "R1", "R2"],
            "road_name": ["Road A", "Road A", "Road A", "Road B"],
            "direction": ["up", "down", "up", "up"],
        })
        lines = gpd.GeoDataFrame({
            "link_id": ["1", "2", "3", "4"],
            "geometry": [
                LineString([(127, 37.50), (127.0002, 37.50)]),
                LineString([(127, 37.5001), (127.0002, 37.5001)]),
                LineString([(127, 37.5002), (127.0002, 37.5002)]),
                LineString([(127.05, 37.55), (127.0502, 37.55)]),
            ],
        }, crs="EPSG:4326")
        original = mapping.audit_link_coverage
        def synthetic_coverage(st, geo):
            result = original(st, geo)
            result["expected_1753_links"] = True
            result["expected_90_roads"] = True
            return result
        with patch.object(mapping, "audit_link_coverage", side_effect=synthetic_coverage):
            candidates, review = mapping.generate_mapping_candidates(volume, stable, lines)
        self.assertEqual(len(review), 4)
        self.assertEqual(review["mapping_status"].eq("unmapped").sum(), 1)
        self.assertEqual(len(candidates), 9)
        self.assertTrue((candidates.groupby("coordinate_version_id").size() <= 3).all())
        self.assertTrue((review.loc[review["spot_id"].eq("A"), "mapping_status"] != "auto_confirmed_road_only").all())
        self.assertIn("direction_comparison", candidates.columns)

    # [서연] Map output names are unique and cannot overwrite earlier evidence.
    def test_map_output_paths_are_unique(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            first = mapping.allocate_map_paths(root, "20261123T235959", True)
            first["json"].touch()
            second = mapping.allocate_map_paths(root, "20261123T235959", True)
            self.assertNotEqual(first["json"], second["json"])
            self.assertNotIn("yang_seoyeon", second["json"].name)

    # [서연] The map command is registered without running any real map.
    def test_map_cli_registration(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(mapping, "execute_map", return_value=0) as mock_run:
                result = mapping.main(["map", "--project-root", str(root), "--data-root", str(root)])
            self.assertEqual(result, 0)
            mock_run.assert_called_once_with(root, root)

    # [서연] Preserve multi-label review reason totals and combined reason patterns.
    def test_mapping_evidence_counts_and_checks(self):
        review = pd.DataFrame({
            "coordinate_version_id": ["V1", "V2", "V3"],
            "spot_id": ["A", "B", "C"],
            "io_type": ["in", "out", "in"],
            "source_rows": [3, 2, 1],
            "mapping_status": ["auto_confirmed_road_only", "manual_review_needed", "unmapped"],
            "candidate_count": [2, 1, 0],
            "review_reasons": ["", "multiple_nearby_roads;far_from_analysis_links", "invalid_coordinate"],
            "selected_link_id": ["11", None, None],
            "selected_road_id": ["R1", None, None],
        })
        candidates = pd.DataFrame({
            "coordinate_version_id": ["V1", "V1", "V2"],
            "candidate_rank": [1, 2, 1],
        })
        evidence = mapping.summarize_mapping_evidence(candidates, review)
        self.assertTrue(evidence["validation_passed"])
        self.assertEqual(evidence["review_reasons_multi_label"]["multiple_nearby_roads"], 1)
        self.assertEqual(evidence["status_counts"]["unmapped"], 1)
        self.assertEqual(len(mapping.mapping_processing_steps()), 7)

    # [서연] Detect broken candidate counts before saving a successful map report.
    def test_mapping_evidence_rejects_candidate_count_mismatch(self):
        review = pd.DataFrame({
            "coordinate_version_id": ["V1"], "spot_id": ["A"], "io_type": ["in"],
            "source_rows": [2], "mapping_status": ["manual_review_needed"],
            "candidate_count": [2], "review_reasons": ["multiple_nearby_roads"],
            "selected_link_id": [None], "selected_road_id": [None],
        })
        candidates = pd.DataFrame({"coordinate_version_id": ["V1"], "candidate_rank": [1]})
        evidence = mapping.summarize_mapping_evidence(candidates, review)
        self.assertFalse(evidence["validation_passed"])
        self.assertFalse(evidence["validation_checks"]["candidate_count_matches_review"])

    # [서연] Candidate diagnostics report mismatches without changing mapping decisions.
    def test_analyze_candidate_frame(self):
        candidates = pd.DataFrame({
            "coordinate_version_id": ["V1", "V1", "V2", "V2"],
            "candidate_rank": [1, 2, 1, 2],
            "road_name_match": [False, True, True, True],
            "spot_id": ["A", "A", "B", "B"],
            "io_type": ["in", "in", "out", "out"],
            "road_id": ["R2", "R1", "R3", "R3"],
            "road_name": ["Road A", "Road A", "Road B", "Road B"],
            "link_road_name": ["Road X", "Road A", "Road B", "Road B"],
            "link_id": ["2", "1", "3", "4"],
            "distance_m": [10.0, 20.0, 5.0, 15.0],
            "mapping_status": ["manual_review_needed"] * 2 + ["auto_confirmed_road_only"] * 2,
            "review_reasons": ["nearest_road_name_mismatch"] * 2 + [""] * 2,
        })
        counts, pairs, detail = mapping.analyze_candidate_frame(candidates)
        self.assertEqual(counts["nearest_road_name_mismatch_versions"], 1)
        self.assertEqual(counts["nearest_mismatch_with_alternative_matching_candidate_within_50m"], 1)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(bool(detail.iloc[0]["alternative_matching_candidate_within_50m"]))
        self.assertTrue(bool(detail.iloc[0]["multiple_candidate_roads"]))

    # [서연] Candidate diagnostic output names are unique and never overwrite prior runs.
    def test_candidate_analysis_paths_are_unique(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            first = mapping.allocate_candidate_analysis_paths(root, "20261123T235959", True)
            first["json"].touch()
            second = mapping.allocate_candidate_analysis_paths(root, "20261123T235959", True)
            self.assertNotEqual(first["json"], second["json"])
            self.assertIn("_002_", second["json"].name)

    # [서연] Register analyze-candidates without reading or writing production data.
    def test_candidate_analysis_cli_registration(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "candidates.csv"
            with patch.object(mapping, "execute_candidate_analysis", return_value=0) as mock_run:
                result = mapping.main([
                    "analyze-candidates", "--project-root", str(root),
                    "--candidates", str(source),
                ])
            self.assertEqual(result, 0)
            mock_run.assert_called_once_with(root, source)

    # [서연] Alternative review suggests eligible links without changing decisions.
    def test_review_alternative_candidates_is_read_only(self):
        candidates = pd.DataFrame({
            "coordinate_version_id": ["V1", "V1", "V1", "V2", "V2"],
            "candidate_rank": [1, 2, 3, 1, 2],
            "link_id": ["11", "12", "13", "21", "22"],
            "road_id": ["RX", "R1", "R1", "R2", "R2"],
            "link_road_name": ["Wrong", "Road A", "Road A", "Road B", "Road B"],
            "distance_m": [10.0, 20.0, 70.0, 5.0, 15.0],
            "road_name_match": [False, True, True, True, True],
        })
        review = pd.DataFrame({
            "coordinate_version_id": ["V1", "V2"],
            "spot_id": ["A", "B"], "io_type": ["in", "out"],
            "road_name": ["Road A", "Road B"],
            "mapping_status": ["manual_review_needed", "auto_confirmed_road_only"],
            "review_reasons": ["nearest_road_name_mismatch", ""],
            "selected_link_id": [None, "21"], "selected_road_id": [None, "R2"],
            "nearest_candidate_link_id": ["11", "21"],
            "nearest_candidate_road_id": ["RX", "R2"],
            "nearest_distance_m": [10.0, 5.0],
            "nearest_road_name_match": [False, True],
            "valid_coordinate": [True, True],
            "max_coordinate_change_m": [20.0, 10.0],
            "invalid_rows_in_spot_direction": [0, 0],
        })
        result = mapping.review_alternative_candidates(candidates, review)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["alternative_link_id"], "12")
        self.assertEqual(result.iloc[0]["mapping_status"], "manual_review_needed")
        self.assertTrue(bool(result.iloc[0]["coordinate_stable_for_auto_rule"]))
        checks = mapping.validate_alternative_review(result, candidates, review)
        self.assertTrue(all(checks.values()))

    # [서연] Select candidate and final-review artifacts only from the same successful run.
    def test_select_mapping_artifact_pair(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            tables = root / "outputs" / "tables"
            tables.mkdir(parents=True)
            for name in (
                "yang_seoyeon_mapping_candidates_20261123T100000_(성공)_v01.csv",
                "yang_seoyeon_mapping_final_review_20261123T100000_(성공)_v01.csv",
                "yang_seoyeon_mapping_candidates_20261123T110000_(성공)_v01.csv",
            ):
                (tables / name).touch()
            candidates, review = mapping.select_mapping_artifact_pair(root)
            self.assertIn("20261123T100000", candidates.name)
            self.assertIn("20261123T100000", review.name)

    # [서연] Register review-alternatives without reading production data.
    def test_review_alternatives_cli_registration(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            candidates = root / "candidates.csv"
            review = root / "review.csv"
            with patch.object(mapping, "execute_review_alternatives", return_value=0) as mock_run:
                result = mapping.main([
                    "review-alternatives", "--project-root", str(root),
                    "--candidates", str(candidates), "--final-review", str(review),
                ])
            self.assertEqual(result, 0)
            mock_run.assert_called_once_with(root, candidates, review)

    # [서연] Nine synthetic rows model seven coordinate versions and two duplicate candidates.
    def road_review_fixture(self):
        rows = []
        for version, (spot, road_id, _decision, count) in mapping.ROAD_REVIEW_SPEC.items():
            for i in range(count):
                rows.append({
                    "coordinate_version_id": version, "spot_id": spot,
                    "road_name": spot, "mapping_status": "manual_review_needed",
                    "selected_link_id": "", "selected_road_id": "",
                    "alternative_link_id": f"{version}_{i}",
                    "alternative_road_id": road_id or "R999",
                    "alternative_distance_m": str(10 + i),
                    "alternative_road_name_match": "True",
                    "latitude": "37.123400", "longitude": "127.123400",
                    "manual_decision": "", "manual_review_notes": "",
                })
        return pd.DataFrame(rows, dtype="string")

    # [서연] Apply exactly seven road-only rows and two pending rows.
    def test_road_review_approved_decisions(self):
        source = self.road_review_fixture()
        output = mapping.apply_road_review(source)
        checks = mapping.validate_road_review(source, output)
        self.assertTrue(all(checks.values()), checks)
        self.assertEqual(output["manual_decision"].value_counts()["road_only_confirmed"], 7)
        self.assertEqual(output["manual_decision"].value_counts()["pending_review"], 2)
        self.assertTrue(source["manual_decision"].eq("").all())
        self.assertIn("road_id=R009", output.loc[output["spot_id"].eq("F-09"), "manual_review_notes"].iloc[0])

    # [서연] Both rows for D-39 and F-03 must have identical decisions and notes.
    def test_road_review_duplicate_version_consistency(self):
        source = self.road_review_fixture()
        output = mapping.apply_road_review(source)
        for version in ("V00555", "V00576"):
            group = output.loc[output["coordinate_version_id"].eq(version)]
            self.assertEqual(len(group), 2)
            self.assertEqual(group["manual_decision"].nunique(), 1)
            self.assertEqual(group["manual_review_notes"].nunique(), 1)
        output.loc[output["coordinate_version_id"].eq("V00555"), "manual_review_notes"] = ["different", "same"]
        self.assertFalse(mapping.validate_road_review(source, output)["duplicate_version_consistency"])

    # [서연] Unknown/missing coordinate versions and incorrect candidate counts are rejected.
    def test_road_review_rejects_invalid_versions(self):
        source = self.road_review_fixture()
        source.loc[0, "coordinate_version_id"] = "V99999"
        with self.assertRaises(ValueError):
            mapping.apply_road_review(source)
        source = self.road_review_fixture().iloc[:-1]
        with self.assertRaises(ValueError):
            mapping.apply_road_review(source)

    # [서연] Detect changes to any field other than the two review columns.
    def test_road_review_detects_immutable_field_change(self):
        source = self.road_review_fixture()
        output = mapping.apply_road_review(source)
        for field in ("alternative_link_id", "alternative_distance_m", "alternative_road_name_match", "mapping_status", "selected_link_id", "selected_road_id", "latitude"):
            changed = output.copy()
            changed.loc[0, field] = "changed"
            self.assertFalse(mapping.validate_road_review(source, changed)["all_other_columns_unchanged"], field)

    # [서연] Exact column order and row order are immutable.
    def test_road_review_detects_row_and_column_reordering(self):
        source = self.road_review_fixture()
        output = mapping.apply_road_review(source)
        self.assertFalse(mapping.validate_road_review(source, output.iloc[::-1].reset_index(drop=True))["all_other_columns_unchanged"])
        changed = output[output.columns[::-1]]
        self.assertFalse(mapping.validate_road_review(source, changed)["same_column_order"])

    # [서연] Round-trip a CSV buffer with all columns kept as strings and blanks preserved.
    def test_road_review_csv_roundtrip(self):
        import io
        source = self.road_review_fixture()
        output = mapping.apply_road_review(source)
        buffer = io.StringIO()
        output.to_csv(buffer, index=False)
        roundtrip = pd.read_csv(io.StringIO(buffer.getvalue()), dtype="string", keep_default_na=False)
        self.assertTrue(all(mapping.validate_road_review(source, roundtrip).values()))

    # [서연] Latest successful v01 is chosen automatically; failed and v02 are ignored.
    def test_road_review_selects_latest_successful_v01(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            tables = root / "outputs" / "tables"
            tables.mkdir(parents=True)
            for stamp, status, version in (("20261123T100000", "성공", "v01"), ("20261123T110000", "성공", "v01"), ("20261123T120000", "실패", "v01"), ("20261123T130000", "성공", "v02")):
                (tables / f"yang_seoyeon_alternative_link_review_{stamp}_({status})_{version}.csv").touch()
            chosen = mapping.select_alternative_review_v01(root)
            self.assertIn("20261123T110000", chosen.name)

    # [서연] v02 must never overwrite an existing file; the original v01 stays unchanged.
    def test_road_review_v02_no_overwrite(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            tables = root / "outputs" / "tables"
            tables.mkdir(parents=True)
            source_path = tables / "yang_seoyeon_alternative_link_review_20261123T100000_(성공)_v01.csv"
            self.road_review_fixture().to_csv(source_path, index=False, encoding="utf-8-sig")
            paths = mapping.road_review_output_paths(root, source_path)
            paths["review"].write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                mapping.execute_record_road_review(root)
            self.assertEqual(paths["review"].read_text(encoding="utf-8"), "existing")
            self.assertTrue(source_path.is_file())

    # [서연] Synthetic end-to-end recording verifies both SHA-256 hashes and Git provenance.
    def test_road_review_sha256_git_audit(self):
        import hashlib
        with TemporaryDirectory() as temp:
            root = Path(temp)
            tables = root / "outputs" / "tables"
            tables.mkdir(parents=True)
            source_path = tables / "yang_seoyeon_alternative_link_review_20261123T100000_(성공)_v01.csv"
            self.road_review_fixture().to_csv(source_path, index=False, encoding="utf-8-sig")
            before = source_path.read_bytes()
            with patch.object(mapping, "git_provenance", return_value={"commit_id": "a" * 40, "has_uncommitted_changes": True, "git_error": None}):
                self.assertEqual(mapping.execute_record_road_review(root), 0)
            paths = mapping.road_review_output_paths(root, source_path)
            report = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "success")
            self.assertEqual(report["git"]["commit_id"], "a" * 40)
            self.assertTrue(report["git"]["has_uncommitted_changes"])
            self.assertEqual(report["input_v01"]["sha256"], hashlib.sha256(before).hexdigest())
            self.assertEqual(report["output_v02"]["sha256"], hashlib.sha256(paths["review"].read_bytes()).hexdigest())
            self.assertTrue(all(report["validation_checks"].values()))
            self.assertTrue(all(report["roundtrip_validation_checks"].values()))
            self.assertEqual(source_path.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                mapping.execute_record_road_review(root)

    # [서연] CLI uses only --project-root; no --road-review input is required.
    def test_record_road_review_cli_registration(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(mapping, "execute_record_road_review", return_value=0) as mock_run:
                self.assertEqual(mapping.main(["record-road-review", "--project-root", str(root)]), 0)
            mock_run.assert_called_once_with(root)


    # [서연] Same-day mixed coordinates must remain visible, not silently collapsed.
    def test_coordinate_history_preserves_daily_variants(self):
        frame = pd.DataFrame({
            "date": ["2026-11-01", "2026-11-02", "2026-11-02", "2026-11-03", "2026-11-01"],
            "spot_id": ["F-03"] * 5,
            "io_type": ["유출"] * 4 + ["유입"],
            "latitude": [37.6, 37.6, 37.5, 37.5, 37.6],
            "longitude": [127.0, 127.0, 126.9, 126.9, 127.0],
        })
        result = mapping.summarize_spot_coordinate_history(frame)
        self.assertEqual(result["rows"], 5)
        outbound = next(x for x in result["directions"] if x["io_type"] == "유출")
        self.assertEqual(len(outbound["coordinate_variants"]), 2)
        self.assertEqual(outbound["days_with_multiple_coordinates"], ["2026-11-02"])
        self.assertEqual(len(outbound["daily_coordinate_rows"]), 4)
        self.assertEqual(sum(x["rows"] for x in outbound["coordinate_variants"]), 4)

    # [서연] Missing fields and invalid dates must fail without producing output.
    def test_coordinate_history_rejects_missing_or_invalid_dates(self):
        frame = pd.DataFrame({"date": ["not-a-date"], "spot_id": ["F-03"],
                              "io_type": ["유출"], "latitude": [37.6], "longitude": [127.0]})
        with self.assertRaisesRegex(ValueError, "Unparseable date"):
            mapping.summarize_spot_coordinate_history(frame)
        with self.assertRaisesRegex(ValueError, "missing columns"):
            mapping.summarize_spot_coordinate_history(frame.drop(columns=["date"]))
        with self.assertRaisesRegex(ValueError, "No volume rows"):
            mapping.summarize_spot_coordinate_history(frame, "X-00")

    # [서연] The CLI delegates only to the read-only history command.
    def test_coordinate_history_cli_registration(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(mapping, "execute_coordinate_history", return_value=0) as run:
                self.assertEqual(mapping.main(["coordinate-history", "--data-root", str(root),
                                               "--spot-id", "F-03"]), 0)
            run.assert_called_once_with(root, "F-03")

    # [서연] Read only synthetic parquet inputs and do not create any output files.
    def test_coordinate_history_execution_read_only(self):
        import io
        from contextlib import redirect_stdout
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "traffic_volume" / "processed"
            folder.mkdir(parents=True)
            train = folder / "yang_seoyeon_traffic_volume_train_20261123T100000_v01_(성공).parquet"
            evaluation = folder / "yang_seoyeon_traffic_volume_eval_20261123T100000_v01_(성공).parquet"
            frame = pd.DataFrame({"date": ["2026-11-01"], "spot_id": ["F-03"],
                                  "io_type": ["유출"], "latitude": [37.6], "longitude": [127.0]})
            train.write_bytes(b"train synthetic")
            evaluation.write_bytes(b"eval synthetic")
            before = {p: p.read_bytes() for p in folder.iterdir()}
            with patch.object(mapping.pd, "read_parquet", return_value=frame):
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(mapping.execute_coordinate_history(root), 0)
            self.assertIn("2026-11-01", output.getvalue())
            self.assertIn("SHA-256", output.getvalue())
            self.assertEqual({p: p.read_bytes() for p in folder.iterdir()}, before)
            self.assertFalse((root / "outputs").exists())
            self.assertFalse((root / "logs").exists())


    # [서연] Distances use projected geometry and preserve direction uncertainty.
    def test_link_location_check_nearest_and_matching_road(self):
        import geopandas as gpd
        from shapely.geometry import LineString
        volume = pd.DataFrame({
            "spot_id": ["F-03", "F-03"], "io_type": ["유출", "유출"],
            "road_name": ["내부순환로"] * 2, "direction_detail": ["홍지문터널→홍은램프"] * 2,
            "latitude": [37.6, 37.6], "longitude": [127.0, 127.01],
        })
        stable = pd.DataFrame({"link_id": ["1", "2"], "road_id": ["R1", "R2"],
                               "road_name": ["다른도로", "내부순환로"], "direction": ["상행", "하행"]})
        lines = gpd.GeoDataFrame({"link_id": ["1", "2"], "geometry": [
            LineString([(127.0, 37.6), (127.0, 37.601)]),
            LineString([(127.01, 37.6), (127.01, 37.601)])]}, crs="EPSG:4326")
        result = mapping.compare_spot_link_locations(volume, stable, lines)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["nearest_link"]["link_id"], "1")
        self.assertEqual(result[0]["nearest_same_name_link"]["link_id"], "2")
        self.assertEqual(result[1]["nearest_link"]["link_id"], "2")
        self.assertIn("unconfirmed", result[0]["direction_verdict"])
        self.assertEqual(result[0]["rows"], 1)

    # [서연] Reject missing/duplicated link geometry rather than inventing spatial matches.
    def test_link_location_check_rejects_incomplete_geometry(self):
        import geopandas as gpd
        from shapely.geometry import LineString
        volume = pd.DataFrame({"spot_id": ["F-03"], "io_type": ["유출"],
                               "road_name": ["A"], "direction_detail": ["A to B"],
                               "latitude": [37.6], "longitude": [127.0]})
        stable = pd.DataFrame({"link_id": ["1", "2"], "road_id": ["R1", "R2"],
                               "road_name": ["A", "B"], "direction": ["up", "down"]})
        lines = gpd.GeoDataFrame({"link_id": ["1"], "geometry": [
            LineString([(127, 37.6), (127.001, 37.6)])]}, crs="EPSG:4326")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            mapping.compare_spot_link_locations(volume, stable, lines)

    # [서연] CLI dispatch must not call mapping or write output files.
    def test_link_location_check_cli_registration(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(mapping, "execute_link_location_check", return_value=0) as run:
                self.assertEqual(mapping.main(["link-location-check", "--project-root", str(root),
                                               "--data-root", str(root), "--spot-id", "F-03"]), 0)
            run.assert_called_once_with(root, root, "F-03")


    # [서연] Build evidence from synthetic R005 geometries without selecting a mapping.
    def test_f03_evidence_geometry_and_pending_status(self):
        import geopandas as gpd
        from shapely.geometry import LineString
        volume = pd.DataFrame({
            "date": ["2025-09-17", "2025-11-28"], "spot_id": ["F-03", "F-03"],
            "io_type": ["유출", "유출"], "road_name": ["내부순환로"] * 2,
            "direction_detail": ["홍지문터널→홍은램프"] * 2,
            "latitude": [37.608679, 37.596363],
            "longitude": [126.998878, 126.95411397],
        })
        stable = pd.DataFrame({"link_id": ["1120002300", "1120002100", "1120001700"],
                               "road_id": ["R005"] * 3, "road_name": ["내부순환로"] * 3,
                               "direction": ["하행"] * 3})
        lines = gpd.GeoDataFrame({"link_id": stable["link_id"], "geometry": [
            LineString([(126.957, 37.596), (126.955589, 37.596091)]),
            LineString([(126.955589, 37.596091), (126.95003, 37.595215),
                        (126.95411397, 37.596363)]),
            LineString([(126.95411397, 37.596363), (126.949, 37.595)]),
        ]}, crs="EPSG:4326")
        # [서연] Construct a properly connected synthetic chain.
        lines.at[1, "geometry"] = LineString([(126.955589, 37.596091),
                                                (126.95411397, 37.596363), (126.95003, 37.595215)])
        lines.at[2, "geometry"] = LineString([(126.95003, 37.595215), (126.949, 37.595)])
        result = mapping.build_f03_evidence(volume, stable, lines)
        self.assertEqual(result["review_status"], "pending_review")
        self.assertEqual(len(result["geometry_endpoint_gaps"]), 2)
        self.assertLess(max(x["geometry_end_to_start_gap_m"] for x in result["geometry_endpoint_gaps"]), 0.01)
        self.assertIn("unconfirmed", result)
        self.assertEqual(result["r005_link_direction_counts"], {"하행": 3})

    # [서연] Logging is exclusive, versioned and does not touch existing artifacts.
    def test_f03_evidence_log_success_and_collision(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "traffic_volume" / "processed"
            folder.mkdir(parents=True)
            for split in ("train", "eval"):
                (folder / f"yang_seoyeon_traffic_volume_{split}_20261123T100000_v01_(성공).parquet").write_bytes(b"test")
            with patch.object(mapping.pd, "read_parquet", return_value=pd.DataFrame()), \
                 patch.object(mapping.pd, "read_csv", return_value=pd.DataFrame()), \
                 patch("geopandas.read_file", return_value=None), \
                 patch.object(mapping, "build_f03_evidence", return_value={"review_status": "pending_review"}), \
                 patch.object(mapping, "git_provenance", return_value={}):
                self.assertEqual(mapping.execute_record_f03_evidence(root, root), 0)
                self.assertEqual(mapping.execute_record_f03_evidence(root, root), 0)
            outputs = list((root / "logs" / "yang_seoyeon" / "traffic_mapping").glob("f03_link_evidence_*_(성공)_v01.json"))
            self.assertEqual(len(outputs), 2)
            for output in outputs:
                payload = json.loads(output.read_text(encoding="utf-8"))
                self.assertFalse(payload["mapping_modified"])
                self.assertEqual(payload["evidence"]["review_status"], "pending_review")
            self.assertFalse((root / "outputs").exists())

    # [서연] Failed evidence collection writes a failure log, not a false success.
    def test_f03_evidence_log_failure(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(mapping, "git_provenance", return_value={}):
                self.assertEqual(mapping.execute_record_f03_evidence(root, root), 1)
            outputs = list((root / "logs" / "yang_seoyeon" / "traffic_mapping").glob("*_(실패)_v01.json"))
            self.assertEqual(len(outputs), 1)
            self.assertIsNone(json.loads(outputs[0].read_text(encoding="utf-8"))["evidence"])

    # [서연] New CLI only dispatches to evidence writer when explicitly invoked.
    def test_f03_evidence_cli_dispatch(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(mapping, "execute_record_f03_evidence", return_value=0) as run:
                self.assertEqual(mapping.main(["record-f03-evidence", "--project-root", str(root),
                                               "--data-root", str(root)]), 0)
            run.assert_called_once_with(root, root)



class BulkManualAssessmentTests(unittest.TestCase):
    # [서연] Verify all unreviewed versions are assessed without confirming links.
    def test_bulk_triage_preserves_unreviewed_and_excludes_previously_reviewed(self):
        review = pd.DataFrame([
            {"coordinate_version_id":"a","mapping_status":"manual_review_needed","previous_review_decision":"",
             "review_reasons":"multiple_nearby_roads","road_name":"테스트로","spot_name":"일반지점","direction_detail":""},
            {"coordinate_version_id":"b","mapping_status":"manual_review_needed","previous_review_decision":"",
             "review_reasons":"large_coordinate_change;coordinate_change_exceeds_auto_threshold","road_name":"지하도로","spot_name":"터널","direction_detail":""},
            {"coordinate_version_id":"c","mapping_status":"manual_review_needed","previous_review_decision":"road_only_confirmed",
             "review_reasons":"multiple_nearby_roads","road_name":"테스트로","spot_name":"","direction_detail":""},
        ])
        candidates = pd.DataFrame([
            {"coordinate_version_id":"a","candidate_rank":1,"road_id":"R001","link_id":"1","distance_m":3,
             "road_name_match":True,"direction_comparison":"not_comparable_without_verified_crosswalk"},
            {"coordinate_version_id":"a","candidate_rank":2,"road_id":"R002","link_id":"2","distance_m":30,
             "road_name_match":False,"direction_comparison":"not_comparable_without_verified_crosswalk"},
            {"coordinate_version_id":"b","candidate_rank":1,"road_id":"R003","link_id":"3","distance_m":2,
             "road_name_match":True,"direction_comparison":"not_comparable_without_verified_crosswalk"},
            {"coordinate_version_id":"b","candidate_rank":2,"road_id":"R004","link_id":"4","distance_m":40,
             "road_name_match":False,"direction_comparison":"not_comparable_without_verified_crosswalk"},
        ])
        result, report = mapping.assess_all_manual_frame(review, candidates)
        self.assertEqual(len(result), 2)
        self.assertEqual(report["unreviewed_versions"], 2)
        self.assertTrue(all(report["validation_checks"].values()))
        a = result.set_index("coordinate_version_id").loc["a"]
        b = result.set_index("coordinate_version_id").loc["b"]
        self.assertEqual(a["assessment"], "road_only_priority_for_validation")
        self.assertEqual(b["assessment"], "manual_review_required")
        self.assertEqual(b["next_check"], "coordinate_history_consistency")
        self.assertTrue(bool(b["coordinate_history_candidate_for_review"]))
        self.assertEqual(report["next_check_counts"]["coordinate_history_consistency"], 1)
        self.assertEqual(report["name_alias_candidates_for_review"], 0)
        self.assertIn("underground_or_underpass", b["structure_risk_flags"])
        self.assertTrue(result["mapping_decision"].eq("unchanged").all())

    # [서연] No saved candidate must not become an automatic road confirmation.
    def test_bulk_triage_missing_candidates_requires_manual_review(self):
        review = pd.DataFrame([{"coordinate_version_id":"z","mapping_status":"manual_review_needed",
            "previous_review_decision":"","review_reasons":"nearest_road_name_mismatch","road_name":"테스트로"}])
        candidates = pd.DataFrame(columns=["coordinate_version_id","candidate_rank","road_id","link_id",
                                            "distance_m","road_name_match","direction_comparison"])
        result, report = mapping.assess_all_manual_frame(review, candidates)
        self.assertEqual(result.iloc[0]["assessment"], "manual_review_required")
        self.assertTrue(all(report["validation_checks"].values()))


class CoordinateHistoryAssessmentTests(unittest.TestCase):
    # [서연] Compare multiple versions in the same spot-direction without confirming links.
    def test_same_road_candidate_across_coordinate_versions(self):
        classified = pd.DataFrame([
            {"coordinate_version_id":"v1","spot_id":"A-01","io_type":"유입","latitude":"37.5",
             "longitude":"127.0","road_name":"테스트로","mapping_status":"manual_review_needed",
             "previous_review_decision":""},
            {"coordinate_version_id":"v2","spot_id":"A-01","io_type":"유입","latitude":"37.5001",
             "longitude":"127.0001","road_name":"테스트로","mapping_status":"manual_review_needed",
             "previous_review_decision":""},
        ])
        candidates = pd.DataFrame([
            {"coordinate_version_id":"v1","road_id":"R001","distance_m":"2","road_name_match":"True"},
            {"coordinate_version_id":"v2","road_id":"R001","distance_m":"4","road_name_match":"True"},
        ])
        bulk = pd.DataFrame([{"coordinate_version_id":"v1",
            "next_check":"coordinate_history_consistency",
            "coordinate_history_candidate_for_review":"True"}])
        result, report = mapping.assess_coordinate_history_frame(classified, candidates, bulk)
        self.assertEqual(report["target_versions"], 1)
        self.assertTrue(all(report["validation_checks"].values()))
        self.assertEqual(result.iloc[0]["history_evidence_status"], "same_road_candidate_across_versions")
        self.assertGreater(result.iloc[0]["max_coordinate_shift_m"], 0)
        self.assertEqual(result.iloc[0]["mapping_decision"], "unchanged")

    # [서연] A missing peer in the classification must remain explicitly unverified.
    def test_no_peer_does_not_claim_road_consistency(self):
        classified = pd.DataFrame([{"coordinate_version_id":"v1","spot_id":"A-01",
            "io_type":"유입","latitude":"37.5","longitude":"127.0",
            "road_name":"테스트로","mapping_status":"manual_review_needed","previous_review_decision":""}])
        candidates = pd.DataFrame([{"coordinate_version_id":"v1","road_id":"R001",
            "distance_m":"2","road_name_match":"True"}])
        bulk = pd.DataFrame([{"coordinate_version_id":"v1",
            "next_check":"coordinate_history_consistency",
            "coordinate_history_candidate_for_review":"True"}])
        result, report = mapping.assess_coordinate_history_frame(classified, candidates, bulk)
        self.assertEqual(result.iloc[0]["history_evidence_status"], "no_peer_in_classification")
        self.assertTrue(all(report["validation_checks"].values()))

class FullRoadEvaluationTests(unittest.TestCase):
    # [서연] Full-road evaluation finds matching roads beyond the nearest three links.
    def test_full_road_candidates_preserve_decisions(self):
        import geopandas as gpd
        from shapely.geometry import LineString
        roads = gpd.GeoDataFrame({
            "road_id": ["R1", "R1", "R1", "R2"],
            "road_name": ["다른로", "다른로", "다른로", "목표로"],
            "link_id": ["1", "2", "3", "4"],
            "geometry": [LineString([(127, 37.5), (127.0001, 37.5)]),
                         LineString([(127, 37.50001), (127.0001, 37.50001)]),
                         LineString([(127, 37.50002), (127.0001, 37.50002)]),
                         LineString([(127, 37.5005), (127.0001, 37.5005)])],
        }, crs="EPSG:4326")
        versions = pd.DataFrame([{"coordinate_version_id":"V00001", "spot_id":"A",
            "io_type":"in", "road_name":"목표로", "latitude":37.5, "longitude":127,
            "valid_coordinate":True}])
        baseline = pd.DataFrame([{"coordinate_version_id":"V00001",
            "mapping_status":"manual_review_needed", "selected_road_id":None,
            "max_coordinate_change_m":0, "invalid_rows_in_spot_direction":0}])
        result, report = mapping.evaluate_full_road_candidates(versions, roads, baseline)
        self.assertEqual(result.iloc[0]["best_road_id"], "R2")
        self.assertEqual(result.iloc[0]["mapping_decision"], "unchanged")
        self.assertTrue(all(report["validation_checks"].values()))
        self.assertEqual(report["new_review_proposals_excluding_baseline_auto"], 0)


class OfficialRoadTablesTests(unittest.TestCase):
    # [서연] Exact road-name join preserves coordinate versions, excludes out-of-scope.
    def test_official_mapping_and_coverage(self):
        stable = pd.DataFrame([{"road_id":f"R{i:03d}", "road_name":f"도로{i}"} for i in range(90)])
        volume = pd.DataFrame([
            {"spot_id":"A", "io_type":"in", "latitude":37.5, "longitude":127.0, "road_name":"도로0"},
            {"spot_id":"A", "io_type":"in", "latitude":37.5001, "longitude":127.0, "road_name":"도로0"},
            {"spot_id":"B", "io_type":"out", "latitude":37.6, "longitude":127.1, "road_name":"외부도로"},
        ])
        result, coverage, report = mapping.build_official_road_mapping(volume, stable)
        self.assertEqual(len(result), 3)
        self.assertEqual(report["mapping_status_counts"]["matched"], 2)
        self.assertEqual(report["mapping_status_counts"]["out_of_scope_not_in_90_roads"], 1)
        self.assertEqual(report["roads_with_spots"], 1)
        self.assertEqual(report["distinct_matched_spots"], 1)
        self.assertEqual(coverage.loc[coverage["road_id"].eq("R000"), "spot_count"].iloc[0], 1)
        self.assertTrue(all(report["validation_checks"].values()))

    # [서연] A normalized-name collision must fail closed.
    def test_official_mapping_rejects_name_collision(self):
        stable = pd.DataFrame([{"road_id":f"R{i:03d}", "road_name":f"도로{i}"} for i in range(90)])
        stable.loc[1, "road_name"] = " 도로0 "
        volume = pd.DataFrame([{"spot_id":"A","io_type":"in","latitude":37.5,
                                "longitude":127.0,"road_name":"도로0"}])
        with self.assertRaises(ValueError):
            mapping.build_official_road_mapping(volume, stable)



class OfficialObservationJoinTests(unittest.TestCase):
    # [서연] Two coordinate variants must not multiply the underlying observations.
    def test_row_preserving_join_and_coverage(self):
        roads = pd.DataFrame([{"road_id":f"R{i:03d}", "road_name":f"도로{i}"} for i in range(90)])
        volume = pd.DataFrame([
            {"date":"2026-01-01","hour":"08시","spot_id":"A","io_type":"in",
             "road_name":"도로0","traffic_volume":10,"latitude":37.5},
            {"date":"2026-01-02","hour":"08시","spot_id":"A","io_type":"in",
             "road_name":"도로0","traffic_volume":None,"latitude":37.6},
            {"date":"2026-01-01","hour":"08시","spot_id":"B","io_type":"out",
             "road_name":"범위밖","traffic_volume":20,"latitude":37.7},
        ])
        result, coverage, report = mapping.build_official_observation_tables(volume, roads)
        self.assertEqual(len(result), 3)
        self.assertEqual(report["matched_observation_rows"], 2)
        self.assertEqual(report["duplicate_date_time_road_spot_direction_rows"], 0)
        self.assertEqual(coverage.loc[coverage["road_id"].eq("R000"), "observation_count"].iloc[0], 2)
        self.assertEqual(coverage.loc[coverage["road_id"].eq("R000"), "valid_observation_count"].iloc[0], 1)
        self.assertTrue(all(report["validation_checks"].values()))

    # [서연] Reject ambiguous clock fields rather than inventing a time join key.
    def test_missing_time_column_fails_closed(self):
        roads = pd.DataFrame([{"road_id":f"R{i:03d}", "road_name":f"도로{i}"} for i in range(90)])
        volume = pd.DataFrame([{"date":"2026-01-01","spot_id":"A","io_type":"in",
                                "road_name":"도로0","traffic_volume":10}])
        with self.assertRaisesRegex(ValueError, "time column"):
            mapping.build_official_observation_tables(volume, roads)


if __name__ == "__main__":
    unittest.main()

# [서연] Synthetic classification checks; no real mapping data is opened.
class ManualReasonClassificationTests(unittest.TestCase):
    def test_reason_groups_and_reviewed_separation(self):
        review = pd.DataFrame({
            'coordinate_version_id': ['a', 'b', 'c', 'd'],
            'mapping_status': ['manual_review_needed'] * 3 + ['auto_confirmed_road_only'],
            'review_reasons': ['distance;name', 'name;distance', 'coordinate_change', ''],
        })
        reviewed = pd.DataFrame({
            'coordinate_version_id': ['a', 'b'],
            'manual_decision': ['road_only_confirmed', 'pending_review'],
        })
        result, report = mapping.classify_manual_review_frame(review, reviewed)
        self.assertEqual(len(result), 3)
        self.assertEqual(report['previously_reviewed_versions'], 2)
        self.assertEqual(report['unreviewed_versions'], 1)
        self.assertEqual(report['manual_reason_combinations']['distance;name'], 2)
        self.assertEqual(result.loc[result.coordinate_version_id.eq('c'), 'triage_group'].iloc[0],
                         'unreviewed_needs_rule_assessment')

    def test_reject_conflicting_review_decisions(self):
        review = pd.DataFrame({'coordinate_version_id': ['a'],
                               'mapping_status': ['manual_review_needed'], 'review_reasons': ['name']})
        reviewed = pd.DataFrame({'coordinate_version_id': ['a', 'a'],
                                 'manual_decision': ['pending_review', 'road_only_confirmed']})
        with self.assertRaises(ValueError):
            mapping.classify_manual_review_frame(review, reviewed)


# [서연] Synthetic candidate assessments; no real traffic files are accessed.
class NearbyRoadAssessmentTests(unittest.TestCase):
    def test_assessment_groups_and_preserves_decisions(self):
        review = pd.DataFrame({
            "coordinate_version_id": ["a", "b", "c", "d"],
            "mapping_status": ["manual_review_needed"] * 4,
            "review_reasons": ["multiple_nearby_roads"] * 3 + ["other"],
            "previous_review_decision": ["", "", "pending_review", ""],
            "road_name": ["A", "B", "C", "D"],
        })
        rows = []
        for version, gap in (("a", 25), ("b", 3), ("c", 40)):
            for rank, road_id, match, distance in ((1, "R1", "True", 4), (2, "R1", "True", 6), (3, "R2", "False", 4 + gap)):
                rows.append({"coordinate_version_id": version, "candidate_rank": rank,
                             "road_id": road_id, "link_id": f"{version}{rank}",
                             "distance_m": distance, "road_name_match": match,
                             "direction_comparison": "not_comparable_without_verified_crosswalk"})
        result, report = mapping.assess_nearby_road_frame(review, pd.DataFrame(rows))
        self.assertEqual(report["target_versions"], 3)
        self.assertEqual(dict(zip(result.coordinate_version_id, result.assessment)), {
            "a": "road_only_priority_for_validation", "b": "additional_manual_review",
            "c": "previously_reviewed_excluded"})
        self.assertTrue(all(report["validation_checks"].values()))
        self.assertTrue(result.mapping_decision.eq("unchanged").all())

    def test_reject_duplicate_candidate_ranks(self):
        review = pd.DataFrame({"coordinate_version_id": ["a"], "mapping_status": ["manual_review_needed"],
                               "review_reasons": ["multiple_nearby_roads"],
                               "previous_review_decision": [""], "road_name": ["A"]})
        candidates = pd.DataFrame({"coordinate_version_id": ["a", "a"], "candidate_rank": [1, 1],
                                   "road_id": ["R1", "R2"], "link_id": ["L1", "L2"],
                                   "distance_m": [1, 2], "road_name_match": ["True", "False"],
                                   "direction_comparison": ["unknown", "unknown"]})
        with self.assertRaises(ValueError):
            mapping.assess_nearby_road_frame(review, candidates)


# [서연] Structure cues are conservative flags, not automatic confirmations.
class NearbyRoadStructureReviewTests(unittest.TestCase):
    def test_structure_flags_only_and_unchanged_mapping(self):
        review = pd.DataFrame({
            "coordinate_version_id": ["a", "b", "c"],
            "mapping_status": ["manual_review_needed"] * 3,
            "review_reasons": ["multiple_nearby_roads"] * 3,
            "previous_review_decision": [""] * 3,
            "road_name": ["신월여의지하도로", "퇴계로", "올림픽대로"],
            "spot_name": ["신월여의지하도로", "퇴계로", "올림픽대로"],
            "direction_detail": ["", "", "진입램프"],
        })
        rows = []
        for version in ("a", "b", "c"):
            for rank, road, match, distance in (
                (1, "R1", "True", 2), (2, "R1", "True", 3), (3, "R2", "False", 55)):
                rows.append({"coordinate_version_id": version, "candidate_rank": rank,
                             "road_id": road, "link_id": f"{version}{rank}",
                             "distance_m": distance, "road_name_match": match,
                             "direction_comparison": "not_comparable_without_verified_crosswalk"})
        result, report = mapping.assess_nearby_road_frame(review, pd.DataFrame(rows))
        by_id = result.set_index("coordinate_version_id")
        self.assertIn("underground_or_underpass", by_id.loc["a", "structure_risk_flags"])
        self.assertEqual(by_id.loc["b", "structure_risk_flags"], "")
        self.assertIn("ramp_or_interchange", by_id.loc["c", "structure_risk_flags"])
        self.assertEqual(report["priority_structure_review_counts"]["structure_cue_requires_review"], 2)
        self.assertTrue(result["mapping_decision"].eq("unchanged").all())
        self.assertFalse(result["geometry_overlap_verified"].any())
        self.assertTrue(all(report["validation_checks"].values()))
