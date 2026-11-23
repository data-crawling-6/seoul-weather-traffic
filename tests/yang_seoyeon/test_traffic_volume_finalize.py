"""[서연] 교통량 실행 폴더 상태·경로 확정 전용 테스트."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from crawling.traffic_volume import (
    _find_sheet,
    _month_sheet_pattern,
    _normalize_locations,
    _rewrite_run_path_references,
    summarize_traffic_volume_missing,
)


class TrafficVolumeFinalizeTest(unittest.TestCase):
    def test_missing_summary_groups_rows(self):
        # [서연] 결측 요약은 저장이나 수집 없이 집계만 수행한다.
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"]),
            "split": ["train", "train", "eval"],
            "spot_id": ["A-01", "A-01", "A-02"],
            "spot_name": ["도로1", "도로1", "도로2"],
            "io_type": ["유입", "유입", "유출"],
            "hour": [8, 9, 8],
            "traffic_volume": [10.0, None, None],
            "source_month": ["2025-01", "2025-01", "2025-01"],
            "source_file": ["a.xlsx", "a.xlsx", "a.xlsx"],
        })
        result = summarize_traffic_volume_missing(frame)
        self.assertEqual(result["summary"].loc[0, "missing_rows"], 2)
        self.assertAlmostEqual(result["summary"].loc[0, "missing_pct"], 66.667, places=3)
        self.assertEqual(int(result["by_split"]["missing_rows"].sum()), 2)

    def test_new_location_layout_is_normalized_by_direction(self):
        # [서연] 신형 좌표 시트의 병합 셀과 단일 방향 열을 보정한다.
        source = pd.DataFrame(
            {
                "지점번호": ["A-01", None],
                "방향": ["유입", "유출"],
                "지점명칭": ["시험도로", "시험도로"],
                "위도": [37.1, 37.2],
                "경도": [127.1, 127.2],
                "유출입 방향": ["동쪽→서쪽", "서쪽→동쪽"],
            }
        )
        result = _normalize_locations(source, "test.xlsx")
        self.assertEqual(result["지점번호"].tolist(), ["A-01", "A-01"])
        self.assertEqual(result["방향"].tolist(), ["유입", "유출"])
        self.assertEqual(result["direction_detail"].tolist(), ["동쪽→서쪽", "서쪽→동쪽"])

    def test_month_sheet_pattern_accepts_padded_and_unpadded_month(self):
        # [서연] TOPIS 파일마다 다른 `1월`/`01월` 표기를 모두 인식해야 한다.
        pattern = _month_sheet_pattern("2023-01")
        self.assertEqual(_find_sheet(["2023년 01월"], pattern), "2023년 01월")
        self.assertEqual(_find_sheet(["2023년 1월"], pattern), "2023년 1월")

    def test_summary_references_final_success_paths(self) -> None:
        old_run_dir = Path(r"D:\traffic_volume\runs\yang_seoyeon\test_(중단)")
        final_run_dir = Path(r"D:\traffic_volume\runs\yang_seoyeon\test_(성공)")
        summary = {
            "status": "성공",
            "download": {
                "log_path": str(old_run_dir / "log_(중단).jsonl"),
                "manifest_path": str(old_run_dir / "manifest_(중단).csv"),
                "screenshots": [str(old_run_dir / "screenshots" / "before.png")],
            },
        }

        finalized = _rewrite_run_path_references(
            summary,
            old_run_dir=old_run_dir,
            final_run_dir=final_run_dir,
            status="성공",
        )

        self.assertEqual(finalized["status"], "성공")
        self.assertEqual(
            finalized["download"]["log_path"],
            str(final_run_dir / "log_(성공).jsonl"),
        )
        self.assertEqual(
            finalized["download"]["manifest_path"],
            str(final_run_dir / "manifest_(성공).csv"),
        )
        self.assertEqual(
            finalized["download"]["screenshots"][0],
            str(final_run_dir / "screenshots" / "before.png"),
        )


if __name__ == "__main__":
    unittest.main()
