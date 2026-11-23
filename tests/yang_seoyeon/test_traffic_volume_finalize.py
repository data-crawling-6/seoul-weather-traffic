"""[서연] 교통량 실행 폴더 상태·경로 확정 전용 테스트."""

from __future__ import annotations

import unittest
from pathlib import Path

from crawling.traffic_volume import _rewrite_run_path_references


class TrafficVolumeFinalizeTest(unittest.TestCase):
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
