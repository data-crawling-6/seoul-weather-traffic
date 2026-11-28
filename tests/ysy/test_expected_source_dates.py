"""월 파일에서 날짜가 통째로 빠졌는지 찾는 기대 날짜 목록 (코드 검토 11번, B가 대신 추가)"""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.join.topis_weather_merge import expected_source_dates  # noqa: E402


class ExpectedDatesTest(unittest.TestCase):
    def test_기간_안_날짜만(self):
        periods = [{"baseline_start": pd.Timestamp("2024-10-30"), "start": pd.Timestamp("2024-11-27"),
                    "end": pd.Timestamp("2025-03-10")}]
        oct_days = expected_source_dates("2024-10", periods)
        self.assertEqual([d.day for d in oct_days], [29, 30, 31])          # 평소값 시작 하루 전부터
        self.assertEqual(len(expected_source_dates("2024-11", periods)), 30)
        self.assertEqual(len(expected_source_dates("2025-03", periods)), 10)
        self.assertEqual(len(expected_source_dates("2025-06", periods)), 0)

    def test_빠진_날짜_찾기(self):
        periods = [{"baseline_start": pd.Timestamp("2024-04-11"), "start": pd.Timestamp("2024-05-09"),
                    "end": pd.Timestamp("2024-09-23")}]
        present = pd.DatetimeIndex(pd.date_range("2024-06-01", "2024-06-30").drop(pd.Timestamp("2024-06-15")))
        missing = expected_source_dates("2024-06", periods).difference(present)
        self.assertEqual(list(missing.strftime("%Y-%m-%d")), ["2024-06-15"])


if __name__ == "__main__":
    unittest.main()
