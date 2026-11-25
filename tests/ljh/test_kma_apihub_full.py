"""kma_apihub.py --part full 의 날짜 목록 확인 (B)

- 연구 기간(periods.yaml periods) 안 평일만 들어간다 (주말 없음)
- 90일 목록에 있는 날(예: 2025-12-04)은 항상 들어가고 split 이 맞다
- 연구 기간 앞 4주(buffer)는 넣지 않는다
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "crawling"))
import kma_apihub as k  # noqa: E402


class TestFullDays(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.days = k.load_full_days()
        cls.by_day = dict(cls.days)

    def test_weekdays_only(self):
        self.assertTrue(all(d.weekday() < 5 for d, _ in self.days))

    def test_case_day_included(self):
        self.assertEqual(self.by_day.get(pd.Timestamp("2025-12-04")), "eval")

    def test_inside_research_period(self):
        self.assertNotIn(pd.Timestamp("2025-11-03"), self.by_day)   # 겨울 앞 4주(buffer)
        self.assertIn(pd.Timestamp("2025-11-27"), self.by_day)      # 겨울 첫날(목)

    def test_more_than_90_days(self):
        self.assertGreater(len(self.days), 600)


if __name__ == "__main__":
    unittest.main()
