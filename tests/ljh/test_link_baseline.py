"""[정희] 정체 정답(is_congested) 규칙 테스트 — src/labels/link_baseline.py

11/23 팀원 리뷰로 정한 규칙을 지키는지 본다.
- 링크 정체: 통행시간 ÷ 평소값 > 1.5 (1.5 딱 맞으면 정체 아님)
- 정체 길이 비율 = 정체 링크 거리 ÷ 도로 전체 거리
- 도로 정체: 비율 >= 0.3 → 1, 아니면 0
- 링크가 하나라도 빠지면 정답은 빈값
"""
from __future__ import annotations

import unittest

import pandas as pd

from src.labels.link_baseline import add_is_congested, road_congestion_length

KEY = {"season": "2022-23_winter", "split": "train", "date": pd.Timestamp("2022-12-21"), "weekday": 2,
       "hour": 18, "road_id": "R001", "direction": "상행", "function_type": "도시고속도로", "weather_group": "맑음"}


def make(links: list[tuple[str, float, float | None]]) -> pd.DataFrame:
    """links = [(링크 아이디, 거리 m, 그 시간 통행시간 초 · 평소값은 모두 100초)] → 도로 한 칸의 결과"""
    d = pd.DataFrame([{**KEY, "link_id": lid, "distance_m": dist, "link_travel_time_seconds": tt}
                      for lid, dist, tt in links])
    b = d[["season", "link_id", "weekday", "hour"]].assign(baseline_link_tt_s=100.0)
    g = road_congestion_length(d, b, link_ratio=1.5)
    return add_is_congested(g, road_length_ratio=0.3).iloc[0]


class LinkBaselineTest(unittest.TestCase):
    def test_정체_길이_30퍼센트_이상이면_정체(self):
        r = make([("L1", 30, 200), ("L2", 70, 100)])          # 30% 정체
        self.assertAlmostEqual(r["congestion_length_ratio"], 0.3)
        self.assertEqual(r["is_congested"], 1)

    def test_30퍼센트_미만이면_정체_아님(self):
        r = make([("L1", 29, 200), ("L2", 71, 100)])          # 29% 정체
        self.assertEqual(r["is_congested"], 0)

    def test_링크_비율_1점5_딱_맞으면_정체_아님(self):
        r = make([("L1", 50, 150), ("L2", 50, 100)])          # 150 ÷ 100 = 1.5 → 정체 아님
        self.assertEqual(r["n_congested_links"], 0)
        self.assertEqual(r["is_congested"], 0)

    def test_링크가_하나라도_빠지면_정답은_빈값(self):
        r = make([("L1", 50, 200), ("L2", 50, None)])
        self.assertFalse(r["is_complete"])
        self.assertTrue(pd.isna(r["is_congested"]))

    def test_비율_분모는_도로_전체_길이(self):
        # 관측된 50m가 모두 정체여도, 전체 100m 기준 50%로 계산한다 (관측 길이 기준이면 100%)
        r = make([("L1", 50, 200), ("L2", 50, None)])
        self.assertAlmostEqual(r["congestion_length_ratio"], 0.5)
        self.assertAlmostEqual(r["missing_length_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
