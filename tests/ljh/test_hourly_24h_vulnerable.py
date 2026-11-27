"""24시간 도로 표 · 반복 취약 도로 계산 확인 (B, 작은 가짜 자료)

- hourly_24h.build_season: 링크가 하나라도 빠지면 도로 통행시간 빈값, 맑은 시간 중앙값이 평소값, 정체 길이 30% 판정
- vulnerable_roads: 사건 절반 이상 상위 20% + 최소 비율 → is_vulnerable, 적중률 계산
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis import hourly_24h as H  # noqa: E402
from src.analysis import vulnerable_roads as V  # noqa: E402


def fake_links(n_days=8, slow_day=None, missing_day=None):
    rows = []
    for i in range(n_days):
        date = pd.Timestamp("2024-12-02") + pd.Timedelta(i, unit="D")   # 월요일부터
        for link, dist in [("L1", 1000.0), ("L2", 500.0)]:
            tt = 60.0 if link == "L1" else 30.0
            if slow_day is not None and i == slow_day and link == "L1":
                tt = 120.0                                                  # L1 만 2배 → 길이 2/3 정체
            if missing_day is not None and i == missing_day and link == "L2":
                tt = np.nan
            rows.append({"link_id": link, "road_id": "R001", "road_name": "도로", "direction": "상행",
                         "function_type": "도시고속도로", "date": date, "hour": 8, "season": "2024-25_winter",
                         "split": "train", "distance_m": dist, "link_travel_time_seconds": tt})
    d = pd.DataFrame(rows)
    d["daytype"] = d["date"].dt.dayofweek.map(H.DAYTYPE)
    for c in ["link_id", "road_id", "road_name", "direction", "function_type", "season", "split", "daytype"]:
        d[c] = d[c].astype("category")
    return d


def labels(dates, bad=()):
    w = pd.DataFrame({"date": pd.to_datetime(sorted(set(dates))), "hour": 8})
    w["event_id"] = np.where(w["date"].isin(pd.to_datetime(list(bad))), 1.0, np.nan)
    w["impact"] = np.where(w["event_id"].notna(), "강한 비", None)
    return w


class TestHourly24h(unittest.TestCase):
    def test_ratio_and_congestion(self):
        H.MIN_N, old = 3, H.MIN_N
        try:
            d = fake_links(slow_day=7, missing_day=3)
            weekday_dates = d.loc[d["daytype"].eq("평일"), "date"]
            g = H.build_season(d[d["date"].isin(weekday_dates)], labels(weekday_dates, bad=["2024-12-09"]), 1.5, 0.3)
        finally:
            H.MIN_N = old
        g = g.set_index("date")
        self.assertTrue(np.isnan(g.loc["2024-12-05", "tt_s"]))                # L2 가 빠진 날
        self.assertTrue(pd.isna(g.loc["2024-12-05", "is_congested"]))
        self.assertAlmostEqual(g.loc["2024-12-02", "baseline_tt_s"], 90.0)
        self.assertAlmostEqual(g.loc["2024-12-09", "ratio"], 150.0 / 90.0)
        self.assertEqual(int(g.loc["2024-12-09", "is_congested"]), 1)      # 1000m / 1500m ≥ 30%
        self.assertEqual(int(g.loc["2024-12-02", "is_congested"]), 0)


class TestVulnerable(unittest.TestCase):
    def setUp(self):
        rows = []
        for ev in range(4):
            for r in range(10):
                ratio = 1.3 if (r == 0 or (r == 1 and ev < 2)) else 1.0 + r / 100
                rows.append({"model_set": "train", "season": "s", "type": "rain", "event_id": ev, "start": "",
                             "intensity": "강한 비", "road_id": f"R{r:03d}", "road_name": f"도로{r}",
                             "function_type": "주간선도로", "direction": "상행", "ratio": ratio})
        self.e = V.event_ratios(pd.DataFrame(rows))

    def test_rule(self):
        s = V.road_summary(self.e, 1.05).set_index("road_id")
        self.assertTrue(s.loc["R000", "is_vulnerable"])                    # 4번 모두 상위
        self.assertTrue(s.loc["R001", "is_vulnerable"])                    # 4번 중 2번 = 절반
        self.assertTrue(s.loc["R009", "is_vulnerable"])                    # 사건 2·3에서 2위 → 절반
        self.assertFalse(s.loc["R008", "is_vulnerable"])                   # 상위 20%(2개)에 한 번도 못 듦
        s2 = V.road_summary(self.e, 1.5).set_index("road_id")
        self.assertFalse(s2["is_vulnerable"].any())                        # 최소 비율에 걸림

    def test_hit_rate(self):
        self.assertAlmostEqual(V.hit_rate({("R000", "상행")}, self.e), 1.0)
        self.assertAlmostEqual(V.hit_rate({("R005", "상행")}, self.e), 0.0)


if __name__ == "__main__":
    unittest.main()
