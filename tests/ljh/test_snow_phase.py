"""[정희] snow_phase 열 테스트 — src/clean/weather.py mark_impact

- 눈 사건 시작 ~ 끝 시각은 falling, 그친 뒤 영향 시간(12시간)은 after, 그 뒤는 빈값
- impact 는 그대로 (눈 강도 이름)
- 더 센 비 사건이 같은 시간을 덮으면 snow_phase 는 빈값, 비 영향이 끝나면 다시 after
"""
from __future__ import annotations

import unittest

import pandas as pd

from src.clean import weather as W


def grid(start, n):
    return pd.DataFrame({"datetime": pd.date_range(start, periods=n, freq="h")})


def event(eid, kind, intensity, start, end):
    return {"event_id": eid, "type": kind, "intensity": intensity,
            "start": pd.Timestamp(start), "end": pd.Timestamp(end)}


class SnowPhaseTest(unittest.TestCase):
    def test_falling_after_blank(self):
        df = grid("2024-11-27 00:00", 24)
        ev = pd.DataFrame([event(1, "snow", "많은 눈", "2024-11-27 02:00", "2024-11-27 05:00")])
        out = W.mark_impact(df, ev).set_index("datetime")
        self.assertEqual(out.at[pd.Timestamp("2024-11-27 01:00"), "snow_phase"], "")
        self.assertEqual(out.at[pd.Timestamp("2024-11-27 02:00"), "snow_phase"], "falling")
        self.assertEqual(out.at[pd.Timestamp("2024-11-27 05:00"), "snow_phase"], "falling")
        self.assertEqual(out.at[pd.Timestamp("2024-11-27 06:00"), "snow_phase"], "after")
        self.assertEqual(out.at[pd.Timestamp("2024-11-27 17:00"), "snow_phase"], "after")   # 그친 뒤 12시간
        self.assertEqual(out.at[pd.Timestamp("2024-11-27 18:00"), "snow_phase"], "")
        self.assertEqual(out.at[pd.Timestamp("2024-11-27 06:00"), "impact"], "많은 눈")      # impact 는 그대로

    def test_stronger_rain_overrides(self):
        df = grid("2024-07-01 00:00", 12)
        ev = pd.DataFrame([event(1, "snow", "보통 눈", "2024-07-01 01:00", "2024-07-01 02:00"),
                           event(2, "rain", "호우", "2024-07-01 04:00", "2024-07-01 05:00")])
        out = W.mark_impact(df, ev).set_index("datetime")
        self.assertEqual(out.at[pd.Timestamp("2024-07-01 03:00"), "snow_phase"], "after")
        self.assertEqual(out.at[pd.Timestamp("2024-07-01 04:00"), "impact"], "호우")
        self.assertEqual(out.at[pd.Timestamp("2024-07-01 04:00"), "snow_phase"], "")
        self.assertEqual(out.at[pd.Timestamp("2024-07-01 11:00"), "snow_phase"], "after")  # 비 영향이 끝나면 눈 영향(12시간)이 남음


if __name__ == "__main__":
    unittest.main()
