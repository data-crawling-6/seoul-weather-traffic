"""[정희] 예보 · AWS 서울 표 테스트 — src/weather_seoul.py

- AWS의 -99(결측)와 음수 강수는 빈값으로 바꾼 뒤 평균 · 최댓값을 낸다
- 예보는 4개 권역을 서울 값으로 모으고, 권역별 값도 열로 남긴다
"""
from __future__ import annotations

import unittest

import pandas as pd

from src.weather_seoul import aws_seoul, fcst_seoul


class WeatherSeoulTest(unittest.TestCase):
    def test_AWS_결측과_음수_강수는_빈값(self):
        a = pd.DataFrame({
            "datetime": ["2025-12-04 18:00:00"] * 3, "STN": [1, 2, 3],
            "TA": [-1.0, -99.0, 1.0], "WS": [1.0, 2.0, -99.0],
            "RN_HR1": [2.0, -1.0, 4.0], "RN_DAY": [5.0, 6.0, -99.0], "HM": [80.0, -99.0, 90.0]})
        r = aws_seoul(a).iloc[0]
        self.assertEqual(r["n_stations"], 3)
        self.assertAlmostEqual(r["ta_mean"], 0.0)          # -1, 1 평균 (-99 뺌)
        self.assertAlmostEqual(r["rn_hr1_mean"], 3.0)      # 2, 4 평균 (-1 뺌)
        self.assertEqual(r["rn_hr1_max"], 4.0)
        self.assertEqual(r["hour"], 18)

    def test_예보는_서울_값과_권역별_열(self):
        f = pd.DataFrame({
            "base": ["2025-12-04 14:00:00"] * 4, "fcst": ["2025-12-04 18:00:00"] * 4, "lead_h": [4] * 4,
            "region": ["동남권", "동북권", "서남권", "서북권"],
            "PCP": [0.0, 1.0, 0.0, 3.0], "SNO": [0.0, 2.0, 0.0, 0.0],
            "PTY": [0, 3, 1, 2], "TMP": [1.0, -1.0, 2.0, 0.0]})
        r = fcst_seoul(f).iloc[0]
        self.assertEqual(r["n_regions"], 4)
        self.assertEqual(r["pcp_max"], 3.0)
        self.assertAlmostEqual(r["pcp_mean"], 1.0)
        self.assertEqual(r["pty_snow_regions"], 2)          # 3 눈 · 2 비/눈
        self.assertEqual(r["pty_rain_regions"], 2)          # 1 비 · 2 비/눈
        self.assertEqual(r["pcp_sw"], 0.0)
        self.assertEqual(r["sno_ne"], 2.0)
        self.assertEqual((r["date"], r["hour"]), ("2025-12-04", 18))


if __name__ == "__main__":
    unittest.main()
