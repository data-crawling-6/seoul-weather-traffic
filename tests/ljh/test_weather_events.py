"""[정희] 눈 · 비 사건 판정 테스트 — src/weather.py extract_events

11/20 12/4 폭설이 21시에야 잡혔던 문제(눈 올 때 시간 강수량이 빈칸)가 다시 생기지 않는지 본다.
- 강수량이 0이어도 현상번호 05(눈)나 적설 증가가 있으면 눈으로 잡는다
- 3시간 넘게 끊기면 다른 사건
"""
from __future__ import annotations

import unittest

import pandas as pd

from src import weather as W


def hours(rows):
    """rows = [(시각, 강수 mm, 신적설 3h cm, 적설 cm, 기온, 현상번호)]"""
    return pd.DataFrame(rows, columns=["datetime", "precip_mm", "new_snow_3h_cm", "snow_depth_cm", "temp_c", "phenomenon"]
                        ).assign(datetime=lambda d: pd.to_datetime(d["datetime"]))


class WeatherEventsTest(unittest.TestCase):
    def test_강수량이_빈칸이어도_현상번호_05면_눈(self):
        df = hours([("2025-12-04 17:00", 0.0, 0.0, 0.0, -1.0, None),
                    ("2025-12-04 18:00", 0.0, 0.0, 0.0, -1.0, "05"),
                    ("2025-12-04 19:00", 0.0, 0.0, 0.0, -1.0, None)])
        out, ev = W.extract_events(df)
        self.assertEqual(list(out["precip_type"]), ["", "snow", ""])
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev.iloc[0]["type"], "snow")

    def test_적설이_늘면_눈(self):
        df = hours([("2025-12-04 17:00", 0.0, 0.0, 0.0, -1.0, None),
                    ("2025-12-04 18:00", 0.0, 0.0, 2.0, -1.0, None)])
        out, _ = W.extract_events(df)
        self.assertEqual(out.loc[1, "precip_type"], "snow")

    def test_3시간_넘게_끊기면_다른_사건(self):
        df = hours([("2023-07-01 08:00", 6.0, 0.0, 0.0, 25.0, "01"),
                    ("2023-07-01 09:00", 0.0, 0.0, 0.0, 25.0, None),
                    ("2023-07-01 13:00", 1.0, 0.0, 0.0, 25.0, "01")])
        _, ev = W.extract_events(df)
        self.assertEqual(len(ev), 2)
        self.assertEqual(ev.iloc[0]["intensity"], "강한 비")   # 시간당 5mm 이상
        self.assertEqual(ev.iloc[1]["intensity"], "약한 비")


if __name__ == "__main__":
    unittest.main()
