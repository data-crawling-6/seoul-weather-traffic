"""[정희] 원본 위치 테스트 — 모든 원본은 저장소 data/raw/ 아래 (io_utils.data_root)

- data_root() 가 환경변수 · D 드라이브와 상관없이 저장소 data/raw/ 인지
- C 결합 코드(src/join/topis_weather_merge.py)가 날씨는 저장소 data/shared/ 의 B 날씨 표를 읽는지
- TOPIS 월 CSV 는 data/raw/topis_speed/ 에서 찾고, 결합에는 TOPIS 원본 50개월 중 44개월만 쓰는지
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from src.io_utils import data_root
from src.join.topis_weather_merge import EXTRA_TOPIS_MONTHS, load_weather, topis_csv, weather_dir

ROOT = Path(__file__).resolve().parents[2]


class DataRootTest(unittest.TestCase):
    def test_원본_위치는_항상_저장소_data_raw(self):
        with mock.patch.dict(os.environ, {"SEOUL_TRAFFIC_DATA_ROOT": r"D:\somewhere"}):
            self.assertEqual(data_root(), ROOT / "data" / "raw")


class WeatherSourceTest(unittest.TestCase):
    def test_날씨는_저장소_표를_읽는다(self):
        self.assertEqual(weather_dir(ROOT), ROOT / "data" / "shared")
        w = load_weather(weather_dir(ROOT))
        self.assertFalse(w.duplicated(["date", "hour"]).any())
        for col in ["event_id", "impact", "precip_mm"]:
            self.assertIn(col, w.columns)


class TopisSourceTest(unittest.TestCase):
    def test_TOPIS_월_CSV_위치(self):
        self.assertEqual(topis_csv(ROOT / "data" / "raw", "2025-12"),
                         ROOT / "data" / "raw" / "topis_speed" / "2025-12.csv")

    def test_결합용_44개월_고르기(self):
        # TOPIS 원본 실제 구성: 겨울 10~3월 × 4번 + 여름 4~9월 × 4번 + 2022-07 · 08 = 50개월
        winters = [f"{y}-{m:02d}" for y in range(2022, 2026) for m in (10, 11, 12)] + \
                  [f"{y + 1}-{m:02d}" for y in range(2022, 2026) for m in (1, 2, 3)]
        summers = [f"{y}-{m:02d}" for y in range(2023, 2027) for m in range(4, 10)]
        release = sorted(set(winters + summers + ["2022-07", "2022-08"]))
        self.assertEqual(len(release), 50)
        used = [m for m in release if m[-2:] != "10" and m not in EXTRA_TOPIS_MONTHS]
        self.assertEqual(len(used), 44)


if __name__ == "__main__":
    unittest.main()
