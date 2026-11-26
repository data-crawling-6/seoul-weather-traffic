"""[정희] 공통 설정(config/periods.yaml) 다루기 테스트 — src/clean/weather.py

11/23 팀원 리뷰로 고친 것을 지키는지 본다.
- 기준값은 yaml이 우선 (코드 값은 yaml에 없을 때 기본값)
- research_period(월-일)로 시즌별 periods를 만든다 (겨울은 해를 넘김, baseline_start = 28일 전)
- yaml을 갱신할 때 다른 키(periods · filters 등)를 지우지 않고, 값이 같으면 파일을 안 건드린다
실제 config/periods.yaml은 건드리지 않고 임시 파일로만 확인한다.
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.clean import weather as W

PERIOD = {"겨울": {"option": "평균 ± 1주", "start": "11-27", "end": "03-10"},
          "여름": {"option": "평균", "start": "05-09", "end": "09-23"}}


class WeatherConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "periods.yaml"

    def tearDown(self):
        self.tmp.cleanup()

    def test_yaml_기준값이_코드_값을_이긴다(self):
        self.path.write_text("thresholds:\n  congestion_road_length_ratio: 0.4\n", encoding="utf-8")
        th = {"congestion_road_length_ratio": 0.3, "snow_temp_c": 1.0}
        W.apply_yaml_thresholds(th, self.path)
        self.assertEqual(th["congestion_road_length_ratio"], 0.4)
        self.assertEqual(th["snow_temp_c"], 1.0)          # yaml에 없는 키는 코드 기본값

    def test_yaml이_없으면_코드_기본값(self):
        th = {"congestion_road_length_ratio": 0.3}
        W.apply_yaml_thresholds(th, self.path)            # 파일 없음
        self.assertEqual(th["congestion_road_length_ratio"], 0.3)

    def test_periods_겨울은_해를_넘기고_평소값은_28일_전(self):
        rows = {r["season"]: r for r in W.build_periods(PERIOD)}
        w = rows["2022-23_winter"]
        self.assertEqual((w["start"], w["end"], w["baseline_start"]),
                         (date(2022, 11, 27), date(2023, 3, 10), date(2022, 10, 30)))
        s = rows["2026_summer"]
        self.assertEqual((s["split"], s["start"], s["baseline_start"]), ("eval", date(2026, 5, 9), date(2026, 4, 11)))
        self.assertEqual(len(rows), len(W.TRAIN_SEASONS) + len(W.EVAL_SEASONS))

    def test_2월_29일은_윤년이_아니면_28일(self):
        p = {"겨울": {"start": "12-01", "end": "02-29"}, "여름": PERIOD["여름"]}
        rows = {r["season"]: r for r in W.build_periods(p)}
        self.assertEqual(rows["2022-23_winter"]["end"], date(2023, 2, 28))
        self.assertEqual(rows["2023-24_winter"]["end"], date(2024, 2, 29))

    def test_값이_같으면_파일을_안_건드린다(self):
        text = ("filters:\n  weekdays_only: true\n"
                "thresholds:\n  congestion_road_length_ratio: 0.3   # 고른 이유 주석\n")
        self.path.write_text(text, encoding="utf-8")
        wrote = W.update_periods_yaml({"thresholds": {"congestion_road_length_ratio": 0.3}}, self.path)
        self.assertFalse(wrote)
        self.assertEqual(self.path.read_text(encoding="utf-8"), text)   # 주석까지 그대로

    def test_갱신해도_다른_키와_기준값은_남는다(self):
        self.path.write_text("filters:\n  weekdays_only: true\n"
                             "thresholds:\n  congestion_link_ratio: 1.5\n", encoding="utf-8")
        wrote = W.update_periods_yaml({"thresholds": {"snow_temp_c": 1.0}}, self.path)
        import yaml
        cfg = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.assertTrue(wrote)
        self.assertEqual(cfg["filters"], {"weekdays_only": True})
        self.assertEqual(cfg["thresholds"], {"congestion_link_ratio": 1.5, "snow_temp_c": 1.0})


if __name__ == "__main__":
    unittest.main()
