"""[민주] 예측 입력 변수 테스트 — src/features.py

- 누수: 예측 시점 t 이후의 통행시간 · 날씨를 바꿔도 입력은 그대로다 (정답 y만 바뀐다)
- 키 중복 없음, 행 수 = 목표 행 × 예측 거리
- 실제 결과 무작위 2,000행: 입력이 t 시점 값과 같다

실제 데이터를 쓰는 테스트는 파일이 없으면 건너뛴다.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.io_utils import SPLIT_DIR
from src.features import HORIZONS, PROC_DIR, TARGET_HOURS, make_rows

KEY = ["date", "target_hour", "horizon", "road_id", "direction"]
INPUTS = ["ratio_t0", "ratio_t1", "ratio_t2", "ratio_t0_missing", "ratio_t1_missing", "ratio_t2_missing",
          "city_ratio_t0", "type_ratio_t0", "temp_c", "temp_drop_3h", "precip_1h", "precip_3h",
          "new_snow_3h_cm", "snow_depth_cm", "precip_type",
          "warn_snow", "warn_snow_upcoming", "warn_snow_hours", "warn_rain", "warn_rain_upcoming",
          "warn_rain_hours", "warn_cold", "warn_cold_upcoming", "warn_cold_hours"]
DATES = pd.to_datetime(["2024-12-03", "2024-12-04"])


def fake_road(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for road, ftype in [("R001", "도시고속도로"), ("R002", "주간선도로"), ("R003", "주간선도로")]:
        for date in DATES:
            for hour in range(24):
                rows.append({"road_id": road, "direction": "상행", "function_type": ftype, "date": date,
                             "hour": hour, "tt": rng.uniform(80, 200), "prior_tt_s": 100.0, "prior_replaced": False})
    rh = pd.DataFrame(rows)
    rh["ratio"] = rh["tt"] / rh["prior_tt_s"]
    return rh


def fake_weather(seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = pd.date_range(DATES[0], DATES[-1] + pd.Timedelta(hours=24), freq="h")
    w = pd.DataFrame({"datetime": dt})
    for c in ["temp_c", "temp_drop_3h", "precip_1h", "precip_3h", "new_snow_3h_cm", "snow_depth_cm",
              "warn_snow_hours", "warn_rain_hours", "warn_cold_hours"]:
        w[c] = rng.normal(size=len(w))
    for c in ["warn_snow", "warn_snow_upcoming", "warn_rain", "warn_rain_upcoming", "warn_cold", "warn_cold_upcoming"]:
        w[c] = rng.integers(0, 5, len(w))
    w["precip_type"] = rng.choice(["none", "rain", "snow"], len(w))
    w["weather_class"] = rng.choice(["맑음", "약한 비·눈", "악천후"], len(w))
    w["weather_kind"] = ""
    return w


class LeakTest(unittest.TestCase):
    def test_t_이후_값을_바꿔도_입력은_그대로(self):
        rh, w = fake_road(), fake_weather()
        base = make_rows(rh, w).set_index(KEY).sort_index()
        cols = w.columns.drop("datetime")
        for date in DATES:
            for target in TARGET_HOURS:
                for h in HORIZONS:
                    t_dt = date + pd.Timedelta(hours=target - h)
                    rh2, w2 = rh.copy(), w.copy()
                    after = rh2["date"] + pd.to_timedelta(rh2["hour"], unit="h") > t_dt
                    rh2.loc[after, "tt"] = 1e6
                    rh2.loc[after, "ratio"] = 1e4
                    future = w2["datetime"] > t_dt
                    w2.loc[future, cols] = w2.loc[future, cols].iloc[::-1].to_numpy()
                    new = make_rows(rh2, w2).set_index(KEY).sort_index()
                    sel = ((base.index.get_level_values("date") == date)
                           & (base.index.get_level_values("target_hour") == target)
                           & (base.index.get_level_values("horizon") == h))
                    pd.testing.assert_frame_equal(base.loc[sel, INPUTS], new.loc[sel, INPUTS], check_like=True)
                    self.assertFalse(np.allclose(base.loc[sel, "y_ratio"], new.loc[sel, "y_ratio"]))

    def test_입력_시각은_목표보다_horizon만큼_앞(self):
        f = make_rows(fake_road(), fake_weather())
        self.assertTrue((f["target_hour"] - f["t_hour"] == f["horizon"]).all())

    def test_비율은_t_t1_t2_시점_값(self):
        rh = fake_road()
        f = make_rows(rh, fake_weather())
        r = rh.set_index(["road_id", "direction", "date", "hour"])["ratio"]
        for lag in [0, 1, 2]:
            idx = pd.MultiIndex.from_arrays([f["road_id"], f["direction"], f["date"], f["t_hour"] - lag])
            np.testing.assert_allclose(f[f"ratio_t{lag}"], r.reindex(idx).to_numpy())


class ShapeTest(unittest.TestCase):
    def test_키_중복_없고_행_수는_목표_행_x_예측_거리(self):
        rh = fake_road()
        f = make_rows(rh, fake_weather())
        self.assertEqual(f.duplicated(KEY).sum(), 0)
        self.assertEqual(len(f), int(rh["hour"].isin(TARGET_HOURS).sum()) * len(HORIZONS))


@unittest.skipUnless((PROC_DIR / "train_dev" / "features_train_dev.parquet").exists(), "결과 파일 없음")
class OutputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = pd.concat([pd.read_parquet(PROC_DIR / SPLIT_DIR[s] / f"features_{SPLIT_DIR[s]}.parquet") for s in ["train", "eval"]],
                          ignore_index=True)

    def test_행_수_와_키_중복(self):
        self.assertEqual(len(self.f), 1_177_200)   # C v04 588,600행 × 예측 거리 2
        self.assertEqual(self.f.duplicated(KEY).sum(), 0)

    def test_무작위_2000행_입력은_t_시점_값(self):
        f = self.f
        h1 = f[f["horizon"] == 1].set_index(["road_id", "direction", "date", "target_hour"])["y_ratio"]
        s = f[(f["horizon"] == 1) & f["t_hour"].isin([8, 18, 19])].sample(2000, random_state=0)
        at_t = h1.reindex(pd.MultiIndex.from_arrays([s["road_id"], s["direction"], s["date"], s["t_hour"]]))
        np.testing.assert_allclose(s["ratio_t0"].to_numpy(), at_t.to_numpy())

        w = pd.concat([pd.read_csv(PROC_DIR / SPLIT_DIR[sp] / f"weather_hourly_{SPLIT_DIR[sp]}.csv", parse_dates=["datetime"])
                       for sp in ["train", "eval"]]).drop_duplicates("datetime").set_index("datetime")
        s = f.sample(2000, random_state=1)
        t_dt = s["date"] + pd.to_timedelta(s["t_hour"], unit="h")
        for col, src in [("temp_c", "temp_c"), ("snow_depth_cm", "snow_depth_cm"), ("precip_1h", "precip_mm"),
                         ("warn_snow", "warn_snow"), ("warn_rain", "warn_rain")]:
            np.testing.assert_allclose(s[col].astype(float).to_numpy(), w[src].reindex(t_dt).astype(float).to_numpy())


if __name__ == "__main__":
    unittest.main()
