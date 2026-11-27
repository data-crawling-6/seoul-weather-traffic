"""[민주] 예측 0~3단계 테스트 — src/analysis/prediction.py

- 입력 열에 정답 · 평가 구분 · 날씨(4단계 이후) 열이 없다
- 1단계 빈값은 1, 선형회귀 빈 비율은 1.0 + 빈값 표시 그대로
- 분 단위 오차 = 비율 오차 x 입력용 평소 통행시간 / 60
- 실제 데이터: Test 시즌이 섞이지 않고, Train · Dev 시즌이 periods.yaml 약속과 같다
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.analysis import prediction as p
from src.io_utils import SPLIT_DIR, load_model_sets


def fake(n=60, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = rng.choice([8, 9, 18, 19, 20], n)
    df = pd.DataFrame({
        "date": pd.to_datetime("2024-12-02") + pd.to_timedelta(rng.integers(0, 5, n), unit="D"),
        "target_hour": hours, "horizon": rng.choice([1, 3], n),
        "road_id": rng.choice(["R001", "R002"], n), "direction": rng.choice(["상행", "하행"], n),
        "function_type": "도시고속도로", "n_links": 10, "total_distance_m": 12000.0, "lanes": 3.0,
        "urban_share": 0.0, "is_riverside": False, "weekday": rng.integers(0, 5, n), "month": 12,
        "season_kind": "winter", "is_vacation_peak": False,
        "ratio_t0": rng.normal(1, 0.1, n), "ratio_t1": rng.normal(1, 0.1, n), "ratio_t2": rng.normal(1, 0.1, n),
        "city_ratio_t0": 1.0, "type_ratio_t0": 1.0,
        "y_ratio": rng.normal(1.1, 0.2, n), "prior_target_s": 600.0, "prior_replaced_target": False,
        "weather_class": rng.choice(["맑음", "악천후"], n), "weather_kind": "",
    })
    df.loc[:4, "ratio_t0"] = np.nan
    for c in p.RATIO_COLS:
        df[f"{c}_missing"] = df[c].isna()
    return df


class TestPrediction(unittest.TestCase):
    def test_inputs_have_no_target_or_weather(self):
        self.assertFalse(set(p.FEATURES) & p.FORBIDDEN)
        weather_like = ("temp", "precip", "snow", "wl", "warn", "fcst", "aws")
        self.assertEqual([c for c in p.FEATURES if c.startswith(weather_like)], [])

    def test_stage1_missing_is_one(self):
        x = fake()
        pred = p.predict_stage1(x)
        self.assertTrue(np.all(pred[:5] == 1.0))
        np.testing.assert_allclose(pred[5:], x["ratio_t0"].to_numpy()[5:])

    def test_linear_fill_keeps_missing_flag(self):
        lf = p.linear_frame(fake())
        self.assertFalse(lf[p.RATIO_COLS].isna().any().any())
        self.assertTrue((lf.loc[:4, "ratio_t0"] == 1.0).all())
        self.assertTrue((lf.loc[:4, "ratio_t0_missing"] == 1).all())

    def test_minutes_metric(self):
        g = pd.DataFrame({"date": ["d"] * 2, "target_hour": [8, 9], "y_ratio": [1.0, 1.2],
                          "pred_x": [1.1, 1.2], "prior_target_s": [600.0, 600.0]})
        m = p.metric_row(g, "pred_x")
        self.assertAlmostEqual(m["mae_min"], 0.5)   # 0.1 x 600초 = 1분, 0분 -> 평균 0.5분
        self.assertAlmostEqual(m["mae_ratio"], 0.05)
        self.assertEqual(m["n_hours"], 2)

    def test_models_fit_and_evaluate(self):
        train, dev = fake(200, 1), fake(60, 2)
        dev["pred_0_평소값"] = p.predict_stage0(dev)
        dev["pred_1_직전상태"] = p.predict_stage1(dev)
        dev["pred_2_선형회귀"] = p.fit_linear(train).predict(p.linear_frame(dev))
        cb = p.fit_catboost(train, dict(p.CATBOOST_PARAMS, iterations=20))
        dev["pred_3_CatBoost"] = cb.predict(p.catboost_frame(dev))
        m = p.evaluate(dev, p.STAGES)
        self.assertEqual(set(m["stage"]), set(p.STAGES))
        self.assertFalse(m[["mae_min", "mae_ratio"]].isna().any().any())
        self.assertNotIn("없음", set(m.loc[m["group_type"] == "눈비", "group"]))

    def test_real_data_has_no_test_season(self):
        path = p.ROOT / "data" / "shared" / SPLIT_DIR["train"] / "features_train_dev.parquet"
        if not path.exists():
            self.skipTest("입력 변수 표가 없음")
        f, _ = p.load()
        sets = load_model_sets()
        self.assertNotIn("test", set(f["model_set"]))
        for ms in ["train", "dev"]:
            seasons = set(f.loc[f["model_set"] == ms, "season"].astype(str))
            self.assertEqual(seasons, {s for s, v in sets.items() if v == ms})


if __name__ == "__main__":
    unittest.main()
