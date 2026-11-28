"""[민주] 예측 0~5단계 테스트 — src/analysis/prediction.py

- 입력 열에 정답 · 평가 구분 열이 없고, 3단계까지는 날씨 열도 없다
- 4단계는 3단계 입력 + 관측 날씨 · 수위 · 특보뿐이고 예보 · AWS는 없다
- 5단계 예보는 발표 시각이 t보다 앞선 것만 쓰고, t 이후 발표 · 관측을 바꿔도 입력이 그대로다
- 악천후 가중치는 '내리는 중인 악천후'(목표 시각 강수 > 0) 행만 w배, 나머지는 1. 3 · 4단계가 같은 w 목록을 쓴다
- 가중치 고르기: 악천후 MAE 최소, 전체 MAE가 1배보다 5% 넘게 나빠지면 제외
- 큰 지연 MAE · 20% 이상 느려짐 적중률 · 정밀도 계산
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
        "precip_target": rng.choice([0.0, 0.5, 3.0], n),
        "snow_phase_target": rng.choice([None, "falling", "after"], n),
    })
    df["weather_kind"] = np.where(df["weather_class"] == "악천후", rng.choice(["눈", "비"], n), "")
    df.loc[:4, "ratio_t0"] = np.nan
    for c in p.RATIO_COLS:
        df[f"{c}_missing"] = df[c].isna()
    for c in p.WEATHER + p.WATER + p.WARN + p.FCST + p.AWS:
        df[c] = rng.normal(size=n)
    df["precip_type"] = rng.choice(["none", "rain", "snow"], n)
    for c in p.WATER:
        df[c] = np.nan
    for c in p.WATER_FLAGS:
        df[c] = df[c].astype(object)
    return df


class TestPrediction(unittest.TestCase):
    def test_inputs_have_no_target_or_weather(self):
        self.assertFalse(set(p.FEATURES) & p.FORBIDDEN)
        weather_like = ("temp", "precip", "snow", "wl", "warn", "fcst", "aws")
        self.assertEqual([c for c in p.FEATURES if c.startswith(weather_like)], [])

    def test_stage4_adds_only_observed_weather(self):
        self.assertEqual(p.FEATURES_4[:len(p.FEATURES)], p.FEATURES)
        added = p.FEATURES_4[len(p.FEATURES):]
        self.assertTrue(all(c.startswith(("temp", "precip", "new_snow", "snow", "wl", "warn")) for c in added))
        self.assertFalse([c for c in p.FEATURES_4 if c.startswith(("fcst", "aws"))])
        self.assertFalse(set(p.FEATURES_4) & p.FORBIDDEN)
        self.assertEqual(len(set(p.FEATURES_4)), len(p.FEATURES_4))

    def test_bad_weather_weight(self):
        x = fake()
        w = p.bad_weights(x, 3)
        bad = x["weather_class"] == "악천후"
        snow, rain = bad & (x["weather_kind"] == "눈"), bad & (x["weather_kind"] == "비")
        expect = (snow & (x["snow_phase_target"] == "falling")) | (rain & (x["precip_target"] > 0))
        np.testing.assert_array_equal(w, np.where(expect, 3.0, 1.0))
        self.assertTrue((w[snow & (x["snow_phase_target"] == "after")] == 1.0).all())   # 그친 뒤는 강조하지 않음
        self.assertTrue((w[rain & (x["precip_target"] == 0)] == 1.0).all())
        names = p.all_stages([1, 2])
        self.assertEqual(names[:6], p.STAGES)
        self.assertEqual(names[6:], ["3_CatBoost_가중2", "4_CatBoost_날씨_가중2", "5_CatBoost_예보_가중2"])
        self.assertFalse({"weather_class", "precip_target", "snow_phase_target"} & set(p.FEATURES_5))

    def test_stage5_adds_only_forecast_and_aws(self):
        self.assertEqual(p.FEATURES_5[:len(p.FEATURES_4)], p.FEATURES_4)
        added = p.FEATURES_5[len(p.FEATURES_4):]
        self.assertTrue(added and all(c.startswith(("fcst_", "aws_")) for c in added))
        self.assertFalse(set(p.FEATURES_5) & p.FORBIDDEN)
        self.assertEqual(len(set(p.FEATURES_5)), len(p.FEATURES_5))

    def forecast_case(self):
        t_dt = pd.Series(pd.to_datetime(["2024-12-02 17:00", "2024-12-02 19:00", "2024-12-02 05:00"]))
        target = pd.Series(pd.to_datetime(["2024-12-02 20:00", "2024-12-02 20:00", "2024-12-02 08:00"]))
        def row(base, fcst, v):
            return {"base": base, "fcst": fcst, **{c: v for c in p.FCST_SRC}}
        fcst = pd.DataFrame([row("2024-12-02 14:00", "2024-12-02 20:00", 14.0),
                             row("2024-12-02 17:00", "2024-12-02 20:00", 17.0),
                             row("2024-12-02 02:00", "2024-12-02 08:00", 2.0),
                             row("2024-12-02 05:00", "2024-12-02 08:00", 5.0)])
        aws = pd.DataFrame({"datetime": pd.to_datetime(["2024-12-02 05:00", "2024-12-02 17:00", "2024-12-02 19:00",
                                                        "2024-12-02 20:00"]), **{c: [5.0, 17.0, 19.0, 20.0] for c in p.AWS_SRC}})
        f = pd.DataFrame({"k": [0, 1, 2]})
        return f, t_dt, target, fcst, aws

    def test_forecast_uses_only_issued_before_t(self):
        out = p.add_forecast_aws(*self.forecast_case())
        # t=17시 -> 17시 발표는 아직 못 씀(14시 것), t=19시 -> 17시 것, t=05시 -> 02시 것
        self.assertEqual(out["fcst_pcp_mean"].tolist(), [14.0, 17.0, 2.0])
        self.assertEqual(out["aws_ta_mean"].tolist(), [17.0, 19.0, 5.0])

    def test_forecast_ignores_values_after_t(self):
        f, t_dt, target, fcst, aws = self.forecast_case()
        base = p.add_forecast_aws(f, t_dt, target, fcst, aws)
        fcst2 = fcst.copy()
        late = pd.to_datetime(fcst2["base"]) >= pd.Timestamp("2024-12-02 17:00")
        fcst2.loc[late, p.FCST_SRC] = 999.0
        aws2 = aws.copy()
        aws2.loc[aws2["datetime"] == pd.Timestamp("2024-12-02 20:00"), p.AWS_SRC] = 999.0
        new = p.add_forecast_aws(f, t_dt, target, fcst2, aws2)
        pd.testing.assert_frame_equal(base.iloc[[0, 2]], new.iloc[[0, 2]])

    def test_vulnerable_groups_by_road_and_direction(self):
        import tempfile
        from pathlib import Path
        v = pd.DataFrame({"type_name": ["비", "비", "눈"], "road_id": ["R001", "R002", "R001"],
                          "direction": ["상행", "상행", "상행"], "is_vulnerable": [True, False, True]})
        x = pd.DataFrame({"road_id": ["R001", "R001", "R002", "R003"], "direction": ["상행", "하행", "상행", "상행"]})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "v.csv"
            v.to_csv(path, index=False)
            self.assertEqual(list(p.vulnerable_of(x, path)), ["눈·비 취약", "나머지", "나머지", "나머지"])

    def test_pick_weight_rule(self):
        def row(stage, gtype, group, mae):
            return {"stage": stage, "horizon": "전체", "group_type": gtype, "group": group, "mae_min": mae}
        m = pd.DataFrame([
            row("3_CatBoost", "전체", "전체", 1.00), row("3_CatBoost", "날씨", "악천후", 2.0),
            row("3_CatBoost_가중2", "전체", "전체", 1.04), row("3_CatBoost_가중2", "날씨", "악천후", 1.8),
            row("3_CatBoost_가중3", "전체", "전체", 1.06), row("3_CatBoost_가중3", "날씨", "악천후", 1.5),
        ])
        self.assertEqual(p.pick_weight(m, "3_CatBoost", [1, 2, 3]), 2)   # 3배는 전체가 6% 나빠져 제외

    def test_slow_metrics(self):
        g = pd.DataFrame({"date": ["d"] * 4, "target_hour": [8] * 4, "y_ratio": [1.0, 1.25, 1.4, 1.5],
                          "pred_x": [1.3, 1.1, 1.3, 1.5], "prior_target_s": [600.0] * 4})
        m = p.metric_row(g, "pred_x")
        self.assertEqual(m["n_big"], 2)
        self.assertAlmostEqual(m["mae_min_big"], 0.5)   # 0.1 x 600초 = 1분, 0분
        self.assertAlmostEqual(m["slow_recall"], 2 / 3)
        self.assertAlmostEqual(m["slow_precision"], 2 / 3)

    def test_save_keeps_old_runs(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            paths = dict(metrics_path=d / "m.csv", pred_path=d / "p.parquet", log_path=d / "l.json")
            for run_at, value in [("2026-11-27T21:00:00+09:00", 1.0), ("2026-11-27T22:30:00+09:00", 2.0)]:
                p.save_outputs(pd.DataFrame({"mae_min": [value]}),
                               {"log": {"run_at": run_at}, "pred": pd.DataFrame({"x": [value]})}, **paths)
            self.assertEqual(pd.read_csv(d / "m.csv")["mae_min"].iloc[0], 2.0)     # 최신은 덮어씀
            old = d / "prediction_runs" / "m_20261127_2100.csv"
            self.assertEqual(pd.read_csv(old)["mae_min"].iloc[0], 1.0)            # 예전 실행은 남음
            self.assertTrue((d / "runs" / "p_20261127_2230.parquet").exists())
            self.assertTrue((d / "runs" / "l_20261127_2100.json").exists())

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
        cb4 = p.fit_catboost(train, dict(p.CATBOOST_PARAMS, iterations=20), p.FEATURES_4, p.bad_weights(train, 2))
        dev["pred_4_CatBoost_날씨"] = cb4.predict(p.catboost_frame(dev, p.FEATURES_4))
        cb5 = p.fit_catboost(train, dict(p.CATBOOST_PARAMS, iterations=20), p.FEATURES_5)
        dev["pred_5_CatBoost_예보"] = cb5.predict(p.catboost_frame(dev, p.FEATURES_5))
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
