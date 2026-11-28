"""[민주] 최종 평가 테스트 — src/analysis/final_eval.py

- 연습 모드는 Test 행이 있으면 멈추고, Train으로 학습하고 Dev로 평가한다
- Test 열기 기록이 있으면 두 번째 실행은 멈춘다
- 사건 부트스트랩: 예측이 같으면 구간이 0, 오차가 절반이면 -50% 근처
- 판정: 악천후 10% 이상 감소 + 맑음 5% 이내일 때만 "날씨 효과 있음"
- 실제 데이터: 연습 모드로 읽으면 Test 시즌이 없다
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.analysis import final_eval as fe
from src.analysis import prediction as p
from tests.kmj.test_prediction import fake


def with_sets(n=120, seed=0) -> pd.DataFrame:
    f = fake(n, seed)
    f["model_set"] = np.resize(["train", "train", "dev"], n)
    f["season"] = "2024-25_winter"
    f["event_target"] = np.where(f["weather_class"] == p.BAD_CLASS, "E" + (np.arange(n) % 6).astype(str), None)
    return f


class TestFinalEval(unittest.TestCase):
    def test_practice_refuses_test_rows(self):
        f = with_sets()
        fit, ev = fe.split(f, open_test=False)
        self.assertEqual(set(fit["model_set"]), {"train"})
        self.assertEqual(set(ev["model_set"]), {"dev"})
        f.loc[0, "model_set"] = "test"
        with self.assertRaises(RuntimeError):
            fe.split(f, open_test=False)
        fit, ev = fe.split(f, open_test=True)
        self.assertEqual(set(fit["model_set"]), {"train", "dev"})
        self.assertEqual(set(ev["model_set"]), {"test"})

    def test_lock_only_once(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(fe, "LOCK_PATH", Path(d) / "test_opened.json"), \
                    mock.patch.object(fe, "ROOT", Path(d)):
                fe.write_lock({"opened_at": "x"})
                with self.assertRaises(RuntimeError):
                    fe.write_lock({"opened_at": "y"})

    def test_event_bootstrap(self):
        bad = with_sets(200, 1)
        bad = bad.loc[bad["weather_class"] == p.BAD_CLASS].copy()
        bad["base"] = bad[p.TARGET] + 0.2
        bad["same"] = bad["base"]
        bad["half"] = bad[p.TARGET] + 0.1
        self.assertEqual(fe.event_bootstrap(bad, "base", "same", n=200), (0.0, 0.0))
        lo, hi = fe.event_bootstrap(bad, "base", "half", n=200)
        self.assertAlmostEqual(lo, -0.5, places=6)
        self.assertAlmostEqual(hi, -0.5, places=6)

    def test_judge_rule(self):
        ev = with_sets(400, 2)
        ev["horizon"] = 3
        ev["pred_3_CatBoost"] = ev[p.TARGET] + 0.2
        bad = ev["weather_class"] == p.BAD_CLASS
        ev["pred_4_CatBoost_날씨"] = np.where(bad, ev[p.TARGET] + 0.17, ev[p.TARGET] + 0.2)   # 악천후 -15%
        ev["pred_5_CatBoost_예보"] = np.where(bad, ev[p.TARGET] + 0.19, ev[p.TARGET] + 0.2)   # 악천후 -5%
        j = fe.judge_table(ev).set_index("stage")
        self.assertAlmostEqual(j.loc["4_CatBoost_날씨", "bad_change"], -0.15)
        self.assertTrue(j.loc["4_CatBoost_날씨", "weather_effect"])
        self.assertFalse(j.loc["5_CatBoost_예보", "weather_effect"])
        ev["pred_4_CatBoost_날씨"] = np.where(bad, ev[p.TARGET] + 0.17, ev[p.TARGET] + 0.22)  # 맑음 +10%
        self.assertFalse(fe.judge_table(ev).set_index("stage").loc["4_CatBoost_날씨", "weather_effect"])

    def test_open_test_not_with_iterations(self):
        with mock.patch("sys.argv", ["final_eval.py", "--open-test", "--iterations", "10"]):
            with self.assertRaises(SystemExit):
                fe.main()

    def test_real_practice_load_has_no_test(self):
        path = p.ROOT / "data" / "shared" / "train_dev" / "features_train_dev.parquet"
        if not path.exists():
            self.skipTest("입력 변수 표가 없음")
        f = fe.load_all(include_test=False)
        self.assertNotIn("test", set(f["model_set"]))
        self.assertEqual(int(f[p.FCST + p.AWS].isna().sum().sum()), 0)


if __name__ == "__main__":
    unittest.main()
