"""[민주] 5단계 SHAP 테스트 — src/analysis/shap_values.py

- 변수별 기여도 합 + 기준값 = 모델 예측
- 요약표 순위는 평균 |SHAP| 큰 순서
- 그림 파일이 저장된다
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import prediction as p
from src.analysis import shap_values as sv
from tests.kmj.test_prediction import fake


class TestShap(unittest.TestCase):
    def test_shap_sums_to_prediction(self):
        from catboost import Pool

        train = fake(200, 3)
        cols = p.STAGE_COLS[sv.FINAL_STAGE]
        model = p.fit_catboost(train, dict(p.CATBOOST_PARAMS, iterations=30), cols)
        x = train.head(20)
        s = sv.shap_matrix(model, x, cols)
        full = model.get_feature_importance(Pool(p.catboost_frame(x, cols), cat_features=p.cat_cols(cols)),
                                            type="ShapValues")
        np.testing.assert_allclose(s.sum(axis=1) + full[:, -1], model.predict(p.catboost_frame(x, cols)), atol=1e-6)

    def test_summary_rank(self):
        s = pd.DataFrame({"a": [0.3, -0.3], "b": [0.1, 0.0]})
        out = sv.summarize({"g": s}).set_index("feature")
        self.assertEqual(out.loc["a", "rank"], 1)
        self.assertAlmostEqual(out.loc["a", "mean_abs"], 0.3)
        self.assertAlmostEqual(out.loc["a", "mean"], 0.0)

    def test_plot_saved(self):
        groups = {f"g{i}": pd.DataFrame({"a": [0.3, -0.3], "b": [0.1, 0.0]}) for i in range(4)}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fig" / "shap.png"
            sv.plot_top(sv.summarize(groups), path)
            self.assertTrue(path.exists() and path.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
