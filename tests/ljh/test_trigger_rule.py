"""악천후 모드 발동 규칙 확인 (B)

- 예보 강수 >= p 또는 예보 눈이면 켠다, with_obs 면 지금 관측 강수도 보탠다
- 재현율 · 정밀도 계산
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis import trigger_rule as T  # noqa: E402


class TriggerTest(unittest.TestCase):
    def setUp(self):
        self.g = pd.DataFrame({"pcp_max": [0.0, 1.0, 0.0, 0.0, np.nan], "sno_max": [0.0, 0.0, 0.3, 0.0, 0.0],
                               "pty_snow_regions": [0, 0, 0, 0, 0], "obs_precip_t": [0.0, 0.0, 0.0, 2.0, 0.0],
                               "truth": [False, True, True, True, False], "truth_kind": ["", "비", "눈", "비", ""],
                               "season": ["s"] * 5})

    def test_rule(self):
        self.assertEqual(T.rule_on(self.g, 1.0).tolist(), [False, True, True, False, False])
        self.assertEqual(T.rule_on(self.g, 1.0, with_obs=True).tolist(), [False, True, True, True, False])
        self.assertEqual(T.rule_on(self.g, 2.0).tolist(), [False, False, True, False, False])

    def test_score(self):
        s = T.score(self.g, T.rule_on(self.g, 1.0, with_obs=True))
        self.assertAlmostEqual(s["recall"], 1.0)
        self.assertAlmostEqual(s["precision"], 1.0)
        s2 = T.score(self.g, T.rule_on(self.g, 1.0))
        self.assertAlmostEqual(s2["recall"], 2 / 3)
        self.assertAlmostEqual(s2["recall_비"], 0.5)


if __name__ == "__main__":
    unittest.main()
