"""경보 기준 임계값 규칙 확인 (B)

- 재현율 80% 이상 가운데 정밀도가 가장 높은 임계값을 고른다
- 80%에 못 미치면 재현율이 가장 높은 임계값
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis import alert_model as A  # noqa: E402


class TestThreshold(unittest.TestCase):
    def test_recall_80_max_precision(self):
        y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        s = np.array([.9, .8, .7, .6, .1, .85, .5, .4, .3, .2])
        thr, p, r = A.threshold_rule(y, s)
        self.assertAlmostEqual(thr, 0.6)          # 위에서 5개 경보: 양성 4개 잡음 (재현율 0.8)
        self.assertAlmostEqual(r, 0.8)
        self.assertAlmostEqual(p, 0.8)

    def test_unreachable_recall(self):
        y = np.array([1, 0, 0, 1])
        s = np.array([.9, .1, .2, .3])
        thr, p, r = A.threshold_rule(y, s)
        self.assertGreaterEqual(r, 0.8)


if __name__ == "__main__":
    unittest.main()
