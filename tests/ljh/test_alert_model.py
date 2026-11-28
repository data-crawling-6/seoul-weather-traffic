"""경보 기준 임계값 규칙 확인 (B)

- 재현율 80% 이상 가운데 정밀도가 가장 높은 임계값을 고른다
- 80%에 못 미치면 재현율이 가장 높은 임계값
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from unittest.mock import patch

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

    def test_winter_threshold_is_applied_only_to_later_summer(self):
        y = np.array([1, 1, 0, 1, 0])
        scores = np.array([.9, .7, .8, .6, .75])
        winter = np.array([True, True, True, False, False])
        threshold = A.threshold_rule(y[winter], scores[winter])[0]
        self.assertEqual(threshold, .7)
        self.assertEqual((scores[~winter] >= threshold).tolist(), [False, True])

    def test_forecast_uses_latest_available_issue_for_target(self):
        d = pd.DataFrame({'date': pd.to_datetime(['2024-01-02'] * 2),
                          'target_hour': [8, 8], 'horizon': [1, 3]})
        fc = pd.DataFrame({'base': pd.to_datetime(['2024-01-02 02:00', '2024-01-02 05:00', '2024-01-02 07:00']),
                           'fcst': pd.to_datetime(['2024-01-02 08:00'] * 3),
                           'lead_h': [6, 3, 1], 'pcp_mean': [1., 2., 99.],
                           'pcp_max': [1., 2., 99.], 'sno_max': [0., 0., 0.],
                           'tmp_min': [0., 0., 0.], 'pty_rain_regions': [0, 0, 0],
                           'pty_snow_regions': [0, 0, 0]})
        with patch.object(A, 'load_processed', return_value=fc):
            actual = A.add_forecast(d)
        self.assertEqual(actual['fc_pcp_max'].tolist(), [2., 1.])

    def test_season_partition_rejects_unexpected_data(self):
        d = pd.DataFrame({'season': ['2024-25_winter', '2025_summer', '2026_summer']})
        with self.assertRaises(ValueError):
            A.dev_season_masks(d)


if __name__ == "__main__":
    unittest.main()
