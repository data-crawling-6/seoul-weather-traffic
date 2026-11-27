"""[서연] 노션 확정 분석 규칙의 회귀 테스트."""
from __future__ import annotations

import inspect
import unittest

import numpy as np
import pandas as pd

from src.analysis.traffic_analysis import (
    analyze_hotspots_anomalies,
    analyze_travel_time_curves,
    cluster_roads,
    validate_relationships,
)


class TrafficAnalysisRuleTest(unittest.TestCase):
    def test_kmeans_records_elbow_and_silhouette(self) -> None:
        columns = [
            "median_change_pct", "change_iqr", "decrease_share",
            "morning_median", "evening_median",
        ]
        values = np.r_[np.arange(4), np.arange(10, 14)]
        train = pd.DataFrame({
            "model_set": "train",
            "road_id": [f"R{i:03d}" for i in range(8)],
            "road_name": [f"도로{i}" for i in range(8)],
            "rain_median": 0.0,
            "snow_median": 0.0,
            "valid_road_hours": 10,
            **{column: values for column in columns},
        })
        dev = train.assign(model_set="dev")
        *_, report = cluster_roads(train, dev)
        self.assertEqual(set(report["elbow_inertias"]), set(report["silhouette_scores"]))
        self.assertTrue(all(value >= 0 for value in report["elbow_inertias"].values()))

    def test_spatial_and_anomaly_rules_exclude_test_and_future_baseline(self) -> None:
        source = inspect.getsource(analyze_hotspots_anomalies)
        self.assertIn("include_test=False", source)
        self.assertIn("r=500.0", source)
        self.assertIn("q_bh", source)
        self.assertIn("prior_tt_s", source)
        self.assertIn('clear["event_id"].isna()', source)
        self.assertIn('clear["impact"].isna()', source)
        self.assertNotIn("9999", source)

    def test_waterlevel_accepts_project_summer_season_codes(self) -> None:
        source = inspect.getsource(validate_relationships)
        self.assertIn(r"S\d{4}", source)
        self.assertIn("waterlevel_hourly_train_dev.csv", source)

    def test_travel_curve_clustering_uses_notion_rules(self) -> None:
        source = inspect.getsource(analyze_travel_time_curves)
        self.assertIn("include_test=False", source)
        self.assertIn('k = 3 if 3 in models', source)
        self.assertIn('[\"all\", \"snow\", \"rain\"]', source)
        self.assertIn("baseline_travel_time_s", source)
        self.assertIn("travel_time_ratio", source)
        self.assertIn("road_id", source)
        self.assertIn("direction", source)
        self.assertIn("traffic_change_pct_median", source)
        self.assertIn('candidate["min_size"] >= 5', source)
        self.assertIn("소군집으로 제외", source)
        self.assertIn("dev_agreement_rate", source)
        self.assertIn("불안정_해석보류", source)


if __name__ == "__main__":
    unittest.main()
