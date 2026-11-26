"""[정희] 모델 세트(Train · Dev · Test) 붙이기 테스트 — src/io_utils.py

- 실제 config/periods.yaml 의 model_sets 가 노션 7.2 결정과 같은지
- 시즌 표기 두 가지(W2022 · 2022-23_winter)와 날짜만 있는 표 모두 붙는지
- Test 는 기본값으로 빠지고 include_test=True 일 때만 남는지
- 모르는 시즌 · 겹치는 시즌은 오류로 멈추는지
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.io_utils import add_model_set, load_model_sets, load_processed

DECIDED = {  # 노션 분석 설계 7.2 (11/25)
    "2022-23_winter": "train", "2023-24_winter": "train", "2023_summer": "train", "2024_summer": "train",
    "2024-25_winter": "dev", "2025_summer": "dev",
    "2025-26_winter": "test", "2026_summer": "test",
}


def project(yaml_text: str) -> tempfile.TemporaryDirectory:
    tmp = tempfile.TemporaryDirectory()
    (Path(tmp.name) / "config").mkdir()
    (Path(tmp.name) / "config" / "periods.yaml").write_text(yaml_text, encoding="utf-8")
    return tmp


class ModelSetTest(unittest.TestCase):
    def test_실제_설정이_팀_결정과_같다(self):
        self.assertEqual(load_model_sets(), DECIDED)

    def test_두_가지_시즌_표기_모두_붙는다(self):
        f = pd.DataFrame({"season": ["W2022", "2024-25_winter", "S2025", "2026_summer"]})
        r = add_model_set(f, include_test=True)
        self.assertEqual(list(r["model_set"]), ["train", "dev", "dev", "test"])

    def test_test는_기본값으로_빠진다(self):
        f = pd.DataFrame({"season": ["2023_summer", "2025-26_winter", "2025_summer"], "v": [1, 2, 3]})
        self.assertEqual(list(add_model_set(f)["v"]), [1, 3])
        self.assertEqual(len(add_model_set(f, include_test=True)), 3)

    def test_season_열이_없으면_날짜로_정한다(self):
        f = pd.DataFrame({"date": ["2022-11-26", "2023-03-10", "2025-07-14", "2025-12-04"]})
        r = add_model_set(f, include_test=True)
        self.assertEqual(list(r["model_set"]), ["train", "train", "dev", "test"])

    def test_모르는_시즌이면_오류(self):
        with self.assertRaises(ValueError):
            add_model_set(pd.DataFrame({"season": ["2021-22_winter"]}))

    def test_한_시즌이_두_세트에_있으면_오류(self):
        with project("model_sets:\n  train: [2023_summer]\n  dev: [2023_summer]\n") as root:
            with self.assertRaises(ValueError):
                load_model_sets(Path(root))

    def test_load_processed는_test면_eval_폴더를_안_읽는다(self):
        with project("model_sets:\n  train: [2023_summer]\n  dev: [2025_summer]\n  test: [2026_summer]\n") as root:
            base = Path(root) / "data" / "shared"
            (base / "train_dev").mkdir(parents=True)
            (base / "holdout").mkdir()
            pd.DataFrame({"season": ["2023_summer", "2025_summer"]}).to_csv(base / "train_dev" / "t_train_dev.csv", index=False)
            pd.DataFrame({"season": ["2026_summer"]}).to_csv(base / "holdout" / "t_holdout.csv", index=False)
            self.assertEqual(sorted(load_processed("t", project_root=Path(root))["model_set"]), ["dev", "train"])
            self.assertEqual(sorted(load_processed("t", include_test=True, project_root=Path(root))["model_set"]),
                             ["dev", "test", "train"])


if __name__ == "__main__":
    unittest.main()
