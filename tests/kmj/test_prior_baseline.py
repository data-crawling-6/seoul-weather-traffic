"""[민주] 입력용 평소값(prior) 규칙 테스트 — src/prior_baseline.py

노션 '분석 설계 3.4' (11/25 A · B 합의) 규칙을 지키는지 본다.
- 기준일 d의 값은 d-28 ~ d-1 맑은 평일 같은 시간 표본으로만 계산 (당일 · 미래 X)
- 표본이 8개 미만이면 지난 시즌 정답용 평소값, 없으면 있는 표본으로
- 도로 통행시간 = 링크 합, 링크가 하나라도 빠지면 빈값 · 표본에서 제외
- 결과 키 중복 없음

실제 데이터를 쓰는 테스트는 파일이 없으면 건너뛴다.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.io_utils import SPLIT_DIR
from src.prior_baseline import (DEFAULT_LINK, MIN_N, PROC_DIR, apply_fallback, load_season,
                                road_hourly, rolling_prior)

START = pd.Timestamp("2023-10-30")  # 평소값 구간 시작
IN_PERIOD = pd.Timestamp("2023-11-27")


def days(n_days: int, tt=None, clear=None) -> pd.DataFrame:
    """링크 L1 · 8시 하루 한 줄. 통행시간 기본값 = 그날 번호(0, 1, 2, ...)라 어느 날이 쓰였는지 알 수 있다"""
    dates = pd.date_range(START, periods=n_days, freq="D")
    d = pd.DataFrame({"link_id": "L1", "hour": 8, "date": dates})
    d["tt"] = np.arange(n_days, dtype=float) if tt is None else tt
    d["ok_day"] = d["date"].dt.weekday < 5
    d["clear"] = True if clear is None else clear
    d["period_scope"] = np.where(d["date"] >= IN_PERIOD, "in_period", "baseline_buffer")
    return d


def prior_of(d: pd.DataFrame) -> pd.DataFrame:
    return rolling_prior(d, ["link_id"], "tt").set_index("date")


class LeakTest(unittest.TestCase):
    def test_당일과_이후_값을_바꿔도_결과가_같다(self):
        d = days(70)
        base = prior_of(d)
        rng = np.random.default_rng(0)
        for target in rng.choice(base.index, 5, replace=False):
            poisoned = d.copy()
            poisoned.loc[poisoned["date"] >= target, "tt"] = 1e6
            self.assertEqual(prior_of(poisoned).loc[target, "window_tt_s"], base.loc[target, "window_tt_s"])

    def test_이전_데이터만_잘라서_다시_계산해도_같다(self):
        d = days(70)
        base = prior_of(d)
        rng = np.random.default_rng(1)
        for target in rng.choice(base.index, 5, replace=False):
            cut = d[d["date"] <= target].copy()
            cut.loc[cut["date"] == target, "tt"] = np.nan   # 당일 행은 기준일로만 남긴다
            r = prior_of(cut).loc[target]
            self.assertEqual(r["window_tt_s"], base.loc[target, "window_tt_s"])
            self.assertEqual(r["prior_n"], base.loc[target, "prior_n"])

    def test_쓰인_날짜는_모두_d_28부터_d_1까지(self):
        d = days(70)
        base = prior_of(d)
        for target, r in base.iterrows():
            i = (target - START).days
            used = [j for j in range(max(0, i - 28), i) if (START + pd.Timedelta(days=j)).weekday() < 5]
            self.assertEqual(r["prior_n"], len(used))
            self.assertEqual(r["window_tt_s"], np.median(used))

    def test_비_오는_날과_주말은_표본이_아니다(self):
        clear = np.ones(70, dtype=bool)
        clear[40:50] = False
        d = days(70, clear=clear)
        r = prior_of(d).loc[START + pd.Timedelta(days=57)]         # 2023-12-26(화)
        used = [j for j in range(29, 57) if (START + pd.Timedelta(days=j)).weekday() < 5 and not 40 <= j < 50]
        self.assertEqual(r["prior_n"], len(used))


class FallbackTest(unittest.TestCase):
    def setUp(self):
        clear = np.zeros(70, dtype=bool)
        clear[21:24] = True                                      # 맑은 평일 3일뿐 (11/20~22 월~수, 8개 미만)
        self.d = days(70, clear=clear)
        self.target = IN_PERIOD + pd.Timedelta(days=1)           # 2023-11-28(화)
        self.prev = pd.DataFrame({"season": ["2022-23_winter"], "link_id": ["L1"],
                                  "weekday": [self.target.weekday()], "hour": [8], "prev_tt_s": [999.0]})

    def run_fallback(self, prev_season):
        r = rolling_prior(self.d, ["link_id"], "tt")
        return apply_fallback(r, self.prev, ["link_id"], prev_season).set_index("date").loc[self.target]

    def test_8개_미만이면_지난_시즌_값(self):
        r = self.run_fallback("2022-23_winter")
        self.assertLess(r["prior_n"], MIN_N)
        self.assertEqual(r["prior_tt_s"], 999.0)
        self.assertEqual(r["prior_source"], "prev_season")
        self.assertTrue(r["prior_replaced"])

    def test_지난_시즌이_없으면_있는_표본으로(self):
        r = self.run_fallback(None)
        self.assertEqual(r["prior_source"], "short_window")
        self.assertEqual(r["prior_n"], 3)
        self.assertEqual(r["prior_tt_s"], 22.0)                  # 21 · 22 · 23번째 날의 중앙값

    def test_지난_시즌에_그_요일_시간이_없어도_있는_표본으로(self):
        self.prev["hour"] = 18
        r = self.run_fallback("2022-23_winter")
        self.assertEqual(r["prior_source"], "short_window")

    def test_8개_이상이면_28일_값(self):
        r = apply_fallback(rolling_prior(days(70), ["link_id"], "tt"), self.prev, ["link_id"], "2022-23_winter")
        r = r.set_index("date").loc[self.target]
        self.assertGreaterEqual(r["prior_n"], MIN_N)
        self.assertEqual(r["prior_source"], "window28")
        self.assertFalse(r["prior_replaced"])


class RoadSumTest(unittest.TestCase):
    def make(self, tt1, tt2):
        d = pd.DataFrame({"road_id": "R001", "direction": "상행", "link_id": ["L1", "L2"],
                          "date": pd.Timestamp("2023-12-01"), "hour": 8, "ok_day": True, "clear": True,
                          "period_scope": "in_period", "link_travel_time_seconds": [tt1, tt2]})
        return road_hourly(d).iloc[0]

    def test_도로값은_링크_합(self):
        self.assertEqual(self.make(100.0, 50.0)["tt"], 150.0)

    def test_링크가_하나라도_빠지면_빈값(self):
        self.assertTrue(pd.isna(self.make(100.0, np.nan)["tt"]))

    def test_빈값_시각은_표본에서_빠진다(self):
        rows = []
        for i, date in enumerate(pd.date_range(START, periods=40, freq="D")):
            tt2 = np.nan if i == 30 else 50.0
            for lid, tt in [("L1", 100.0), ("L2", tt2)]:
                rows.append({"road_id": "R001", "direction": "상행", "link_id": lid, "date": date, "hour": 8,
                             "ok_day": date.weekday() < 5, "clear": True, "link_travel_time_seconds": tt,
                             "period_scope": "in_period" if date >= IN_PERIOD else "baseline_buffer"})
        road = road_hourly(pd.DataFrame(rows))
        r = rolling_prior(road, ["road_id", "direction"], "tt").set_index("date").loc[START + pd.Timedelta(days=35)]
        weekdays = sum((START + pd.Timedelta(days=j)).weekday() < 5 for j in range(7, 35))
        missing_is_weekday = (START + pd.Timedelta(days=30)).weekday() < 5
        self.assertEqual(r["prior_n"], weekdays - int(missing_is_weekday))
        self.assertEqual(r["window_tt_s"], 150.0)


@unittest.skipUnless((PROC_DIR / "train_dev" / "prior_road_train_dev.parquet").exists(), "결과 파일 없음")
class OutputTest(unittest.TestCase):
    def test_도로_결과_키_중복_없음(self):
        for split in ["train", "eval"]:
            r = pd.read_parquet(PROC_DIR / SPLIT_DIR[split] / f"prior_road_{SPLIT_DIR[split]}.parquet")
            self.assertEqual(r.duplicated(["date", "hour", "road_id", "direction"]).sum(), 0)

    def test_링크_결과_키_중복_없음(self):
        for split in ["train", "eval"]:
            path = PROC_DIR / SPLIT_DIR[split] / f"prior_link_{SPLIT_DIR[split]}.parquet"
            if path.exists():
                r = pd.read_parquet(path, columns=["date", "hour", "link_id"])
                self.assertEqual(r.duplicated().sum(), 0)

    def test_대체_표시_열은_window28이_아닌_행과_같다(self):
        for split in ["train", "eval"]:
            r = pd.read_parquet(PROC_DIR / SPLIT_DIR[split] / f"prior_road_{SPLIT_DIR[split]}.parquet", columns=["prior_source", "prior_replaced"])
            self.assertTrue((r["prior_replaced"] == (r["prior_source"].astype(str) != "window28")).all())

    def test_결과는_연구_기간_평일만(self):
        r = pd.read_parquet(PROC_DIR / "train_dev" / "prior_road_train_dev.parquet", columns=["date"])
        self.assertTrue((pd.to_datetime(r["date"]).dt.weekday < 5).all())


@unittest.skipUnless(DEFAULT_LINK.exists() and (PROC_DIR / "train_dev" / "prior_road_train_dev.parquet").exists(),
                     "링크 표 v02 또는 결과 파일 없음")
class RealDataLeakTest(unittest.TestCase):
    def test_실제_도로값을_d_이전_데이터만으로_다시_계산(self):
        season = "2024-25_winter"
        d = load_season(DEFAULT_LINK, season)
        road = road_hourly(d)
        out = pd.read_parquet(PROC_DIR / "train_dev" / "prior_road_train_dev.parquet")
        out = out[(out["season"] == season) & (out["prior_source"] == "window28")].sample(5, random_state=0)
        for _, r in out.iterrows():
            day = pd.Timestamp(r["date"])
            s = road[(road["road_id"] == r["road_id"]) & (road["direction"] == r["direction"]) &
                     (road["hour"] == r["hour"]) & road["ok_day"] & road["clear"] & road["tt"].notna() &
                     (road["date"] < day) & (road["date"] >= day - pd.Timedelta(days=28))]
            self.assertTrue((s["date"] < day).all())
            self.assertEqual(len(s), r["prior_n"])
            self.assertAlmostEqual(s["tt"].median(), float(r["prior_tt_s"]), places=1)


if __name__ == "__main__":
    unittest.main()
