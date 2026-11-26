"""[정희] 날씨 판정 붙이기 테스트 — src/io_utils.py attach_weather_labels()

- C 표 안의 event_id · impact 를 저장소 weather_hourly 값으로 바꾸고, 다른 행 수를 세는지
- 값이 같으면 0개로 세고 경고하지 않는지
- 날씨 표에 없는 date · hour 가 있으면 맑음으로 바뀌지 않고 오류로 멈추는지
- 실제 저장소 weather_hourly 가 date · hour 중복 없이 읽히는지
- 실제 C 통행시간 표(v04) · 링크 표(v02)의 날씨 판정이 지금 저장소와 같은지 (파일이 있을 때만)
"""
from __future__ import annotations

import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.io_utils import attach_weather_labels, load_weather_labels

ROOT = Path(__file__).resolve().parents[2]
TRAFFIC = ROOT / "data" / "raw" / "traffic"
ROAD_V04 = TRAFFIC / "yang_seoyeon_topis_weather_commute_20261122_v04.parquet"
LINK_V02 = TRAFFIC / "yang_seoyeon_topis_weather_link_hourly_20261122_v02.parquet"


def labels() -> pd.DataFrame:
    """2시간짜리 날씨 표: 8시 맑음, 9시 강한 비 (사건 3번)."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2023-12-01", "2023-12-01"]),
        "hour": [8, 9],
        "event_id": [np.nan, 3.0],
        "event_intensity": [np.nan, "강한 비"],
        "impact": [np.nan, "강한 비"],
    })


def has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


class AttachTest(unittest.TestCase):
    def test_다른_값은_바꾸고_개수를_센다(self):
        # C 표 복사본: 9시를 맑음으로, 8시를 약한 비로 잘못 들고 있다
        f = pd.DataFrame({"date": ["2023-12-01"] * 3, "hour": [8, 9, 9],
                          "event_id": [7.0, np.nan, np.nan], "impact": [np.nan, np.nan, np.nan]})
        with self.assertWarns(UserWarning):
            r = attach_weather_labels(f, labels=labels())
        self.assertEqual(r.attrs["weather_label_mismatch"], {"event_id": 3, "impact": 2})
        self.assertTrue(np.isnan(r["event_id"].iloc[0]))
        self.assertEqual(list(r["impact"].iloc[1:]), ["강한 비", "강한 비"])
        self.assertEqual(list(r["event_intensity"].iloc[1:]), ["강한 비", "강한 비"])  # 없던 열은 새로 생김

    def test_같은_값이면_0개이고_경고가_없다(self):
        f = pd.DataFrame({"date": pd.to_datetime(["2023-12-01 00:00", "2023-12-01 00:00"]), "hour": [8, 9],
                          "event_id": [np.nan, 3.0], "event_intensity": [None, "강한 비"],
                          "impact": [None, "강한 비"]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = attach_weather_labels(f, labels=labels())
        self.assertEqual(r.attrs["weather_label_mismatch"], {"event_id": 0, "event_intensity": 0, "impact": 0})

    def test_날씨_표에_없는_시간은_오류(self):
        f = pd.DataFrame({"date": ["2023-12-01", "2023-12-02"], "hour": [8, 8]})
        with self.assertRaises(ValueError):
            attach_weather_labels(f, labels=labels())

    def test_실제_날씨_표는_중복_없이_읽힌다(self):
        w = load_weather_labels()
        self.assertFalse(w.duplicated(["date", "hour"]).any())
        self.assertEqual((w["hour"].min(), w["hour"].max()), (0, 23))
        self.assertGreater(w["impact"].notna().sum(), 0)


@unittest.skipUnless(has_pyarrow(), "pyarrow 없음")
class RealTableTest(unittest.TestCase):
    """지금은 C 표의 날씨 판정이 저장소와 같아야 한다. 0이 아니면 판정이 바뀐 것이니 C 표를 다시 만들 때."""

    @unittest.skipUnless(ROAD_V04.is_file(), "통행시간 v04 없음")
    def test_통행시간_v04_날씨_판정이_저장소와_같다(self):
        d = pd.read_parquet(ROAD_V04, columns=["date", "hour", "event_id", "impact"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = attach_weather_labels(d)
        self.assertEqual(r.attrs["weather_label_mismatch"], {"event_id": 0, "impact": 0})

    @unittest.skipUnless(LINK_V02.is_file(), "링크 v02 없음")
    def test_링크_v02_날씨_판정이_저장소와_같다(self):
        import pyarrow.dataset as ds
        d = (ds.dataset(LINK_V02).to_table(columns=["date", "hour", "event_id", "impact"])
             .to_pandas().drop_duplicates(["date", "hour", "event_id", "impact"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = attach_weather_labels(d)
        self.assertEqual(r.attrs["weather_label_mismatch"], {"event_id": 0, "impact": 0})


if __name__ == "__main__":
    unittest.main()
