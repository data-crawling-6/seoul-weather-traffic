"""[정희] 특보 통보문의 서울 권역 해석 테스트 — src/clean/warnings_seoul.py

11/20 '서울(서울동남권 제외)'를 '동남권만'으로 거꾸로 읽어 한파특보가 25일 이어졌던 버그가
다시 생기지 않는지 본다.
"""
from __future__ import annotations

import unittest

from src.clean import warnings_seoul as WR
from src.clean.warnings_seoul import seoul_areas


class KmaWarningsTest(unittest.TestCase):
    def test_제외는_그_권역만_뺀다(self):
        self.assertEqual(seoul_areas("서울(서울동남권 제외)"), ["동북권", "서남권", "서북권"])

    def test_서울만_있으면_4개_권역_모두(self):
        self.assertEqual(seoul_areas("서울"), ["동남권", "동북권", "서남권", "서북권"])

    def test_괄호_안_권역만(self):
        self.assertEqual(seoul_areas("경기도(고양), 서울(동북권, 서북권)"), ["동북권", "서북권"])

    def test_서울이_없으면_빈_목록(self):
        self.assertEqual(seoul_areas("경기도(고양, 파주)"), [])


if __name__ == "__main__":
    unittest.main()


class AppliedAtTest(unittest.TestCase):
    """코드 검토 8번: 발효가 발표보다 앞서면 발표 시각부터 반영한다."""

    def test_발효_07시_발표_09시는_09시부터(self):
        import pandas as pd
        items = pd.DataFrame([{"kind": "rain", "level": 1, "action": "발표", "seoul_areas": ["동남권"],
                               "effective_at": pd.Timestamp("2024-07-01 07:00"), "issued_at": pd.Timestamp("2024-07-01 09:00")}])
        hours = pd.Series(pd.date_range("2024-07-01 07:00", periods=3, freq="h"))
        st = WR.hourly_state(items, hours)
        self.assertEqual(st["warn_rain"].tolist(), [0, 0, 1])
