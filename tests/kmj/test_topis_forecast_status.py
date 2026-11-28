"""TOPIS 예보 수집 완료 판정 (코드 검토 10번, B가 대신 추가)"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "crawling"))
try:
    import topis_forecast as T  # noqa: E402
except ImportError:   # selenium 이 없는 환경
    T = None


@unittest.skipIf(T is None, "selenium 없음")
class StatusTest(unittest.TestCase):
    def test_전부_실패면_완료_아님(self):
        s = T.completion_status(expected=174, n_failed=174, link_rows=0, weekly_rows=7, failed=["a"] * 174)
        self.assertFalse(s["complete"])

    def test_일부_실패_기준(self):
        self.assertTrue(T.completion_status(174, 5, 10000, 7, [])["complete"])     # 97% 성공
        self.assertFalse(T.completion_status(174, 20, 9000, 7, [])["complete"])    # 88% 성공

    def test_완료_표시_파일(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            T.write_status(out, T.completion_status(10, 0, 100, 7, []))
            self.assertTrue((out / "_COMPLETE").exists())
            T.write_status(out, T.completion_status(10, 10, 0, 7, []))
            self.assertFalse((out / "_COMPLETE").exists())               # 다시 받다 실패하면 표시를 지운다
            self.assertTrue((out / "status.json").exists())


if __name__ == "__main__":
    unittest.main()
