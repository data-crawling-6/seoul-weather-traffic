"""[민주] 실시간 속도 수집 테스트 — crawling/realtime_speed.py

실제 API는 부르지 않고 가짜 응답으로 확인한다.
"""
from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "crawling"))

import realtime_speed as rs  # noqa: E402

KEY = "SECRETKEY123"


def xml(spd="42.5", trv="87", code="INFO-000") -> bytes:
    return (f"<TrafficInfo><RESULT><CODE>{code}</CODE></RESULT>"
            f"<row><link_id>1</link_id><prcs_spd>{spd}</prcs_spd><prcs_trv_time>{trv}</prcs_trv_time></row>"
            "</TrafficInfo>").encode()


class Resp:
    def __init__(self, content: bytes, status: int = 200):
        self.content, self.status_code = content, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    """fail_once 에 든 링크는 처음 한 번만 연결 오류 (오류 문구에 주소 · 키가 들어가게)"""

    def __init__(self, fail_once=(), fail_always=()):
        self.fail_once, self.fail_always, self.calls = set(fail_once), set(fail_always), []

    def get(self, url, timeout):
        link_id = url.rsplit("/", 1)[-1]
        self.calls.append(link_id)
        if link_id in self.fail_always or link_id in self.fail_once:
            self.fail_once.discard(link_id)
            raise requests.ConnectionError(f"failed {url}")
        return Resp(xml(trv=str(len(link_id))))


def links(n=4, road="R001"):
    return [{"link_id": f"10{i}", "road_id": road, "road_name": "도로", "direction": "상행"} for i in range(n)]


class TestRealtimeSpeed(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(rs.parse(xml()), {"code": "INFO-000", "prcs_spd": "42.5", "prcs_trv_time": "87"})

    def test_hour_end_matches_topis(self):
        # 07:20 호출은 07~08시 칸(TOPIS "~08시")
        self.assertEqual(rs.hour_end(datetime(2026, 12, 1, 7, 20)), 8)
        self.assertEqual(rs.hour_end(datetime(2026, 12, 1, 19, 45)), 20)

    def test_error_does_not_leak_key(self):
        row = rs.fetch(FakeSession(fail_always={"100"}), KEY, links(1)[0])
        self.assertEqual(row["code"], "EXC:ConnectionError")
        self.assertNotIn(KEY, json.dumps(row, ensure_ascii=False))

    def test_retry_failed_links_once(self):
        s = FakeSession(fail_once={"101"}, fail_always={"103"})
        rows = rs.collect(s, KEY, links(4))
        self.assertEqual([r["code"] for r in rows], ["INFO-000", "INFO-000", "INFO-000", "EXC:ConnectionError"])
        self.assertEqual(s.calls.count("101"), 2)
        self.assertEqual(s.calls.count("100"), 1)

    def test_road_sum_blank_when_link_missing(self):
        rows = rs.collect(FakeSession(fail_always={"103"}), KEY, links(4, "R001")) + \
            rs.collect(FakeSession(), KEY, links(2, "R002"))
        roads = {r["road_id"]: r for r in rs.road_summary(rows)}
        self.assertEqual(roads["R001"]["trv_time_s"], "")
        self.assertEqual(roads["R001"]["n_ok"], 3)
        self.assertEqual(roads["R002"]["trv_time_s"], 6.0)  # 링크 id 길이 3 x 2개

    def test_save_never_overwrites(self):
        rows = rs.collect(FakeSession(), KEY, links(2))
        started = datetime(2026, 12, 1, 7, 20, tzinfo=rs.KST)
        with tempfile.TemporaryDirectory() as d:
            p = rs.save(Path(d), started, rows, 1.0)
            summary = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual((summary["links"], summary["links_ok"], summary["hour_end"]), (2, 2, 8))
            with gzip.open(Path(d) / "2026-12-01" / "0720_link.csv.gz", "rt", encoding="utf-8") as f:
                self.assertEqual(f.readline().strip().split(","), rs.LINK_COLS)
            with self.assertRaises(FileExistsError):
                rs.save(Path(d), started, rows, 1.0)

    def test_stable_links_count(self):
        all_links = rs.load_links()
        self.assertEqual(len(all_links), 1753)
        self.assertEqual(len({l["link_id"] for l in all_links}), 1753)


if __name__ == "__main__":
    unittest.main()
