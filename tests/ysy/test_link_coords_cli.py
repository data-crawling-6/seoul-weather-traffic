"""[서연] Vertex CLI의 로컬 입력 처리 전용 테스트."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import xml.etree.ElementTree as ET

from crawling.link_coords import (
    _xml_text,
    load_topis_link_ids,
    vertices_to_lines,
)


class LinkCoordsCliTest(unittest.TestCase):
    def test_vertices_to_lines_builds_one_geometry(self) -> None:
        # [서연] 소량 좌표만 사용해 네트워크 없이 선형 복원을 확인한다.
        vertices = pd.DataFrame({
            "link_id": ["L1", "L1"],
            "vertex_order": [1, 2],
            "grs80tm_x": [197521, 197476],
            "grs80tm_y": [452291, 452270],
        })
        result = vertices_to_lines(vertices, source_epsg=5181)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["vertex_count"], 2)
        self.assertTrue(126.0 < result.iloc[0].geometry.coords[0][0] < 128.0)

    def test_xml_text_accepts_lowercase_api_tags(self) -> None:
        # [서연] LinkVerInfo 실제 응답은 소문자 태그를 사용한다.
        row = ET.fromstring("<row><link_id>L1</link_id><ver_seq>2</ver_seq></row>")
        self.assertEqual(_xml_text(row, "LINK_ID"), "L1")
        self.assertEqual(_xml_text(row, "VER_SEQ"), "2")

    def test_load_topis_link_ids_deduplicates_all_months(self) -> None:
        # [서연] 네트워크 없이 월별 링크 합집합과 중복 제거만 확인한다.
        paths = [Path("2025-01.csv"), Path("2025-02.csv")]
        frames = [
            pd.DataFrame({"링크아이디": pd.Series(["L2", "L1"], dtype="string")}),
            pd.DataFrame({"링크아이디": pd.Series(["L2", "L3"], dtype="string")}),
        ]
        with patch.object(Path, "glob", return_value=iter(paths)), patch(
            "crawling.link_coords.pd.read_csv", side_effect=frames
        ):
            self.assertEqual(load_topis_link_ids(Path("unused")), ["L1", "L2", "L3"])


if __name__ == "__main__":
    unittest.main()
