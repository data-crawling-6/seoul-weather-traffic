"""[정희] 교통량 지점 방향(상행 · 하행) 붙이기 테스트 — src/volume_direction.py

억지로 붙이지 않는다는 원칙을 지키는지 본다.
- 출발 → 도착이 한 방향 링크로만 이어지면 그 방향
- 두 방향 모두 이어지면 ambiguous, 못 찾으면 not_found (둘 다 방향 빈값)
- 같은 지점의 유입 · 유출이 같은 방향이면 conflict (둘 다 빈값)
"""
from __future__ import annotations

import unittest

import pandas as pd

from src.volume_direction import build_map, find_direction, parse_detail


def links(rows):
    return pd.DataFrame(rows, columns=["도로명", "링크아이디", "시점명", "종점명", "방향"])


# 상행 A → B → C, 하행 C → B → A
ROAD = links([("성산로", "1", "A", "B", "상행"), ("성산로", "2", "B", "C", "상행"),
              ("성산로", "3", "C", "B", "하행"), ("성산로", "4", "B", "A", "하행")])


class VolumeDirectionTest(unittest.TestCase):
    def test_지명_뽑기(self):
        self.assertEqual(parse_detail("[성산로]봉원고가차도->독립문역"), ("봉원고가차도", "독립문역"))
        self.assertEqual(parse_detail("[성산로] 봉원 고가차도 -> 독립문역"), ("봉원고가차도", "독립문역"))
        self.assertEqual(parse_detail("형식이 다름"), ("", ""))

    def test_한_방향으로만_이어지면_그_방향(self):
        self.assertEqual(find_direction(ROAD, "A", "C")[:2], ("상행", "path_exact"))
        self.assertEqual(find_direction(ROAD, "C", "A")[:2], ("하행", "path_exact"))

    def test_두_방향_모두_이어지면_빈값(self):
        both = pd.concat([ROAD, links([("성산로", "5", "A", "C", "하행")])])
        d, method, valid = find_direction(both, "A", "C")
        self.assertIsNone(d)
        self.assertEqual(method, "ambiguous")
        self.assertEqual(sorted(valid), ["상행", "하행"])

    def test_못_찾으면_빈값(self):
        self.assertEqual(find_direction(ROAD, "X", "Y")[:2], (None, "not_found"))

    def test_이름_일부만_같아도_찾는다(self):
        road = links([("올림픽대로", "1", "행주대교남단", "가양대교남단", "하행"),
                      ("올림픽대로", "2", "가양대교남단", "방화대교남단", "하행")])
        self.assertEqual(find_direction(road, "행주대교", "방화대교남단")[:2], ("하행", "path_partial"))

    def test_유입_유출이_같은_방향이면_둘_다_빈값(self):
        v = pd.DataFrame([
            {"spot_id": "S1", "spot_name": "s", "road_id": "R1", "road_name": "성산로", "normalized_road_name": "성산로",
             "io_type": "유입", "direction_detail": "[성산로]A->C"},
            {"spot_id": "S1", "spot_name": "s", "road_id": "R1", "road_name": "성산로", "normalized_road_name": "성산로",
             "io_type": "유출", "direction_detail": "[성산로]A->B"},   # 둘 다 상행으로 이어짐
        ])
        m = build_map(v, ROAD)
        self.assertTrue(m["direction"].isna().all())
        self.assertEqual(set(m["match_method"]), {"conflict"})


if __name__ == "__main__":
    unittest.main()
