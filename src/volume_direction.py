"""
교통량 지점에 TOPIS 방향(상행 · 하행)을 붙이는 연결표 (B 담당)

C 교통량 릴리즈(yang-seoyeon-traffic-volume-v1)는 도로(road_id)까지만 연결돼 있고,
B 정체 표(road_congestion_length)는 도로 × 방향(상행 · 하행) 단위라 방향을 따로 붙인다.
C 릴리즈 파일은 고치지 않고, 연결표와 도로 × 방향 교통량 표를 B 쪽에 새로 만든다.

방법
- 교통량 direction_detail "[성산로]봉원고가차도->독립문역" 에서 출발 · 도착 지명을 뽑는다.
- TOPIS 원본의 같은 도로 링크를 방향별로 '시점명 → 종점명' 길로 잇는다.
- 출발 지명에서 도착 지명까지 링크를 따라갈 수 있는 방향이 하나뿐이면 그 방향으로 붙인다.
  1) path_exact   : 지명이 링크 시점 · 종점 이름과 그대로 같다 (공백만 무시)
  2) path_partial : 1)로 못 찾으면, 한쪽 이름이 다른 쪽 이름에 들어 있는 경우까지 본다 (2글자 이상)
- 두 방향 모두 되거나(ambiguous) 둘 다 안 되면(not_found) 방향을 비운다. 억지로 붙이지 않는다.
- 같은 지점의 유입 · 유출이 같은 방향으로 붙으면(conflict) 둘 다 비운다.

입력
  data/raw/c_traffic/yang_seoyeon_traffic_weather_days_joined_20261124T021832_._v01.csv  (C 릴리즈)
  data/raw/topis/topis_speed_2025-12.zip   (A 릴리즈 topis-data-v1, 링크 이름 · 방향만 쓴다)

결과
  data/processed/traffic_volume_direction_map.csv
      지점 × 유입 · 유출마다 road_id · direction · 붙인 방법(match_method)
  data/processed/{train,eval}/traffic_volume_road_dir_{split}.csv
      날짜 × 시간 × 도로 × 방향 교통량 (정체 표와 date + hour + road_id + direction 으로 붙인다)
      volume_mean          그 도로 · 방향에 붙은 지점들의 교통량 평균 (대/시간)
      n_spots              붙은 지점 수
      n_spots_observed     교통량이 있는 지점 수 (0이면 volume_mean 빈값)
  logs/volume_direction.json   붙은 지점 수 · 방법별 개수

실행
  python src/volume_direction.py
  python src/volume_direction.py --volume 경로.csv --topis 경로.zip
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
DEFAULT_VOLUME = ROOT / "data" / "raw" / "c_traffic" / "yang_seoyeon_traffic_weather_days_joined_20261124T021832_._v01.csv"
DEFAULT_TOPIS = ROOT / "data" / "raw" / "topis" / "topis_speed_2025-12.zip"
PROC_DIR = ROOT / "data" / "processed"
MIN_PARTIAL = 2   # 부분 일치로 볼 최소 글자 수


def norm(s) -> str:
    return re.sub(r"\s+", "", str(s)) if pd.notna(s) else ""


def parse_detail(s) -> tuple[str, str]:
    """'[도로명]출발->도착' 에서 출발 · 도착 지명."""
    m = re.match(r"\s*\[.*?\]\s*(.*?)\s*->\s*(.*)", str(s))
    return (norm(m.group(1)), norm(m.group(2))) if m else ("", "")


def load_links(path: Path) -> pd.DataFrame:
    cols = ["도로명", "링크아이디", "시점명", "종점명", "방향"]
    t = pd.read_csv(path, encoding="utf-8-sig", usecols=cols, dtype=str, low_memory=False)
    t = t.drop_duplicates("링크아이디")
    for c in ["도로명", "시점명", "종점명"]:
        t[c] = t[c].map(norm)
    return t


def _reachable(edges: dict[str, set[str]], starts: set[str], goals: set[str]) -> bool:
    seen, q = set(starts), deque(starts)
    while q:
        n = q.popleft()
        if n in goals:
            return True
        for m in edges.get(n, ()):
            if m not in seen:
                seen.add(m)
                q.append(m)
    return False


def _candidates(name: str, nodes: set[str], partial: bool) -> set[str]:
    if not name:
        return set()
    if not partial:
        return {name} & nodes
    if len(name) < MIN_PARTIAL:
        return set()
    return {n for n in nodes if len(n) >= MIN_PARTIAL and (name in n or n in name)}


def find_direction(road_links: pd.DataFrame, frm: str, to: str) -> tuple[str | None, str, list[str]]:
    """(방향, 방법, 길이 이어진 방향 목록)"""
    graphs = {}
    for d, g in road_links.groupby("방향"):
        edges = defaultdict(set)
        for a, b in zip(g["시점명"], g["종점명"]):
            edges[a].add(b)
        nodes = set(g["시점명"]) | set(g["종점명"])
        graphs[d] = (edges, nodes)
    for partial, method in ((False, "path_exact"), (True, "path_partial")):
        valid = []
        for d, (edges, nodes) in graphs.items():
            s, e = _candidates(frm, nodes, partial), _candidates(to, nodes, partial)
            if s and e and not (s & e) and _reachable(edges, s, e):
                valid.append(d)
        if len(valid) == 1:
            return valid[0], method, valid
        if len(valid) > 1:
            return None, "ambiguous", valid
    return None, "not_found", []


def build_map(v: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    spots = v.drop_duplicates(["spot_id", "io_type"])[
        ["spot_id", "spot_name", "road_id", "road_name", "normalized_road_name", "io_type", "direction_detail"]].copy()
    rows = []
    for r in spots.itertuples(index=False):
        frm, to = parse_detail(r.direction_detail)
        road_links = links[links["도로명"] == norm(r.normalized_road_name)]
        d, method, valid = find_direction(road_links, frm, to)
        rows.append({**r._asdict(), "from_name": frm, "to_name": to, "direction": d,
                     "match_method": method, "valid_directions": ",".join(valid),
                     "n_topis_links": len(road_links)})
    m = pd.DataFrame(rows)
    # 같은 지점의 유입 · 유출이 같은 방향이면 둘 다 비운다
    same = (m.dropna(subset=["direction"]).groupby("spot_id")["direction"]
            .agg(lambda s: len(s) == 2 and s.nunique() == 1))
    bad = same[same].index
    m.loc[m["spot_id"].isin(bad), ["direction", "match_method"]] = [None, "conflict"]
    return m.sort_values(["spot_id", "io_type"]).reset_index(drop=True)


def road_dir_volume(v: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    x = v.merge(m[["spot_id", "io_type", "direction"]], on=["spot_id", "io_type"], how="left")
    x = x[x["direction"].notna()]
    g = (x.groupby(["date", "hour", "split", "role", "season", "road_id", "road_name", "direction"])
         .agg(volume_mean=("traffic_volume", "mean"), n_spots=("spot_id", "nunique"),
              n_spots_observed=("traffic_volume", "count"))
         .reset_index())
    g["volume_mean"] = g["volume_mean"].round(1)
    return g.sort_values(["date", "hour", "road_id", "direction"]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description="교통량 지점에 TOPIS 방향 붙이기 (B)")
    p.add_argument("--volume", type=Path, default=DEFAULT_VOLUME, help="C 교통량 릴리즈 CSV")
    p.add_argument("--topis", type=Path, default=DEFAULT_TOPIS, help="TOPIS 월 원본 (zip 또는 csv)")
    a = p.parse_args()

    v = pd.read_csv(a.volume, encoding="utf-8-sig", low_memory=False)
    links = load_links(a.topis)
    m = build_map(v, links)
    m.to_csv(PROC_DIR / "traffic_volume_direction_map.csv", index=False, encoding="utf-8-sig")

    g = road_dir_volume(v, m)
    for split in ("train", "eval"):
        out = PROC_DIR / split
        out.mkdir(parents=True, exist_ok=True)
        g[g["split"] == split].to_csv(out / f"traffic_volume_road_dir_{split}.csv", index=False, encoding="utf-8-sig")

    matched = m["direction"].notna()
    spot_ok = m.groupby("spot_id")["direction"].apply(lambda s: s.notna().all())
    summary = {
        "run_at": datetime.now(KST).isoformat(timespec="seconds"),
        "volume_input": a.volume.name,
        "topis_input": a.topis.name,
        "spot_io_rows": len(m),
        "spot_io_matched": int(matched.sum()),
        "spots_both_matched": int(spot_ok.sum()),
        "spots_total": int(m["spot_id"].nunique()),
        "roads_with_direction": int(m.loc[matched, "road_id"].nunique()),
        "match_method_counts": m["match_method"].value_counts().to_dict(),
        "road_dir_rows": {s: int((g["split"] == s).sum()) for s in ("train", "eval")},
    }
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "volume_direction.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
