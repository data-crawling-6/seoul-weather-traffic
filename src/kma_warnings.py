"""
기상특보 통보문 → 서울 특보 상태 (B 담당)

crawling/kma_warning.py 가 받은 목록 · 상세를 읽어서
1. parse_items()     : 통보문 한 건 안의 항목 '(1) 호우주의보 발표 : 시각 / 지역' 을 한 줄씩 푼다
2. seoul_items()     : 해당지역에 서울이 들어간 대설 · 호우 · 한파 항목만 남긴다
                       ('서울' = 4개 권역 전체, '서울(동북권, 서북권)' = 그 권역만)
3. hourly_state()    : 시간마다 서울 특보 상태를 만든다 (모델 입력 ⑤ 특보)
     warn_<kind>          0 없음 · 1 주의보 · 2 경보 (서울 4개 권역 중 가장 높은 단계)
     warn_<kind>_areas    특보가 걸린 서울 권역 수 (0~4)
     warn_<kind>_upcoming 발표는 됐지만 아직 발효 전 (미리 알 수 있는 정보)
     warn_<kind>_hours    이번 특보가 발효된 뒤 지난 시간
4. episodes()        : 서울에 특보가 걸렸다 풀린 한 번 한 번 (생존분석 출발점)

시각 기준: 그 시각까지 '발효' 된 것만 반영한다 (미래 정보 누수 방지).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "kma_warning"
KINDS = {"대설": "snow", "호우": "rain", "한파": "cold"}
LEVEL = {"주의보": 1, "경보": 2}
SEOUL_AREAS = ["동남권", "동북권", "서남권", "서북권"]

TIME_RE = re.compile(r"\((\d+)\)\s*(대설|호우|한파|[가-힣]+?)(주의보|경보)\s*(발표|해제|변경|연장)?\s*:\s*"
                     r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*(\d{1,2})시\s*(\d{1,2})분")
AREA_RE = re.compile(r"\((\d+)\)\s*(대설|호우|한파|[가-힣]+?)(주의보|경보)\s*(발표|해제|변경|연장)?\s*:\s*(.+)")
SEOUL_RE = re.compile(r"서울(?:특별시)?(?:\(([^)]*)\))?")


def _section(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    return text[i + len(start): j if j > 0 else None]


def seoul_areas(area_text: str) -> list[str]:
    """'경기도(고양), 서울(동북권, 서북권)' → ['동북권', '서북권'] / '서울' → 4개 권역 모두"""
    found = set()
    for m in SEOUL_RE.finditer(area_text):
        if m.group(1) is None:
            found.update(SEOUL_AREAS)
        else:
            for a in m.group(1).split(","):
                a = a.strip()
                found.update([x for x in SEOUL_AREAS if x in a] or [])
    return sorted(found)


def parse_items(text: str) -> list[dict]:
    """상세 원문 한 건 → 항목 목록"""
    norm = "\n".join(line.strip() for line in str(text).splitlines() if line.strip())
    t_sec = _section(norm, "발효시각", "해당지역")
    a_sec = _section(norm, "해당지역", "내용")
    times = {m.group(1): m for m in TIME_RE.finditer(t_sec)}
    items = []
    for line in a_sec.splitlines():
        m = AREA_RE.search(line)
        if not m:
            continue
        k = m.group(1)
        tm = times.get(k)
        eff = (pd.Timestamp(int(tm.group(5)), int(tm.group(6)), int(tm.group(7)), int(tm.group(8)), int(tm.group(9)))
               if tm else pd.NaT)
        items.append({"item": int(k), "kind_ko": m.group(2), "level_ko": m.group(3),
                      "action": m.group(4) or "발표", "effective_at": eff, "areas": m.group(5).strip()})
    return items


def load_items(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    lst = pd.read_csv(raw_dir / "warning_list.csv", dtype=str)
    det = pd.read_csv(raw_dir / "warning_detail.csv", dtype=str)
    df = lst.merge(det, on=["inpTma", "no", "stn"], how="inner")
    rows = []
    for r in df.itertuples():
        for it in parse_items(r.text):
            it.update({"issued_at": pd.Timestamp(r.issued_at), "title": r.title, "inpTma": r.inpTma})
            rows.append(it)
    return pd.DataFrame(rows)


def seoul_items(items: pd.DataFrame) -> pd.DataFrame:
    it = items[items["kind_ko"].isin(KINDS)].copy()
    it["seoul_areas"] = it["areas"].map(seoul_areas)
    it = it[it["seoul_areas"].map(len) > 0].copy()
    it["kind"] = it["kind_ko"].map(KINDS)
    it["level"] = it["level_ko"].map(LEVEL)
    it["effective_at"] = it["effective_at"].fillna(it["issued_at"])
    return it.sort_values(["effective_at", "issued_at", "item"]).reset_index(drop=True)


def hourly_state(items: pd.DataFrame, hours: pd.Series) -> pd.DataFrame:
    """hours(정시 목록)마다 서울 특보 상태. 그 시각까지 발효된 항목만 반영한다."""
    hours = pd.to_datetime(pd.Series(hours)).sort_values().reset_index(drop=True)
    out = pd.DataFrame({"datetime": hours})
    for kind in KINDS.values():
        ev = items[items["kind"] == kind]
        state = {a: 0 for a in SEOUL_AREAS}
        lvl, cnt, since = [], [], []
        i, n = 0, len(ev)
        started = None
        recs = ev.to_dict("records")
        for h in hours:
            while i < n and recs[i]["effective_at"] <= h:
                r = recs[i]
                for a in r["seoul_areas"]:
                    if r["action"] == "해제":
                        if state[a] == r["level"]:
                            state[a] = 0
                    else:  # 발표 · 변경 · 연장
                        state[a] = r["level"]
                i += 1
            cur = max(state.values())
            if cur > 0 and started is None:
                started = h
            if cur == 0:
                started = None
            lvl.append(cur)
            cnt.append(sum(v > 0 for v in state.values()))
            since.append((h - started) / pd.Timedelta(hours=1) if started is not None else np.nan)
        # 발표는 됐지만(issued ≤ h) 아직 발효 전(effective > h)인 '발표' 항목이 있으면 예고 상태
        up = np.zeros(len(hours), dtype=int)
        for r in ev[(ev["action"] != "해제") & (ev["effective_at"] > ev["issued_at"])].itertuples():
            up[((hours >= r.issued_at) & (hours < r.effective_at)).to_numpy()] = 1
        out[f"warn_{kind}"] = lvl
        out[f"warn_{kind}_areas"] = cnt
        out[f"warn_{kind}_upcoming"] = up
        out[f"warn_{kind}_hours"] = since
    return out


def episodes(state: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """서울에 특보가 걸렸다 풀린 구간. 생존분석의 출발점(발표 · 발효 시각)으로 쓴다."""
    rows = []
    for kind in KINDS.values():
        col = state[f"warn_{kind}"]
        on = col > 0
        grp = (on != on.shift()).cumsum()
        for _, g in state[on].groupby(grp[on]):
            start, end = g["datetime"].iloc[0], g["datetime"].iloc[-1]
            first = items[(items["kind"] == kind) & (items["action"] != "해제")
                          & (items["effective_at"] <= start)].tail(1)
            rows.append({"kind": kind, "effective_start": start, "end": end + pd.Timedelta(hours=1),
                         "hours": len(g), "max_level": int(g[f"warn_{kind}"].max()),
                         "issued_at": first["issued_at"].iloc[0] if len(first) else pd.NaT})
    return pd.DataFrame(rows).sort_values("effective_start").reset_index(drop=True)


def build(hours: pd.Series, raw_dir: Path = RAW_DIR):
    """원본이 있으면 (시간별 상태, 서울 항목, 구간) 을 돌려준다. 없으면 None."""
    if not (raw_dir / "warning_detail.csv").exists():
        return None
    items = seoul_items(load_items(raw_dir))
    state = hourly_state(items, hours)
    return state, items, episodes(state, items)
