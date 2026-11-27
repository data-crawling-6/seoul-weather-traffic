"""서울시 실시간 도로 소통 정보 수집 (A)

TOPIS 12월 통행속도 파일은 다음 해 1/10에 올라와 발표 전에 못 받는다. 그래서 12/11까지 평일 출퇴근 시간에
실시간 속도를 직접 저장해, 비 · 눈 오는 날 TOPIS 교통예보와 실제를 비교한다 (노션 결과 3.8).

- API: 서울 열린데이터광장 TrafficInfo. 링크 1개당 1번, XML, 응답 link_id · prcs_spd · prcs_trv_time
  (데이터셋 페이지 OA-13291은 '종료'로 나오지만 API는 응답함, 11/27 시험 250번 모두 INFO-000)
- 대상: 안정 링크 1,753개 (outputs/tables/stable_links_20261122_v01.csv), 동시 8개 -> 약 1분
- 실패한 링크는 끝에 한 번 더 부른다. 그래도 빠진 링크는 빈값으로 두고 0으로 채우지 않는다
- hour_end: 호출한 시각이 속한 1시간의 끝 시각 (07:20 호출 -> 8, TOPIS "~08시"와 같은 칸)
- 인증키는 환경변수 SEOUL_OPEN_DATA_API_KEY 에서만 읽고 주소 · 로그에 남기지 않는다

사용법
    python crawling/realtime_speed.py --out data/raw/realtime_speed
    python crawling/realtime_speed.py --out probe --limit 20      # 시험: 링크 20개만

결과 (호출 날짜 폴더, 같은 이름이 있으면 덮어쓰지 않고 멈춘다)
    {out}/YYYY-MM-DD/HHMM_link.csv.gz   구간 원본
    {out}/YYYY-MM-DD/HHMM_road.csv      도로 x 방향 통행시간 합 (링크가 하나라도 빠지면 빈값)
    {out}/YYYY-MM-DD/HHMM_summary.json  호출 수 · 응답 코드 · 걸린 시간
"""
import argparse
import csv
import gzip
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]
LINKS = ROOT / "outputs" / "tables" / "stable_links_20261122_v01.csv"
URL = "http://openapi.seoul.go.kr:8088/{key}/xml/TrafficInfo/1/5/{link_id}"
OK = "INFO-000"
WORKERS = 8
LINK_COLS = ["link_id", "road_id", "road_name", "direction", "called_at", "hour_end", "code", "prcs_spd", "prcs_trv_time"]


def load_links(limit: int | None = None) -> list[dict]:
    with LINKS.open(encoding="utf-8-sig") as f:
        rows = [{k: r[k] for k in ("link_id", "road_id", "road_name", "direction")} for r in csv.DictReader(f)]
    return rows[:limit] if limit else rows


def hour_end(t: datetime) -> int:
    return t.hour + 1


def parse(content: bytes) -> dict:
    root = ET.fromstring(content)
    return {
        "code": root.findtext(".//CODE") or "",
        "prcs_spd": root.findtext(".//prcs_spd") or "",
        "prcs_trv_time": root.findtext(".//prcs_trv_time") or "",
    }


def fetch(session, key: str, link: dict) -> dict:
    now = datetime.now(KST)
    row = dict(link, called_at=now.isoformat(timespec="seconds"), hour_end=hour_end(now))
    try:
        r = session.get(URL.format(key=key, link_id=link["link_id"]), timeout=20)
        r.raise_for_status()
        row.update(parse(r.content))
    except Exception as e:  # 오류 문구에 주소(키 포함)가 들어갈 수 있어 종류만 남긴다
        row.update(code=f"EXC:{type(e).__name__}", prcs_spd="", prcs_trv_time="")
    return row


def collect(session, key: str, links: list[dict]) -> list[dict]:
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(lambda l: fetch(session, key, l), links))
    retry = [i for i, r in enumerate(rows) if r["code"] != OK or not r["prcs_trv_time"]]
    if retry:
        time.sleep(3)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            again = list(ex.map(lambda i: fetch(session, key, links[i]), retry))
        for i, r in zip(retry, again):
            if r["code"] == OK and r["prcs_trv_time"]:
                rows[i] = r
    return rows


def road_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["road_id"], r["road_name"], r["direction"]), []).append(r)
    out = []
    for (road_id, road_name, direction), g in sorted(groups.items()):
        ok = [r for r in g if r["code"] == OK and r["prcs_trv_time"] != ""]
        complete = len(ok) == len(g)
        out.append({
            "road_id": road_id,
            "road_name": road_name,
            "direction": direction,
            "n_links": len(g),
            "n_ok": len(ok),
            "trv_time_s": sum(float(r["prcs_trv_time"]) for r in ok) if complete else "",
            "hour_end": g[0]["hour_end"],
        })
    return out


def save(out: Path, started: datetime, rows: list[dict], seconds: float) -> Path:
    day = out / started.strftime("%Y-%m-%d")
    stem = started.strftime("%H%M")
    paths = [day / f"{stem}_link.csv.gz", day / f"{stem}_road.csv", day / f"{stem}_summary.json"]
    if any(p.exists() for p in paths):
        raise FileExistsError(f"{day / stem}_* 가 이미 있습니다. 덮어쓰지 않습니다.")
    day.mkdir(parents=True, exist_ok=True)

    with gzip.open(paths[0], "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LINK_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    roads = road_summary(rows)
    with paths[1].open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(roads[0]))
        w.writeheader()
        w.writerows(roads)

    codes = Counter(r["code"] for r in rows)
    paths[2].write_text(json.dumps({
        "started_at": started.isoformat(timespec="seconds"),
        "hour_end": hour_end(started),
        "links": len(rows),
        "links_ok": sum(1 for r in rows if r["code"] == OK and r["prcs_trv_time"]),
        "roads_complete": sum(1 for r in roads if r["trv_time_s"] != ""),
        "roads": len(roads),
        "codes": dict(codes),
        "seconds": round(seconds, 1),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return paths[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "raw" / "realtime_speed"))
    ap.add_argument("--limit", type=int, help="시험용: 앞에서부터 링크 몇 개만")
    args = ap.parse_args()

    key = os.environ.get("SEOUL_OPEN_DATA_API_KEY", "").strip()
    if not key:
        sys.exit("환경변수 SEOUL_OPEN_DATA_API_KEY 가 없습니다.")

    links = load_links(args.limit)
    started = datetime.now(KST)
    t0 = time.perf_counter()
    rows = collect(requests.Session(), key, links)
    summary_path = save(Path(args.out), started, rows, time.perf_counter() - t0)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False))
    if summary["links_ok"] == 0:
        sys.exit("값이 하나도 오지 않았습니다. 응답 코드를 확인하세요.")


if __name__ == "__main__":
    main()
