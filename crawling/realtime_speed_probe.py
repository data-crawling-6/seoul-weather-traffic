"""서울시 실시간 도로 소통 API(TrafficInfo) 시험 호출 (A)

본 수집을 만들기 전에 확인할 것
1. 해외(GitHub 서버)에서 응답이 오는지, 데이터셋이 '종료'로 나와도 값이 오는지
2. 한 번 부르는 데 걸리는 시간 -> 1,753개 링크를 한 시각에 몇 분 안에 받을 수 있는지
3. 여러 개를 동시에 불러도 막히지 않는지

안정 링크 1,753개에서 고정 순서로 뽑아 1개씩 50번, 동시 8개로 200번 부른다 (총 250번).
인증키는 환경변수 SEOUL_OPEN_DATA_API_KEY 에서만 읽고 주소 · 로그에 남기지 않는다.

사용법
    python crawling/realtime_speed_probe.py --out probe
결과
    {out}/probe_calls.csv (호출마다 결과), 요약은 화면에 출력
"""
import argparse
import csv
import os
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]
LINKS = ROOT / "outputs" / "tables" / "stable_links_20261122_v01.csv"
BASE = "http://openapi.seoul.go.kr:8088/{key}/xml/TrafficInfo/1/5/{link_id}"


def call(session: requests.Session, key: str, link_id: str) -> dict:
    row = {"link_id": link_id, "called_at": datetime.now(KST).isoformat(timespec="seconds")}
    start = time.perf_counter()
    try:
        r = session.get(BASE.format(key=key, link_id=link_id), timeout=20)
        row["http"] = r.status_code
        root = ET.fromstring(r.content)
        row["code"] = root.findtext(".//CODE") or ""
        row["prcs_spd"] = root.findtext(".//prcs_spd") or ""
        row["prcs_trv_time"] = root.findtext(".//prcs_trv_time") or ""
    except Exception as e:  # 오류 문구에 주소(키 포함)가 들어갈 수 있어 종류만 남긴다
        row["http"] = ""
        row["code"] = f"EXC:{type(e).__name__}"
    row["sec"] = round(time.perf_counter() - start, 3)
    return row


def summary(name: str, rows: list[dict], wall: float, workers: int) -> None:
    ok = [r for r in rows if r.get("prcs_spd")]
    secs = sorted(r["sec"] for r in rows)
    codes: dict[str, int] = {}
    for r in rows:
        codes[r["code"] or "(없음)"] = codes.get(r["code"] or "(없음)", 0) + 1
    per_link = wall / len(rows)
    print(f"[{name}] 호출 {len(rows)} · 값 있음 {len(ok)} · 코드 {codes}")
    print(f"  한 번 걸린 시간: 중앙값 {statistics.median(secs):.2f}초 · 95% {secs[int(len(secs) * 0.95) - 1]:.2f}초")
    print(f"  전체 {wall:.1f}초 (동시 {workers}개) -> 1,753개 예상 {1753 * per_link / 60:.1f}분")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="probe")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    key = os.environ.get("SEOUL_OPEN_DATA_API_KEY", "").strip()
    if not key:
        sys.exit("환경변수 SEOUL_OPEN_DATA_API_KEY 가 없습니다.")

    with LINKS.open(encoding="utf-8-sig") as f:
        links = [r["link_id"] for r in csv.DictReader(f)]
    picks = links[::7][:250]  # 도로마다 고르게 퍼지도록 7개마다 하나

    session = requests.Session()
    rows: list[dict] = []

    t0 = time.perf_counter()
    seq = [call(session, key, l) for l in picks[:50]]
    summary("1개씩", seq, time.perf_counter() - t0, 1)
    rows += [dict(r, mode="seq") for r in seq]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        par = list(ex.map(lambda l: call(session, key, l), picks[50:250]))
    summary(f"동시 {args.workers}개", par, time.perf_counter() - t0, args.workers)
    rows += [dict(r, mode=f"par{args.workers}") for r in par]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["mode", "link_id", "called_at", "http", "code", "prcs_spd", "prcs_trv_time", "sec"]
    with (out / "probe_calls.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"저장: {out / 'probe_calls.csv'}")


if __name__ == "__main__":
    main()
