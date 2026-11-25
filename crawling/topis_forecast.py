"""TOPIS 교통예보 스냅샷 수집 (A)

기존 서비스(TOPIS 교통예보)가 날씨를 반영하는지 보려고, 매일 같은 시각에 예보를 저장한다.
나중에 비 온 날의 '예보 vs 실제 속도'를 비교한다 (노션 주제 3.1 빈틈 검증).

1. Selenium: 교통예보 페이지(/prdc/openTodayFcst.do)를 열고 주간 예보 표가 그려질 때까지 기다린 뒤 캡처
2. 같은 세션으로 페이지가 쓰는 요청을 직접 호출 (네트워크 분석으로 확인)
   - selectPdWeeklyList.do       : 7일치 아침(07~09) · 저녁(18~20) 혼잡지수 · 속도, 도로 종류별
   - selectPdTimesTrfList.do     : 도로 종류별 시간 예보
   - selectRoadNameList.do       : 도로 이름 -> 도로 코드(axisCd)
   - selectPdTimesLinkTrfInfoList.do : 도로 · 방향별 구간(링크) 시간 예보
3. 분석 도로(config/roads.csv) 90개 중 예보에 있는 87개 도로 × 2방향 × 오늘 · 내일 07~20시를 저장
   (서부간선지하도로 · 신월여의지하도로 · 녹사평대로는 예보에 없음, 11/25 확인)

사용법
    python crawling/topis_forecast.py                  # 오늘 · 내일
    python crawling/topis_forecast.py --days 3         # 오늘부터 3일
    python crawling/topis_forecast.py --out data/forecast

결과 (실행한 날짜별 폴더)
    {out}/YYYY-MM-DD/weekly.csv, roadtype_hourly.csv, link_hourly.csv, axis_map.csv
    logs/kmj/topis_forecast.log, logs/kmj/screenshots/forecast_YYYY-MM-DD.png
"""
import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

sys.stdout.reconfigure(encoding="utf-8")

KST = timezone(timedelta(hours=9))
BASE = "https://topis.seoul.go.kr"
PAGE_URL = BASE + "/prdc/openTodayFcst.do"
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "kmj"
LOG_FILE = LOG_DIR / "topis_forecast.log"
SHOT_DIR = LOG_DIR / "screenshots"
WAIT_SEC = 30
PAUSE_SEC = 0.2
FIRST_HOUR, LAST_HOUR = 7, 20  # 출근 ~ 퇴근 (TOPIS 예보는 07시부터)


def now_kst():
    return datetime.now(KST)


def log(msg):
    print(msg, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{now_kst():%Y-%m-%d %H:%M:%S} | {msg}\n")


def open_page(headless=True):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1400,2400")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    driver.get(PAGE_URL)
    # 주간 예보 표는 페이지가 연 뒤 따로 받아 와서 그린다. 7일치 줄(li.cont)이 생길 때까지 기다림
    WebDriverWait(driver, WAIT_SEC).until(
        lambda d: len(d.find_elements(By.CSS_SELECTOR, "#prd_weekly_div1 li.cont")) >= 7)
    return driver


def make_session(driver):
    s = requests.Session()
    for c in driver.get_cookies():
        s.cookies.set(c["name"], c["value"], domain=c.get("domain"))
    s.headers.update({
        "User-Agent": driver.execute_script("return navigator.userAgent"),
        "Referer": PAGE_URL,
        "X-Requested-With": "XMLHttpRequest",
    })
    return s


def post(session, path, data, retries=3):
    for attempt in range(1, retries + 1):
        try:
            r = session.post(BASE + path, data=data, timeout=30)
            r.raise_for_status()
            return r.json().get("rows", [])
        except (requests.RequestException, ValueError) as e:
            if attempt == retries:
                raise
            log(f"  [재시도 {attempt}/{retries}] {path} {data}: {e}")
            time.sleep(3 * attempt)


def axis_map(session, road_names):
    rows = []
    for name in road_names:
        hits = [r for r in post(session, "/prdc/selectRoadNameList.do", {"sRoadName": name}) if r["axisName"] == name]
        rows.append({"road_name": name, "axisCd": hits[0]["axisCd"] if hits else None,
                     "stNodeNm": hits[0]["stNodeNm"] if hits else None, "edNodeNm": hits[0]["edNodeNm"] if hits else None})
        time.sleep(PAUSE_SEC)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2, help="오늘부터 며칠치 구간 예보를 받을지")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "forecast")
    ap.add_argument("--show", action="store_true", help="크롬 창을 띄워서 실행")
    a = ap.parse_args()

    fetched = now_kst()
    today = fetched.date()
    days = [today + timedelta(days=i) for i in range(a.days)]
    out_dir = a.out / f"{today:%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = fetched.isoformat(timespec="seconds")
    log(f"=== 교통예보 수집 시작: {', '.join(map(str, days))} ===")

    driver = open_page(headless=not a.show)
    try:
        SHOT_DIR.mkdir(parents=True, exist_ok=True)
        driver.save_screenshot(str(SHOT_DIR / f"forecast_{today:%Y-%m-%d}.png"))
        session = make_session(driver)

        weekly = pd.DataFrame(post(session, "/prdc/selectPdWeeklyList.do",
                                   {"pdDtF": f"{today:%Y%m%d}", "pdDtT": f"{today + timedelta(days=6):%Y%m%d}"}))
        weekly.assign(fetched_at=stamp).to_csv(out_dir / "weekly.csv", index=False, encoding="utf-8-sig")
        log(f"[주간] {len(weekly)}행")

        roadtype = []
        for d in days:
            roadtype += post(session, "/prdc/selectPdTimesTrfList.do",
                             {"pdDtHhF": f"{d:%Y%m%d}{FIRST_HOUR:02d}", "pdDtHhT": f"{d:%Y%m%d}{LAST_HOUR:02d}"})
        pd.DataFrame(roadtype).assign(fetched_at=stamp).to_csv(out_dir / "roadtype_hourly.csv", index=False, encoding="utf-8-sig")
        log(f"[도로 종류별] {len(roadtype)}행")

        roads = pd.read_csv(ROOT / "config" / "roads.csv", encoding="utf-8-sig")
        amap = axis_map(session, roads["road_name"].tolist()).merge(roads[["road_id", "road_name"]], on="road_name")
        amap.to_csv(out_dir / "axis_map.csv", index=False, encoding="utf-8-sig")
        found = amap.dropna(subset=["axisCd"])
        log(f"[도로 코드] {len(found)}/{len(amap)}개 도로가 예보에 있음")

        links, failed = [], []
        for _, r in found.iterrows():
            for direction in ["1", "2"]:
                for d in days:
                    try:
                        rows = post(session, "/prdc/selectPdTimesLinkTrfInfoList.do", {
                            "pdDtHhF": f"{d:%Y%m%d}{FIRST_HOUR:02d}", "pdDtHhT": f"{d:%Y%m%d}{LAST_HOUR:02d}",
                            "axisCd": r["axisCd"], "axisDirDivCd": direction})
                    except Exception as e:  # 한 도로가 실패해도 나머지는 계속
                        failed.append(f"{r['road_name']}/{direction}/{d}")
                        log(f"  [실패] {r['road_name']} 방향{direction} {d}: {e}")
                        continue
                    for row in rows:
                        row.update(road_id=r["road_id"], road_name=r["road_name"], axisCd=r["axisCd"])
                    links += rows
                    time.sleep(PAUSE_SEC)
        link_df = pd.DataFrame(links).assign(fetched_at=stamp)
        link_df.to_csv(out_dir / "link_hourly.csv", index=False, encoding="utf-8-sig")
        n_links = link_df["linkId"].nunique() if len(link_df) else 0
        log(f"[구간] {len(link_df)}행, 링크 {n_links}개, 실패 {len(failed)}건")
    finally:
        driver.quit()
    log(f"=== 끝: {out_dir.relative_to(ROOT)} ===")


if __name__ == "__main__":
    main()
