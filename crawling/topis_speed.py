"""TOPIS 도로별 일자별 통행속도 수집 (A)

1. Selenium: 연도 선택 -> 월별 목록 갱신 대기 -> 파일 정보 추출
2. 다운로드: 월 버튼이 보내는 POST /downloadFileRefRoom.do 를 requests로 직접 호출
   (재시도, 이어받기, 크기/형식 검사)
3. 검증: 엑셀을 CSV로 옮기면서 날짜 범위, 빠진 날, 결측률, 속도 0 개수 기록

사용법
    python crawling/topis_speed.py --season winter-2024
    python crawling/topis_speed.py --months 2022-08
    python crawling/topis_speed.py --all
    python crawling/topis_speed.py --season winter-2024 --recheck   # 다시 받지 않고 검증만
"""
import argparse
import calendar
import csv
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests
from openpyxl import load_workbook
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

sys.stdout.reconfigure(encoding="utf-8")

BASE = "https://topis.seoul.go.kr"
PAGE_URL = BASE + "/refRoom/openRefRoom_1.do"
DOWNLOAD_URL = BASE + "/downloadFileRefRoom.do"

ROOT = Path(__file__).resolve().parents[1]
XLSX_DIR = ROOT / "data" / "raw" / "topis" / "xlsx"
CSV_DIR = ROOT / "data" / "raw" / "topis" / "csv"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "topis_speed.log"
NET_LOG = LOG_DIR / "topis_network.log"
MANIFEST = LOG_DIR / "topis_manifest.csv"
SHOT_DIR = LOG_DIR / "screenshots"

WAIT_SEC = 20
RETRIES = 3
PAUSE_SEC = 2
MISSING_WARN = 0.01  # 2024-08, 2025-07처럼 결측이 튀는 달을 잡기 위한 기준

EXPECTED_HEAD = ["일자", "요일", "도로명", "링크아이디", "시점명", "종점명",
                 "방향", "거리", "차선수", "기능유형구분", "도심/외곽구분", "권역구분"]
N_COLS = 12 + 24  # 정보 12열 + 시간별 속도 24열

# 연구 기간 (11/20 확정): 겨울 11/27~3/10, 여름 5/9~9/23
# 평소값용으로 기간 앞 4주도 필요 -> 여름은 4월 파일까지 받음 (겨울 4주 전 10/30은 11월로 충분)
ALL_SEASONS = ["winter-2022", "winter-2023", "winter-2024", "winter-2025",
               "summer-2023", "summer-2024", "summer-2025", "summer-2026"]

MANIFEST_COLS = ["연월", "상태", "메모", "원본파일명", "서버파일명", "게시글번호",
                 "크기MB", "행수", "링크수", "날짜수", "달력일수", "첫날", "마지막날",
                 "결측칸", "결측률", "속도0칸", "수집시각"]

# onclick="fn_refRoomFileDown('파일명', '서버파일명', '게시글번호', '06')"
ONCLICK_RE = re.compile(r"fn_refRoomFileDown\('([^']*)',\s*'([^']*)',\s*'([^']*)',\s*'([^']*)'\)")
# 2021년 파일만 "01월" 형식이라 0을 선택적으로 받음
NAME_RE = re.compile(r"(\d{4})년\s*0?(\d{1,2})월")


def log(msg):
    print(msg, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}\n")


def season_months(season):
    kind, year = season.split("-")
    y = int(year)
    if kind == "winter":
        return [(y, 11), (y, 12), (y + 1, 1), (y + 1, 2), (y + 1, 3)]
    if kind == "summer":
        return [(y, m) for m in range(4, 10)]
    raise ValueError(f"시즌 이름 오류: {season} (예: winter-2024, summer-2025)")


def ym(y, m):
    return f"{y}-{m:02d}"


def now_str():
    return datetime.now().isoformat(timespec="seconds")


def load_manifest():
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(encoding="utf-8-sig") as f:
        return {row["연월"]: row for row in csv.DictReader(f)}


def save_manifest(rows):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        for key in sorted(rows):
            w.writerow({c: rows[key].get(c, "") for c in MANIFEST_COLS})


# ---------------------------------------------------------------- Selenium
def make_driver(headless):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1400,1000")
    # 브라우저가 보낸 AJAX 요청을 기록해서 PPT 근거로 남김
    options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
    return webdriver.Chrome(options=options)


def open_daily_tab(driver, wait):
    driver.get(PAGE_URL)
    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a[href='#velocityDaily']"))).click()
    # 연도 옵션은 탭을 연 뒤 selectYearList.do로 따로 받아 온다.
    # 그 전에 연도를 바꾸면 "undefined년" + 2014년 목록이 뜨므로 옵션이 채워질 때까지 기다림
    wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "#srchYear option[value]")) >= 2)


def select_year_and_read(driver, wait, year):
    """연도를 고르고 목록이 그 연도로 바뀐 걸 확인한 뒤 {월: 다운로드 정보}를 돌려준다."""
    box = driver.find_element(By.ID, "srchYear")
    if not box.find_elements(By.CSS_SELECTOR, f"option[value='{year}']"):
        raise RuntimeError(f"연도 목록에 {year}년이 없음")
    Select(box).select_by_value(str(year))

    def list_is_new(d):
        # 제목과 12칸, 그리고 모든 파일명이 선택한 연도여야 '새 목록'으로 인정
        if d.find_element(By.ID, "selectedYear").text.strip() != f"{year}년":
            return False
        cells = d.find_elements(By.CSS_SELECTOR, "#resultData1_2 > div")
        if len(cells) != 12:
            return False
        names = [m.group(1) for c in cells
                 for m in [ONCLICK_RE.search(c.get_attribute("innerHTML"))] if m]
        return all(f"{year}년" in n for n in names)

    wait.until(list_is_new)

    files = {}
    for h5 in driver.find_elements(By.CSS_SELECTOR, "#resultData1_2 h5[onclick]"):
        m = ONCLICK_RE.search(h5.get_attribute("onclick"))
        if not m:
            continue
        name, path_nm, seq, div = m.groups()
        nm = NAME_RE.search(name)
        if not nm or int(nm.group(1)) != year:
            log(f"  [건너뜀] 연도가 맞지 않는 항목: {name}")
            continue
        files[int(nm.group(2))] = {"apndFileNm": name, "apndFilePathNm": path_nm,
                                   "bdwrSeq": seq, "blbdDivCd": div}
    return files


def save_network_log(driver):
    lines = []
    for entry in driver.get_log("performance"):
        msg = json.loads(entry["message"])["message"]
        if msg.get("method") != "Network.requestWillBeSent":
            continue
        req = msg["params"]["request"]
        if ".do" in req["url"] and req["method"] == "POST":
            lines.append(f"POST {req['url']} | {req.get('postData', '')}")
    with NET_LOG.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {line}\n")


# ---------------------------------------------------------------- 다운로드
def make_session(driver):
    """Selenium 브라우저의 쿠키를 그대로 쓰는 requests 세션"""
    s = requests.Session()
    for c in driver.get_cookies():
        s.cookies.set(c["name"], c["value"], domain=c.get("domain"))
    s.headers.update({
        "User-Agent": driver.execute_script("return navigator.userAgent"),
        "Referer": PAGE_URL,
    })
    return s


def looks_like_xlsx(path):
    # xlsx는 zip이라 'PK'로 시작. 에러 페이지(HTML)가 저장되는 경우를 거름
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False
    with path.open("rb") as f:
        return f.read(2) == b"PK"


def download(session, info, dest):
    tmp = dest.with_suffix(".part")
    for attempt in range(1, RETRIES + 1):
        try:
            with session.post(DOWNLOAD_URL, data=info, stream=True, timeout=(10, 120)) as r:
                r.raise_for_status()
                expected = int(r.headers.get("Content-Length", 0))
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
            got = tmp.stat().st_size
            if expected and got != expected:
                raise IOError(f"크기 불일치 {got} / {expected} bytes")
            if not looks_like_xlsx(tmp):
                raise IOError("엑셀 형식이 아님")
            tmp.replace(dest)
            return
        except (requests.RequestException, IOError) as e:
            log(f"  [재시도 {attempt}/{RETRIES}] {dest.name}: {e}")
            tmp.unlink(missing_ok=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"{RETRIES}번 시도 후 실패: {dest.name}")


# ---------------------------------------------------------------- 검증 + CSV
def verify_and_convert(xlsx, csv_path, y, m):
    """엑셀을 한 번 읽으면서 CSV로 쓰고 품질 지표를 센다."""
    wb = load_workbook(xlsx, read_only=True)
    rows = wb.active.iter_rows(values_only=True)
    header = list(next(rows))
    notes = []
    if header[:12] != EXPECTED_HEAD or len(header) != N_COLS:
        notes.append(f"열 구성 다름({len(header)}열)")

    prefix = f"{y}{m:02d}"
    dates, links = set(), set()
    n_rows = n_missing = n_zero = n_outside = 0
    tmp = csv_path.with_suffix(".part")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            if row[0] is None:
                continue
            w.writerow(row)
            n_rows += 1
            d = str(int(row[0])) if isinstance(row[0], (int, float)) else str(row[0]).strip()
            dates.add(d)
            if not d.startswith(prefix):
                n_outside += 1
            links.add(row[3])
            # 속도는 "49.1" 같은 글자로, 결측은 "NULL" 글자로 들어 있다.
            # 결측과 속도 0은 의미가 달라서 따로 셈
            for v in row[12:N_COLS]:
                if v is None or str(v).strip() in ("", "NULL"):
                    n_missing += 1
                elif float(v) == 0:
                    n_zero += 1
    wb.close()
    tmp.replace(csv_path)

    days = calendar.monthrange(y, m)[1]
    miss_rate = n_missing / max(n_rows * 24, 1)
    if n_outside:
        notes.append(f"다른 달 날짜 {n_outside}행")
    if len(dates) < days:  # 예: 2020-12는 1~14일만 있음
        notes.append(f"날짜 {len(dates)}/{days}일만 있음")
    if miss_rate > MISSING_WARN:
        notes.append(f"결측률 {miss_rate:.2%} 높음")

    return {"상태": "WARN" if notes else "OK", "메모": "; ".join(notes),
            "행수": n_rows, "링크수": len(links), "날짜수": len(dates), "달력일수": days,
            "첫날": min(dates, default=""), "마지막날": max(dates, default=""),
            "결측칸": n_missing, "결측률": f"{miss_rate:.4f}", "속도0칸": n_zero,
            "크기MB": f"{xlsx.stat().st_size / 1024 / 1024:.1f}"}


# ---------------------------------------------------------------- 실행
def process_month(session, files, manifest, year, m, recheck=False):
    key = ym(year, m)
    xlsx = XLSX_DIR / f"topis_speed_{key}.xlsx"
    csv_path = CSV_DIR / f"topis_speed_{key}.csv"

    done = manifest.get(key, {}).get("상태") in ("OK", "WARN")
    if done and not recheck and looks_like_xlsx(xlsx) and csv_path.exists():
        log(f"[건너뜀] {key}: 이미 완료")
        return "SKIP"
    if m not in files:
        log(f"[실패] {key}: 목록에 파일 없음")
        manifest[key] = {"연월": key, "상태": "FAIL", "메모": "목록에 없음", "수집시각": now_str()}
        return "FAIL"

    info = files[m]
    try:
        if looks_like_xlsx(xlsx):
            log(f"[이어받기] {key}: 엑셀 있음, 검증과 CSV만")
        else:
            log(f"[다운로드] {key}: {info['apndFileNm']}")
            download(session, info, xlsx)
            time.sleep(PAUSE_SEC)
        log(f"[검증] {key}: 엑셀 읽는 중...")
        result = verify_and_convert(xlsx, csv_path, year, m)
    except Exception as e:
        log(f"[실패] {key}: {type(e).__name__}: {e}")
        manifest[key] = {"연월": key, "상태": "FAIL", "메모": f"{type(e).__name__}: {e}",
                         "원본파일명": info["apndFileNm"], "수집시각": now_str()}
        return "FAIL"

    manifest[key] = {"연월": key, "원본파일명": info["apndFileNm"],
                     "서버파일명": info["apndFilePathNm"], "게시글번호": info["bdwrSeq"],
                     "수집시각": now_str(), **result}
    log(f"[{result['상태']}] {key}: {result['행수']:,}행, 링크 {result['링크수']:,}, "
        f"{result['날짜수']}/{result['달력일수']}일, 결측률 {float(result['결측률']):.2%}"
        + (f" ({result['메모']})" if result["메모"] else ""))
    return result["상태"]


def run(targets, headless, recheck=False):
    for d in (XLSX_DIR, CSV_DIR, SHOT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    by_year = defaultdict(list)
    for y, m in targets:
        by_year[y].append(m)
    log(f"=== 수집 시작: {', '.join(ym(y, m) for y, m in targets)} ===")

    driver = make_driver(headless)
    wait = WebDriverWait(driver, WAIT_SEC)
    counts = defaultdict(int)
    try:
        open_daily_tab(driver, wait)
        driver.save_screenshot(str(SHOT_DIR / "topis_0_before.png"))

        for year in sorted(by_year):
            files = select_year_and_read(driver, wait, year)
            driver.save_screenshot(str(SHOT_DIR / f"topis_{year}_after.png"))
            log(f"[목록] {year}년: 파일 있는 달 {sorted(files)}")
            session = make_session(driver)
            for m in sorted(by_year[year]):
                counts[process_month(session, files, manifest, year, m, recheck)] += 1
                save_manifest(manifest)  # 중간에 멈춰도 여기까지는 기록이 남게
        save_network_log(driver)
    finally:
        save_manifest(manifest)
        driver.quit()
    log("=== 끝: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + " ===")


def main():
    p = argparse.ArgumentParser(description="TOPIS 도로별 일자별 통행속도 수집")
    p.add_argument("--season", nargs="+", help="예: winter-2024 summer-2025")
    p.add_argument("--months", nargs="+", help="예: 2024-12 2022-08")
    p.add_argument("--all", action="store_true", help="8개 시즌 전체")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--recheck", action="store_true", help="받은 파일을 다시 검증하고 CSV를 새로 만듦")
    a = p.parse_args()

    if not (a.all or a.season or a.months):
        p.error("--season, --months, --all 중 하나는 필요합니다")
    seasons = ALL_SEASONS + (a.season or []) if a.all else (a.season or [])
    targets = [t for s in seasons for t in season_months(s)]
    for text in a.months or []:
        y, m = text.split("-")
        targets.append((int(y), int(m)))
    run(sorted(set(targets)), a.headless, a.recheck)


if __name__ == "__main__":
    main()
