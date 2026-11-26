"""[서연] 네이버 뉴스 검색 화면 Selenium 동적 수집 및 분석 도로명 탐색.

공개 검색 결과의 메타데이터만 제한적으로 수집한다. 기사 본문·댓글은 수집하지
않으며 로그인, 캡차, 접근 제한을 우회하지 않는다. import만으로 실행되지 않는다.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:  # python crawling/naver_news.py 로 바로 실행해도 src 를 찾게
    sys.path.insert(0, str(ROOT))
from src.io_utils import SPLIT_DIR, data_root as _data_root, derived_root as _derived_root  # noqa: E402

# 원본 데이터 위치: 저장소 data/raw/ (io_utils.data_root)
DEFAULT_DATA_ROOT = _data_root()
SEARCH_URL = "https://search.naver.com/search.naver"
DEFAULT_SNOW_QUERIES = ["서울 폭설 교통", "서울 눈 교통 마비", "서울 폭설 출근길 통제"]
DEFAULT_RAIN_QUERIES = ["서울 폭우 교통", "서울 호우 도로 통제", "서울 침수 도로 통제"]
BLOCK_PATTERNS = ("captcha", "비정상적인 접근", "자동입력 방지", "접근이 제한")
TITLE_BOILERPLATE = {"새 창 열림", "네이버뉴스", "기사원문", "언론사 선정"}
SESSION_END_ERRORS = {"InvalidSessionIdException", "NoSuchWindowException"}


def _selenium():
    try:
        from selenium import webdriver
        from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError as exc:
        raise RuntimeError("selenium이 없습니다. 자동 설치하지 않습니다.") from exc
    return webdriver, StaleElementReferenceException, TimeoutException, By, WebDriverWait


def git_state(root: Path = ROOT) -> dict:
    """[서연] 실패 당시 코드 상태를 재현할 수 있도록 Git 상태를 기록한다."""
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, text=True, capture_output=True, check=False,
            encoding="utf-8", errors="replace",
        )
        return result.stdout.strip() if result.returncode == 0 else "확인 불가"

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "has_uncommitted_changes": bool(status and status != "확인 불가"),
        "status_files": status.splitlines() if status not in ("", "확인 불가") else [],
    }


def load_event_days(root: Path = ROOT) -> pd.DataFrame:
    """[서연] B가 확정한 train/eval 목록에서 악천후일만 읽는다."""
    frames = []
    for split in ("train", "eval"):
        path = root / "data" / "shared" / SPLIT_DIR[split] / f"weather_days_{SPLIT_DIR[split]}.csv"
        if not path.exists():
            raise FileNotFoundError(f"악천후일 목록이 없습니다: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        required = {"date", "role", "split", "intensity", "precip_mm", "new_snow_cm"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} 필수 열 누락: {sorted(missing)}")
        frames.append(frame)
    days = pd.concat(frames, ignore_index=True)
    # [서연] 역할 문자열은 팀 파일 값을 그대로 사용하고, 강도 값이 있는 행만 악천후일로 선택한다.
    days = days.loc[days["intensity"].notna()].copy()
    days["date"] = pd.to_datetime(days["date"], errors="raise")
    if days["date"].duplicated().any():
        raise ValueError("악천후일 날짜가 train/eval 목록에서 중복됩니다.")
    return days.sort_values("date").reset_index(drop=True)


def load_roads(root: Path = ROOT) -> pd.DataFrame:
    roads = pd.read_csv(root / "config/roads.csv", encoding="utf-8-sig", dtype="string")
    required = {"road_id", "road_name"}
    missing = required - set(roads.columns)
    if missing:
        raise ValueError(f"roads.csv 필수 열 누락: {sorted(missing)}")
    return roads[["road_id", "road_name"]].dropna().drop_duplicates()


def _match_key(value: object) -> str:
    """[서연] 공백·구두점을 제거한 보수적 문자열 매칭 키를 만든다."""
    return re.sub(r"[^0-9A-Za-z가-힣]", "", str(value)).casefold()


def load_road_aliases(root: Path = ROOT) -> pd.DataFrame:
    """[서연] 정식 도로명과 검토된 별칭을 합쳐 중복·충돌을 검증한다."""
    roads = load_roads(root)
    aliases = roads.assign(alias=roads["road_name"], alias_source="official", review_status="reviewed")
    path = root / "config/road_aliases.csv"
    if path.exists():
        extra = pd.read_csv(path, encoding="utf-8-sig", dtype="string", keep_default_na=False)
        required = {"road_id", "alias", "review_status"}
        if required - set(extra):
            raise ValueError(f"road_aliases.csv 필수 열 누락: {sorted(required-set(extra))}")
        extra = extra.loc[extra["review_status"].eq("reviewed"), ["road_id", "alias", "review_status"]].copy()
        unknown = sorted(set(extra["road_id"]) - set(roads["road_id"]))
        if unknown:
            raise ValueError(f"road_aliases.csv의 미등록 road_id: {unknown}")
        extra = extra.merge(roads, on="road_id", how="left", validate="many_to_one")
        extra["alias_source"] = "curated_alias"
        aliases = pd.concat([aliases, extra[aliases.columns]], ignore_index=True)
    aliases["alias"] = aliases["alias"].astype(str).str.strip()
    aliases["alias_key"] = aliases["alias"].map(_match_key)
    aliases = aliases.loc[aliases["alias_key"].str.len().ge(2)].drop_duplicates(["road_id", "alias_key"])
    conflicts = aliases.groupby("alias_key")["road_id"].nunique()
    if conflicts.gt(1).any():
        bad = sorted(conflicts[conflicts.gt(1)].index)
        raise ValueError(f"여러 도로에 중복된 별칭: {bad}")
    return aliases.sort_values(["road_id", "alias_source", "alias"]).reset_index(drop=True)


def match_road_aliases(text: object, aliases: pd.DataFrame) -> pd.DataFrame:
    """[서연] 문자열에 포함된 별칭을 도로별로 한 번만 반환한다."""
    key = _match_key(text)
    hits = aliases.loc[aliases["alias_key"].map(lambda alias: alias in key)].copy()
    if hits.empty:
        return hits
    hits["alias_length"] = hits["alias_key"].str.len()
    return hits.sort_values(["road_id", "alias_length"], ascending=[True, False]).drop_duplicates("road_id")


def queries_for_day(row: pd.Series) -> list[str]:
    """[서연] B의 기상 강도를 바꾸지 않고 눈/비 검색어만 선택한다."""
    snow = pd.to_numeric(pd.Series([row.get("new_snow_cm")]), errors="coerce").fillna(0).iloc[0]
    return DEFAULT_SNOW_QUERIES if snow > 0 else DEFAULT_RAIN_QUERIES


def build_search_url(query: str, start: datetime, end: datetime) -> str:
    params = {
        "where": "news",
        "query": query,
        "sm": "tab_opt",
        "sort": "0",
        "photo": "0",
        "field": "0",
        "pd": "3",
        "ds": start.strftime("%Y.%m.%d"),
        "de": end.strftime("%Y.%m.%d"),
    }
    return f"{SEARCH_URL}?{urlencode(params)}"


def create_driver(headless: bool = False):
    webdriver, _, _, _, _ = _selenium()
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1440,1100")
    options.add_argument("--lang=ko-KR")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    return webdriver.Chrome(options=options)


def _check_access(driver) -> None:
    text = (driver.title + " " + driver.page_source[:20000]).lower()
    if any(pattern.lower() in text for pattern in BLOCK_PATTERNS):
        raise RuntimeError("접근 제한 또는 캡차가 감지되어 자동화를 중단합니다.")


def _result_anchors(driver) -> list:
    _, StaleElementReferenceException, _, By, _ = _selenium()
    selectors = [
        "a.news_tit",
        "a[href*='n.news.naver.com/article']",
        "a[href*='news.naver.com/main/read']",
        "#main_pack a[href]",
        "#content a[href]",
        "main a[href]",
    ]
    found = []
    seen = set()
    for selector in selectors:
        for element in driver.find_elements(By.CSS_SELECTOR, selector):
            try:
                url = (element.get_attribute("href") or "").strip()
            except StaleElementReferenceException:
                continue
            try:
                title = (element.get_attribute("title") or element.text or "").strip()
            except StaleElementReferenceException:
                continue
            host = urlparse(url).netloc.lower()
            path = urlparse(url).path.rstrip("/")
            is_known_news = "news.naver.com" in host and ("/article/" in url or "read" in url)
            is_external_article = (bool(host) and bool(path)
                                   and not host.endswith("search.naver.com")
                                   and not host.endswith("naver.com"))
            if url and len(title) >= 6 and (is_known_news or is_external_article) and url not in seen:
                seen.add(url)
                found.append(element)
    return found


def _expanded_container_text(driver, element) -> str:
    """[서연] 바뀌는 CSS 클래스 대신 가까운 상위 결과 블록의 텍스트를 찾는다."""
    text = driver.execute_script(
        """
        let node = arguments[0];
        let best = node.innerText || '';
        for (let i = 0; i < 7 && node.parentElement; i++) {
            node = node.parentElement;
            const value = (node.innerText || '').trim();
            if (value.length > best.length && value.length <= 2500) best = value;
            if (/20\\d{2}[.\\-/]\\d{1,2}[.\\-/]\\d{1,2}|\\d+\\s*(분|시간|일)\\s*전/.test(value)) break;
        }
        return best;
        """,
        element,
    )
    return str(text or "")


def collect_visible_results(driver, query: str, event: pd.Series, roads: pd.DataFrame) -> list[dict]:
    """[서연] 현재 DOM에서 뉴스 URL·제목·화면 날짜 근거만 추출한다."""
    _, StaleElementReferenceException, _, By, _ = _selenium()
    records = []
    date_pattern = re.compile(r"(?:20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}|\d+\s*(?:분|시간|일)\s*전)")
    for element in _result_anchors(driver):
        try:
            url = (element.get_attribute("href") or "").strip()
            raw_title = (element.get_attribute("title") or element.text or "").strip()
            title_lines = [line.strip() for line in raw_title.splitlines()
                           if line.strip() and line.strip() not in TITLE_BOILERPLATE]
            title = " ".join(dict.fromkeys(title_lines)).strip()
            if not url or not title:
                continue
            container_text = _expanded_container_text(driver, element)
            date_match = date_pattern.search(container_text)
            host = urlparse(url).netloc.lower()
            if "news.naver.com" not in host and date_match is None:
                continue
            road_hits = match_road_aliases(title + " " + container_text, roads)
            records.append({
                "event_date": event["date"].strftime("%Y-%m-%d"),
                "split": event["split"],
                "intensity": event["intensity"],
                "query": query,
                "title": title,
                "url": url,
                "published_text": date_match.group(0) if date_match else "",
                "mentioned_road_ids": "|".join(road_hits["road_id"].astype(str)),
                "mentioned_road_names": "|".join(road_hits["road_name"].astype(str)),
                "matched_road_aliases": "|".join(road_hits["alias"].astype(str)),
                "collection_method": "selenium_dynamic_scroll",
            })
        except StaleElementReferenceException:
            continue
    return records


def collect_one_search(
    driver,
    event: pd.Series,
    query: str,
    roads: pd.DataFrame,
    max_scrolls: int = 8,
    wait_seconds: int = 10,
    max_results: int = 30,
    progress: Callable[[str], None] = print,
) -> tuple[list[dict], dict]:
    """[서연] 새 URL 증가를 기준으로 동적 스크롤하고 상한에서 종료한다."""
    _, _, TimeoutException, _, WebDriverWait = _selenium()
    event_date = event["date"].to_pydatetime()
    url = build_search_url(query, event_date - timedelta(days=1), event_date + timedelta(days=1))
    driver.get(url)
    try:
        WebDriverWait(driver, wait_seconds).until(lambda d: d.execute_script("return document.readyState") == "complete")
        WebDriverWait(driver, wait_seconds).until(lambda d: len(_result_anchors(d)) > 0)
    except TimeoutException as exc:
        _check_access(driver)
        raise RuntimeError("뉴스 검색 결과가 제한 시간 안에 표시되지 않았습니다.") from exc
    _check_access(driver)

    collected: dict[str, dict] = {}
    scroll_log = []
    no_growth = 0
    for scroll_index in range(max_scrolls + 1):
        before = len(collected)
        for record in collect_visible_results(driver, query, event, roads):
            # [서연] 같은 제목의 네이버 링크·언론사 원문·반복 노출은 한 후보로 센다.
            title_key = re.sub(r"\s+", "", record["title"]).lower()
            collected.setdefault(title_key, record)
        added = len(collected) - before
        scroll_log.append({"scroll": scroll_index, "unique_results": len(collected), "new_results": added})
        progress(f"[서연] {event['date'].date()} / {query} / scroll={scroll_index} / unique={len(collected)} / new={added}")
        if len(collected) >= max_results:
            break
        if scroll_index > 0 and added == 0:
            no_growth += 1
        else:
            no_growth = 0
        if no_growth >= 2:
            break
        previous_height = driver.execute_script("return document.body.scrollHeight")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        try:
            WebDriverWait(driver, wait_seconds).until(
                lambda d: d.execute_script("return document.body.scrollHeight") > previous_height
                or len(_result_anchors(d)) > len(collected)
            )
        except TimeoutException:
            pass
    records = list(collected.values())[:max_results]
    return records, {"search_url": url, "scrolls": scroll_log, "rows": len(records)}


def _unique_run_dir(parent: Path, prefix: str) -> tuple[str, Path]:
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    version = 1
    while True:
        name = f"{prefix}_{stamp}_v{version:02d}_(중단)"
        path = parent / name
        if not path.exists():
            path.mkdir(parents=True)
            return stamp, path
        version += 1


def run_collection(
    *,
    root: Path = ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    headless: bool = False,
    max_scrolls: int = 8,
    max_results: int = 30,
    retries: int = 2,
    test_date: str | None = None,
    resume_from: Path | None = None,
    save_screenshot: bool = False,
    progress: Callable[[str], None] = print,
) -> dict:
    """[서연] 사용자가 명시적으로 호출할 때만 Selenium 수집과 파일 저장을 실행한다."""
    days = load_event_days(root)
    if test_date:
        target = pd.Timestamp(test_date)
        days = days.loc[days["date"].eq(target)].copy()
        if days.empty:
            raise ValueError(f"악천후일 목록에 없는 테스트 날짜입니다: {test_date}")
    roads = load_road_aliases(root)
    mode = "test" if test_date else "collection"
    log_parent = root / "logs/ysy/naver_news"
    raw_parent = data_root / "naver_news"  # 기사 원문: data/raw/naver_news/
    log_parent.mkdir(parents=True, exist_ok=True)
    raw_parent.mkdir(parents=True, exist_ok=True)
    stamp, run_dir = _unique_run_dir(log_parent, f"naver_news_{mode}")
    started = datetime.now().astimezone()
    driver = None
    all_records = []
    completed_searches: set[tuple[str, str]] = set()
    if resume_from is not None:
        resume_from = Path(resume_from)
        if not resume_from.is_file():
            raise FileNotFoundError(f"이어받을 CSV를 찾을 수 없습니다: {resume_from}")
        previous = pd.read_csv(resume_from, encoding="utf-8-sig")
        required = {"event_date", "query", "url"}
        missing = required.difference(previous.columns)
        if missing:
            raise ValueError(f"이어받기 CSV 필수 열 누락: {sorted(missing)}")
        previous["event_date"] = pd.to_datetime(previous["event_date"]).dt.strftime("%Y-%m-%d")
        all_records = previous.to_dict("records")
        completed_searches = set(zip(previous["event_date"], previous["query"]))
        progress(f"[서연] 기존 {len(previous):,}행, 완료 검색 {len(completed_searches):,}개를 이어받습니다.")
    search_logs = []
    failed_searches = []
    error = None
    final_status = "실패"
    try:
        driver = create_driver(headless=headless)
        for _, event in days.iterrows():
            for query in queries_for_day(event):
                event_date = event["date"].strftime("%Y-%m-%d")
                if (event_date, query) in completed_searches:
                    search_logs.append({"event_date": event_date, "query": query,
                                        "attempt": 0, "status": "기존결과사용"})
                    continue
                last_error = None
                for attempt in range(1, retries + 2):
                    try:
                        records, detail = collect_one_search(
                            driver, event, query, roads,
                            max_scrolls=max_scrolls, max_results=max_results, progress=progress,
                        )
                        detail.update({"event_date": event["date"].strftime("%Y-%m-%d"),
                                       "query": query, "attempt": attempt, "status": "성공"})
                        search_logs.append(detail)
                        all_records.extend(records)
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        error_type = type(exc).__name__
                        session_restarted = False
                        # [서연] 종료된 Chrome 세션은 재사용할 수 없으므로 새 전용 세션으로만 재시도한다.
                        if error_type in SESSION_END_ERRORS and attempt <= retries:
                            try:
                                driver.quit()
                            except Exception:
                                pass
                            driver = create_driver(headless=headless)
                            session_restarted = True
                            progress("[서연] Chrome 세션 종료 감지: 새 전용 브라우저로 재시도합니다.")
                        search_logs.append({"event_date": event["date"].strftime("%Y-%m-%d"),
                                            "query": query, "attempt": attempt, "status": "실패",
                                            "error_type": error_type, "error": str(exc),
                                            "session_restarted": session_restarted})
                        progress(f"[서연] 재시도 {attempt}/{retries + 1}: {type(exc).__name__}: {exc}")
                        if "캡차" in str(exc) or "접근 제한" in str(exc):
                            raise
                        time.sleep(min(2 * attempt, 6))
                if last_error is not None:
                    failed_searches.append({"event_date": event["date"].strftime("%Y-%m-%d"),
                                            "query": query, "error_type": type(last_error).__name__,
                                            "error": str(last_error)})
                    continue
        frame = pd.DataFrame(all_records).drop_duplicates(subset=["event_date", "query", "url"])
        if frame.empty:
            raise RuntimeError("수집된 뉴스 검색 결과가 0건입니다.")
        final_status = "부분성공" if failed_searches else "성공"
        output_path = raw_parent / f"naver_news_raw_{stamp}_v01_({final_status}).csv"
        if output_path.exists():
            raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {output_path}")
        frame["collected_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        if save_screenshot and driver is not None:
            driver.save_screenshot(str(run_dir / f"naver_news_final_{stamp}.png"))
    except KeyboardInterrupt:
        error = {"type": "KeyboardInterrupt", "message": "사용자가 실행을 중단했습니다."}
        final_status = "중단"
        output_path = None
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        output_path = None
        # [서연] 선택자 변경·접근 제한을 재현할 수 있도록 실패 당시 공개 검색 화면만 보존한다.
        if driver is not None:
            try:
                driver.save_screenshot(str(run_dir / f"naver_news_failure_{stamp}.png"))
                (run_dir / f"naver_news_failure_{stamp}.html").write_text(
                    driver.page_source, encoding="utf-8"
                )
            except Exception:
                pass
    finally:
        if driver is not None:
            driver.quit()

    final_dir = run_dir.with_name(run_dir.name.rsplit("_(", 1)[0] + f"_({final_status})")
    if final_dir.exists():
        raise FileExistsError(f"실행 기록 폴더가 이미 존재합니다: {final_dir}")
    run_dir.rename(final_dir)
    finished = datetime.now().astimezone()
    summary = {
        "status": final_status,
        "mode": mode,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "event_days": len(days),
        "search_attempts": len(search_logs),
        "failed_searches": failed_searches,
        "rows": len(pd.DataFrame(all_records).drop_duplicates(subset=["event_date", "query", "url"])) if all_records else 0,
        "road_mentioned_rows": int(pd.DataFrame(all_records).drop_duplicates(subset=["event_date", "query", "url"])["mentioned_road_ids"].fillna("").astype(str).str.strip().ne("").sum()) if all_records else 0,
        "headless": headless,
        "max_scrolls": max_scrolls,
        "max_results_per_search": max_results,
        "retries": retries,
        "resume_from": str(resume_from) if resume_from else None,
        "resumed_searches": len(completed_searches),
        "git": git_state(root),
        "output_path": str(output_path) if output_path else None,
        "run_dir": str(final_dir),
        "search_logs": search_logs,
        "error": error,
    }
    summary_path = final_dir / f"naver_news_summary_{stamp}_({final_status}).json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    if final_status in {"실패", "중단"}:
        raise RuntimeError(f"뉴스 수집 {final_status} 기록: {final_dir}; 원인: {error}")
    return summary


def preprocess_news(raw_path: Path, output_dir: Path, log_dir: Path) -> dict:
    """[서연] 원자료를 보존하고 기사 및 날짜별 연결 후보를 일괄 정리한다."""
    import hashlib
    import unicodedata
    from urllib.parse import urlsplit, urlunsplit, parse_qsl

    df = pd.read_csv(raw_path, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    required = {'event_date','split','intensity','query','title','url','published_text',
                'mentioned_road_ids','mentioned_road_names'}
    if required - set(df):
        raise ValueError(f'필수 열 누락: {sorted(required-set(df))}')
    df.insert(0, 'raw_row', range(1, len(df)+1))
    def canonical(url):
        p = urlsplit(url.strip())
        q = [(k,v) for k,v in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith('utm_')]
        return urlunsplit((p.scheme.lower(),p.netloc.lower(),p.path,urlencode(sorted(q)),''))
    def title_key(t):
        return re.sub(r'[^\w]', '', unicodedata.normalize('NFKC',t).casefold())
    df['canonical_url'] = df.url.map(canonical)
    df['title_key'] = df.title.map(title_key)
    df['article_id'] = df.canonical_url.map(lambda x: hashlib.sha256(x.encode()).hexdigest()[:24])
    df['same_title_urls'] = df.groupby('title_key').canonical_url.transform('nunique')
    dates = pd.to_datetime(df.published_text.str.replace(r'[./]', '-', regex=True), format='%Y-%m-%d', errors='coerce')
    event = pd.to_datetime(df.event_date, format='%Y-%m-%d', errors='raise')
    df['published_date_candidate'] = dates.dt.strftime('%Y-%m-%d').fillna('')
    df['date_offset'] = (dates-event).dt.days.astype('Int64')
    df['date_status'] = ['review' if pd.isna(x) else 'in_window' if abs(x)<=1 else 'out_of_window' for x in df.date_offset]
    patterns = {'도로통제':r'통제|통행\s*제한|도로\s*폐쇄|통행\s*재개', '교통사고':r'추돌|충돌|교통\s*사고|차량\s*사고|전복', '정체':r'정체|교통\s*(?:혼잡|마비|대란|체증)|출근길\s*비상|퇴근길\s*비상', '침수':r'침수|범람|물에\s*잠', '결빙':r'결빙|빙판|블랙\s*아이스'}
    weather = re.compile(r'폭우|호우|폭설|대설|강설|강우|장맛비|장마|눈폭탄|눈길|빗길|결빙|빙판|태풍|집중\s*호우|폭풍|폭설|폭우')
    traffic = re.compile(r'도로|교통|차량|통행|출근길|퇴근길|운전|추돌|정체|잠수교|고속도로')
    df['article_types'] = df.title.map(lambda t:'|'.join(k for k,p in patterns.items() if re.search(p,t)))
    df['weather_title'] = df.title.map(lambda t:bool(weather.search(t)))
    df['traffic_title'] = df.title.map(lambda t:bool(traffic.search(t)))
    df['relevance'] = ['candidate' if w and t else 'review' if w or t or r else 'low_evidence' for w,t,r in zip(df.weather_title,df.traffic_title,df.mentioned_road_ids)]
    road_rows=[]
    for row in df.itertuples(index=False):
        ids=row.mentioned_road_ids.split('|') if row.mentioned_road_ids.strip() else []
        names=row.mentioned_road_names.split('|') if row.mentioned_road_names.strip() else []
        if len(ids)!=len(names):
            raise ValueError(f'도로 ID/이름 개수 불일치: raw_row={row.raw_row}')
        for rid,name in zip(ids,names):
            if not rid.strip() or not name.strip(): continue
            direct = bool(title_key(name)) and title_key(name) in title_key(row.title)
            road_rows.append({'raw_row':row.raw_row,'article_id':row.article_id,'event_date':row.event_date,'split':row.split,'road_id':rid,'road_name':name,'title':row.title,'url':row.url,'article_types':row.article_types,'date_status':row.date_status,'relevance':row.relevance,'road_evidence':'title' if direct else 'search_context_only','case_status':'candidate_review' if direct and row.relevance=='candidate' and row.date_status=='in_window' else 'manual_review'})
    roads=pd.DataFrame(road_rows,columns=['raw_row','article_id','event_date','split','road_id','road_name','title','url','article_types','date_status','relevance','road_evidence','case_status'])
    # [서연] URL별 기사 목록과 날짜별 연결을 분리하고 원본 검색어는 유지한다.
    articles=df.drop_duplicates('article_id')[['article_id','canonical_url','title','title_key','same_title_urls']].copy()
    links=df.copy()
    links['queries_for_event_article']=links.groupby(['event_date','split','article_id'])['query'].transform(lambda s:'|'.join(sorted(set(s))))
    links=links.sort_values(['event_date','article_id','raw_row']).drop_duplicates(['event_date','split','article_id'])
    roads=roads.sort_values(['case_status','raw_row']).drop_duplicates(['event_date','split','article_id','road_id'])
    stamp=datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%f')
    output_dir.mkdir(parents=True,exist_ok=True); log_dir.mkdir(parents=True,exist_ok=True)
    outputs={}
    for name,table in [('news_audit',df),('news_articles',articles),('news_event_links',links),('news_road_candidates',roads)]:
        path=output_dir/f'{name}_{stamp}_(성공)_v01.csv'
        with path.open('x',encoding='utf-8-sig',newline='') as f: table.to_csv(f,index=False)
        outputs[name]=str(path)
    summary={'status':'성공','source':str(raw_path),'raw_rows':len(df),'raw_unique_urls':int(df.url.nunique()),'raw_road_mentioned_rows':int(df.mentioned_road_ids.str.strip().ne('').sum()),'articles':len(articles),'event_links':len(links),'road_candidates':len(roads),'date_status':df.date_status.value_counts().to_dict(),'relevance':df.relevance.value_counts().to_dict(),'road_case_status':roads.case_status.value_counts().to_dict(),'outputs':outputs,'limitations':'제목 및 검색 화면 메타데이터 기반 후보이며 기사 본문, 실제 사건일, 인과관계는 미검증'}
    path=log_dir/f'yang_seoyeon_news_preprocess_{stamp}_(성공)_v01.json'
    with path.open('x',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    return summary


def analyze_existing_news(
    raw_path: Path,
    output_dir: Path,
    log_dir: Path,
    root: Path = ROOT,
    top_terms: int = 100,
) -> dict:
    """[서연] 기존 뉴스 제목에 별칭을 재적용하고 검색어 후보 근거를 만든다.

    기사 본문을 요청하거나 Selenium을 실행하지 않는다. 단어 후보는 사람이 검토할
    자료일 뿐 추가 검색어로 자동 확정하지 않는다.
    """
    df = pd.read_csv(raw_path, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    required = {"event_date", "split", "title", "url", "mentioned_road_ids"}
    if required - set(df):
        raise ValueError(f"기존 뉴스 필수 열 누락: {sorted(required-set(df))}")
    aliases = load_road_aliases(root)
    rows: list[dict] = []
    for raw_row, row in enumerate(df.itertuples(index=False), start=1):
        existing_ids = {value for value in str(row.mentioned_road_ids).split("|") if value}
        for hit in match_road_aliases(row.title, aliases).itertuples(index=False):
            rows.append({
                "raw_row": raw_row,
                "event_date": row.event_date,
                "split": row.split,
                "title": row.title,
                "url": row.url,
                "road_id": hit.road_id,
                "road_name": hit.road_name,
                "matched_alias": hit.alias,
                "alias_source": hit.alias_source,
                "already_matched_in_raw": hit.road_id in existing_ids,
                "match_status": "existing_match" if hit.road_id in existing_ids else "new_alias_match_review_needed",
            })
    rematch = pd.DataFrame(rows, columns=[
        "raw_row", "event_date", "split", "title", "url", "road_id", "road_name",
        "matched_alias", "alias_source", "already_matched_in_raw", "match_status",
    ]).drop_duplicates(["event_date", "url", "road_id"])

    stopwords = {
        "서울", "서울시", "교통", "도로", "기자", "관련", "대한", "이번", "오늘",
        "내일", "오전", "오후", "지역", "가운데", "통해", "위해", "부터", "까지",
    }
    titles = df[["url", "title"]].drop_duplicates("url")["title"].astype(str).tolist()
    documents: list[list[str]] = []
    for title in titles:
        tokens = [token for token in re.findall(r"[가-힣A-Za-z0-9]+", title.casefold())
                  if len(token) >= 2 and token not in stopwords and not token.isdigit()]
        documents.append(tokens)
    doc_freq: Counter[str] = Counter()
    total_freq: Counter[str] = Counter()
    tfidf_sum: Counter[str] = Counter()
    n_docs = len(documents)
    for tokens in documents:
        counts = Counter(tokens)
        total_freq.update(counts)
        doc_freq.update(counts.keys())
    for tokens in documents:
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        for term, count in counts.items():
            idf = math.log((1 + n_docs) / (1 + doc_freq[term])) + 1
            tfidf_sum[term] += (count / total) * idf
    terms = pd.DataFrame([
        {
            "term": term,
            "document_frequency": doc_freq[term],
            "total_frequency": total_freq[term],
            "mean_tfidf": tfidf_sum[term] / n_docs if n_docs else 0,
            "selection_status": "manual_review_needed",
            "review_notes": "",
        }
        for term in total_freq
    ])
    if not terms.empty:
        terms = terms.sort_values(
            ["document_frequency", "mean_tfidf", "term"], ascending=[False, False, True]
        ).head(top_terms).reset_index(drop=True)

    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    rematch_path = output_dir / f"news_alias_rematch_{stamp}_(성공)_v01.csv"
    terms_path = output_dir / f"news_term_candidates_{stamp}_(성공)_v01.csv"
    with rematch_path.open("x", encoding="utf-8-sig", newline="") as file:
        rematch.to_csv(file, index=False)
    with terms_path.open("x", encoding="utf-8-sig", newline="") as file:
        terms.to_csv(file, index=False)
    summary = {
        "status": "성공",
        "source": str(raw_path),
        "source_rows": int(len(df)),
        "unique_articles": int(len(titles)),
        "alias_rows": int(len(aliases)),
        "rematched_rows": int(len(rematch)),
        "new_alias_match_review_needed": int(rematch["match_status"].eq("new_alias_match_review_needed").sum()),
        "term_candidates": int(len(terms)),
        "outputs": {"alias_rematch": str(rematch_path), "term_candidates": str(terms_path)},
        "limitations": "제목 문자열 기반 후보. 기사 본문·도로 사건 관계 미확정, 검색어는 사람 검토 후 선택",
    }
    log_path = log_dir / f"news_alias_analysis_{stamp}_(성공)_v01.json"
    with log_path.open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    summary["log"] = str(log_path)
    return summary


def preprocess_main():
    parser=argparse.ArgumentParser(description='[서연] 뉴스 원자료 전처리')
    parser.add_argument('mode',choices=['preprocess'])
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=_derived_root()/'naver_news')
    parser.add_argument('--log-dir',type=Path,default=ROOT/'logs'/'ysy'/'naver_news')
    args=parser.parse_args()
    print(json.dumps(preprocess_news(args.input,args.output_dir,args.log_dir),ensure_ascii=False,indent=2))


def analyze_existing_main() -> None:
    parser = argparse.ArgumentParser(description="[서연] 기존 뉴스 별칭 재매칭 및 검색어 후보 분석")
    parser.add_argument("mode", choices=["analyze-existing"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=_derived_root() / "naver_news")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs" / "ysy" / "naver_news")
    parser.add_argument("--top-terms", type=int, default=100)
    args = parser.parse_args()
    result = analyze_existing_news(args.input, args.output_dir, args.log_dir, top_terms=args.top_terms)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "preprocess":
        preprocess_main()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "analyze-existing":
        analyze_existing_main()
        return
    parser = argparse.ArgumentParser(description="[서연] 네이버 뉴스 Selenium 동적 수집")
    parser.add_argument("mode", choices=["test", "collect"])
    parser.add_argument("--test-date", help="test 모드에서 사용할 B 확정 악천후일 YYYY-MM-DD")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-scrolls", type=int, default=8)
    parser.add_argument("--max-results", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume-from", type=Path, help="부분성공 CSV를 읽어 완료된 날짜·검색어를 건너뜁니다.")
    parser.add_argument("--save-screenshot", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    if args.mode == "test" and not args.test_date:
        parser.error("test 모드에는 --test-date가 필요합니다.")
    if args.mode == "collect" and args.test_date:
        parser.error("collect 모드에서는 --test-date를 사용하지 않습니다.")
    result = run_collection(
        data_root=args.data_root,
        headless=args.headless,
        max_scrolls=args.max_scrolls,
        max_results=args.max_results,
        retries=args.retries,
        test_date=args.test_date if args.mode == "test" else None,
        resume_from=args.resume_from,
        save_screenshot=args.save_screenshot,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
