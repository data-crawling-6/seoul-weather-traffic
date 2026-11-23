"""TOPIS 월별 교통량 XLSX 수집·전처리 (C 담당).

수집은 TOPIS의 실제 화면에서 ``지점별 일자별 교통량`` 탭을 열고,
연도를 바꿨을 때 AJAX로 갱신되는 월 목록을 확인한 뒤 다운로드 버튼을
누르는 방식이다. import만으로 브라우저·네트워크·파일 저장은 발생하지 않는다.

실행 단계
    python crawling/traffic_volume.py months
    python crawling/traffic_volume.py download
    python crawling/traffic_volume.py preprocess

원본과 결과의 기본 위치는 ``SEOUL_TRAFFIC_DATA_ROOT`` 환경변수이며,
값이 없으면 ``D:\\seoul-weather-traffic-data``를 사용한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]
TOPIS_URL = "https://topis.seoul.go.kr/refRoom/openRefRoom_2.do"
DAY_FILES = (
    ROOT / "data" / "processed" / "train" / "weather_days_train.csv",
    ROOT / "data" / "processed" / "eval" / "weather_days_eval.csv",
)

DATA_ROOT_ENV = "SEOUL_TRAFFIC_DATA_ROOT"
DEFAULT_DATA_ROOT = Path(r"D:\seoul-weather-traffic-data")
RAW_SUBDIR = Path("traffic_volume/raw/xlsx")
PROCESSED_SUBDIR = Path("traffic_volume/processed")
LOG_SUBDIR = Path("traffic_volume/logs/yang_seoyeon")
RUN_SUBDIR = Path("traffic_volume/runs/yang_seoyeon")
LEGACY_RUN_SUBDIR = Path("traffic_volume/runs")

# 교통량 XLSX는 시작 시각, TOPIS 속도는 끝 시각을 열 이름으로 쓴다.
# 7시(07:00~08:00) -> TOPIS ~08시와 결합한다.
VOLUME_TO_TOPIS_END_HOUR = {7: 8, 8: 9, 17: 18, 18: 19, 19: 20}
OUTPUT_COLUMNS = [
    "date",
    "split",
    "role",
    "season",
    "season_name",
    "spot_id",
    "spot_name",
    "road_name",
    "io_type",
    "direction_detail",
    "latitude",
    "longitude",
    "volume_start_hour",
    "hour",
    "commute_period",
    "traffic_volume",
    "source_month",
    "source_file",
]


def data_root() -> Path:
    """외부 데이터 루트를 반환한다."""
    return Path(os.getenv(DATA_ROOT_ENV, str(DEFAULT_DATA_ROOT))).expanduser().resolve()


def _now() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _file_sha256(path: Path) -> str | None:
    """파일이 존재할 때 코드 버전 확인용 SHA-256을 반환한다."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(project_root: Path = ROOT) -> dict[str, object]:
    """현재 Git 커밋과 미커밋 파일명을 기록한다. diff 본문은 저장하지 않는다."""
    def git(*args: str) -> tuple[int, str]:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return completed.returncode, completed.stdout.strip()

    commit_code, commit = git("rev-parse", "HEAD")
    branch_code, branch = git("branch", "--show-current")
    status_code, status = git("status", "--porcelain")
    changed_files = [line[3:] for line in status.splitlines() if len(line) >= 4]
    code_files = [
        project_root / "crawling" / "traffic_volume.py",
        project_root / "notebooks" / "00_ysy_collect.ipynb",
    ]
    return {
        "git_available": commit_code == branch_code == status_code == 0,
        "git_commit": commit or None,
        "git_branch": branch or None,
        "git_dirty": bool(status),
        "git_changed_files": changed_files,
        "code_sha256": {
            str(path.relative_to(project_root)): _file_sha256(path) for path in code_files
        },
    }


def _unique_run_dir(base: Path, run_id: str, label: str = "") -> tuple[Path, str]:
    """처음부터 (중단)으로 표시된 고유 실행 폴더를 만든다."""
    runs_dir = base / RUN_SUBDIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", label).strip("_")
    label_part = f"_{safe_label}" if safe_label else ""
    for version in range(1, 1000):
        version_text = f"v{version:02d}"
        candidate = runs_dir / (
            f"yang_seoyeon_traffic_volume{label_part}_{run_id}_{version_text}_(중단)"
        )
        try:
            candidate.mkdir()
            return candidate, version_text
        except FileExistsError:
            continue
    raise RuntimeError("고유 실행 폴더 버전을 만들지 못했습니다.")


def _rename_status(path: Path, old: str, new: str) -> Path:
    """파일 또는 폴더 이름의 상태를 바꾸되 기존 대상은 덮어쓰지 않는다."""
    destination = path.with_name(path.name.replace(f"({old})", f"({new})"))
    if destination.exists():
        raise FileExistsError(destination)
    path.rename(destination)
    return destination


def list_run_history(root: Path | None = None) -> pd.DataFrame:
    """과거 성공·실패·중단 실행을 읽기만 한다. Selenium을 실행하지 않는다."""
    base = (root or data_root()).resolve()
    columns = [
        "run_dir", "run_id", "version", "run_mode", "test_month", "status",
        "started_at", "finished_at",
        "git_commit", "git_branch", "git_dirty", "downloaded", "skipped",
        "failed", "retries", "rows", "summary_path",
    ]
    rows: list[dict[str, object]] = []
    run_dirs: list[Path] = []
    seen: set[Path] = set()
    # 새 개인 경로와 기존 경로를 함께 읽어 과거 실행 이력을 보존한다.
    for history_root in (base / RUN_SUBDIR, base / LEGACY_RUN_SUBDIR):
        if not history_root.is_dir():
            continue
        for run_dir in sorted(history_root.glob("yang_seoyeon_traffic_volume_*")):
            resolved = run_dir.resolve()
            if resolved not in seen:
                seen.add(resolved)
                run_dirs.append(run_dir)
    for run_dir in run_dirs:
        status_match = re.search(r"\((성공|실패|중단)\)$", run_dir.name)
        status = status_match.group(1) if status_match else "미확인"
        summaries = sorted(run_dir.glob("yang_seoyeon_summary_*.json"))
        configs = sorted(run_dir.glob("yang_seoyeon_run_config_*.json"))
        summary: dict[str, object] = {}
        config: dict[str, object] = {}
        try:
            if summaries:
                summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
            if configs:
                config = json.loads(configs[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        provenance = config.get("code_provenance", {}) if isinstance(config, dict) else {}
        counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
        rows.append({
            "run_dir": str(run_dir),
            "run_id": config.get("run_id"),
            "version": config.get("version"),
            "run_mode": config.get("run_mode", "full_pipeline"),
            "test_month": config.get("test_month"),
            "status": status,
            "started_at": config.get("started_at"),
            "finished_at": summary.get("finished_at"),
            "git_commit": provenance.get("git_commit") if isinstance(provenance, dict) else None,
            "git_branch": provenance.get("git_branch") if isinstance(provenance, dict) else None,
            "git_dirty": provenance.get("git_dirty") if isinstance(provenance, dict) else None,
            "downloaded": counts.get("downloaded"),
            "skipped": counts.get("skipped"),
            "failed": counts.get("failed"),
            "retries": counts.get("retries"),
            "rows": summary.get("rows"),
            "summary_path": str(summaries[-1]) if summaries else None,
        })
    return pd.DataFrame(rows, columns=columns)


def load_target_days(day_files: Iterable[Path] = DAY_FILES) -> pd.DataFrame:
    """정희의 train/eval 악천후일·비교일 90일을 읽고 검증한다."""
    frames: list[pd.DataFrame] = []
    for path in day_files:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, encoding="utf-8-sig")
        required = {"date", "role", "season", "split", "season_name"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path.name} 필수 열 누락: {missing}")
        frames.append(frame)

    days = pd.concat(frames, ignore_index=True)
    days["date"] = pd.to_datetime(days["date"], errors="raise").dt.normalize()
    if days["date"].duplicated().any():
        duplicates = days.loc[days["date"].duplicated(False), "date"].dt.strftime("%Y-%m-%d")
        raise ValueError(f"90일 목록에 중복 날짜가 있습니다: {duplicates.tolist()}")
    if len(days) != 90:
        raise ValueError(f"악천후일·비교일 목록은 90행이어야 합니다: 실제 {len(days)}행")
    return days.sort_values("date").reset_index(drop=True)


def required_months(days: pd.DataFrame | None = None) -> list[str]:
    """90일 목록에서 중복 없는 YYYY-MM 26개를 정렬해 반환한다."""
    target = load_target_days() if days is None else days
    months = sorted(target["date"].dt.strftime("%Y-%m").unique().tolist())
    if len(months) != 26:
        raise ValueError(f"필요 연월은 현재 기준 26개여야 합니다: 실제 {len(months)}개")
    return months


def _selenium():
    try:
        from selenium import webdriver
        from selenium.common.exceptions import StaleElementReferenceException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import Select, WebDriverWait
    except ImportError as exc:
        raise RuntimeError("selenium이 없습니다. 자동 설치하지 않습니다.") from exc
    return webdriver, StaleElementReferenceException, By, Select, WebDriverWait


def build_driver(download_dir: Path, *, headless: bool = False):
    """개인 Chrome 프로필을 사용하지 않는 다운로드 전용 브라우저를 만든다."""
    webdriver, _, _, _, _ = _selenium()
    download_dir.mkdir(parents=True, exist_ok=True)
    options = webdriver.ChromeOptions()
    options.add_argument("--lang=ko-KR")
    options.add_argument("--window-size=1440,1000")
    options.add_argument("--disable-notifications")
    if headless:
        options.add_argument("--headless=new")
    options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": str(download_dir.resolve()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        },
    )
    return webdriver.Chrome(options=options)


def _wait_for_new_xlsx(
    directory: Path,
    before: set[Path],
    *,
    timeout: int,
) -> Path:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        partial = list(directory.glob("*.crdownload")) + list(directory.glob("*.tmp"))
        completed = [
            path
            for path in set(directory.glob("*.xlsx")) - before
            if path.is_file() and path.stat().st_size > 0
        ]
        if completed and not partial:
            return max(completed, key=lambda path: path.stat().st_mtime)
        time.sleep(0.5)
    raise TimeoutError(f"{timeout}초 안에 XLSX 다운로드 완료를 확인하지 못했습니다.")


def _open_daily_volume_tab(driver, wait_seconds: int) -> tuple[object, object, object]:
    _, _, By, Select, WebDriverWait = _selenium()
    wait = WebDriverWait(driver, wait_seconds)
    driver.get(TOPIS_URL)
    tab = wait.until(
        lambda current: current.find_element(By.CSS_SELECTOR, "a[href='#trafficvolDaily']")
    )
    driver.execute_script("arguments[0].click();", tab)
    wait.until(lambda current: len(Select(current.find_element(By.ID, "srchYear")).options) > 1)
    return By, Select, wait


def _select_year_and_wait(driver, By, Select, wait, year: str) -> None:
    """연도를 변경하고 AJAX 월 목록이 새 DOM으로 교체될 때까지 기다린다."""
    previous = driver.find_elements(By.CSS_SELECTOR, "#resultData2_2 h5")
    previous_first = previous[0] if previous else None
    Select(driver.find_element(By.ID, "srchYear")).select_by_value(year)
    driver.execute_script("$('#srchYear').trigger('change');")
    wait.until(lambda current: current.find_element(By.ID, "selectedYear").text.strip().startswith(year))
    if previous_first is not None:
        try:
            wait.until(lambda _: not previous_first.is_enabled())
        except Exception:
            # selectedYear와 아래 월 버튼 존재 여부를 다시 확인하므로 여기서 무한 대기하지 않는다.
            pass
    wait.until(lambda current: len(current.find_elements(By.CSS_SELECTOR, "#resultData2_2 .active h5[onclick]")) > 0)


def download_required_months(
    *,
    root: Path | None = None,
    months: Iterable[str] | None = None,
    headless: bool = False,
    retries: int = 3,
    wait_seconds: int = 60,
    run_dir: Path | None = None,
    run_id: str | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """TOPIS 화면을 조작해 필요한 월별 XLSX를 다운로드한다.

    이미 ``topis_traffic_volume_YYYY-MM.xlsx``가 있으면 크기가 0이 아닌지
    확인하고 건너뛴다. 실패한 월은 제한 횟수만 재시도하고 다음 월로 진행한다.
    """
    base = (root or data_root()).resolve()
    raw_dir = base / RAW_SUBDIR
    record_dir = run_dir or (base / LOG_SUBDIR)
    record_dir.mkdir(parents=True, exist_ok=True)
    effective_run_id = run_id or datetime.now(KST).strftime("%Y%m%dT%H%M%S")
    log_path = record_dir / f"yang_seoyeon_collection_log_{effective_run_id}_(중단).jsonl"
    manifest_path = record_dir / f"yang_seoyeon_download_manifest_{effective_run_id}_(중단).csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    targets = list(months or required_months())
    invalid = [month for month in targets if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month)]
    if invalid:
        raise ValueError(f"잘못된 연월 형식: {invalid}")

    summary: dict[str, object] = {
        "downloaded": [], "skipped": [], "failed": [], "retries": 0,
        "log_path": str(log_path), "manifest_path": str(manifest_path),
    }
    manifest_rows: list[dict[str, object]] = []
    driver = None
    try:
        driver = build_driver(raw_dir, headless=headless)
        By, Select, wait = _open_daily_volume_tab(driver, wait_seconds)
        capabilities = driver.capabilities or {}
        browser_event = {
            "at": _now(),
            "run_id": effective_run_id,
            "status": "browser_started",
            "browser_name": capabilities.get("browserName"),
            "browser_version": capabilities.get("browserVersion"),
            "driver_version": (capabilities.get("chrome", {}) or {}).get("chromedriverVersion"),
            "headless": headless,
        }
        _append_jsonl(log_path, browser_event)
        for month in targets:
            destination = raw_dir / f"yang_seoyeon_topis_traffic_volume_{month}.xlsx"
            if destination.is_file() and destination.stat().st_size > 0:
                progress(f"[건너뜀] {month}: 기존 파일 {destination.name}")
                summary["skipped"].append(month)
                event = {
                    "at": _now(), "run_id": effective_run_id, "month": month,
                    "status": "skipped_existing", "path": str(destination),
                    "bytes": destination.stat().st_size,
                }
                manifest_rows.append(event)
                pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
                _append_jsonl(log_path, event)
                continue

            year, month_number = month.split("-")
            last_error = ""
            for attempt in range(1, retries + 1):
                try:
                    _select_year_and_wait(driver, By, Select, wait, year)
                    xpath = (
                        "//*[@id='resultData2_2']//div[contains(@class,'active')]"
                        f"/h5[starts-with(normalize-space(.),'{int(month_number)}월')]"
                    )
                    button = wait.until(lambda current: current.find_element(By.XPATH, xpath))
                    label = button.text.strip()
                    before = set(raw_dir.glob("*.xlsx"))
                    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", button)
                    driver.execute_script("arguments[0].click();", button)
                    downloaded = _wait_for_new_xlsx(raw_dir, before, timeout=wait_seconds)
                    if destination.exists():
                        raise FileExistsError(f"대상 파일이 이미 생겼습니다: {destination}")
                    downloaded.replace(destination)
                    progress(f"[완료] {month}: {destination.name} ({destination.stat().st_size:,} bytes)")
                    summary["downloaded"].append(month)
                    event = {
                        "at": _now(), "run_id": effective_run_id, "month": month,
                        "status": "downloaded", "attempt": attempt,
                        "screen_label": label, "path": str(destination),
                        "bytes": destination.stat().st_size,
                    }
                    manifest_rows.append(event)
                    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
                    _append_jsonl(log_path, event)
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    summary["retries"] += 1
                    progress(f"[재시도 {attempt}/{retries}] {month}: {last_error}")
                    _append_jsonl(
                        log_path,
                        {
                            "at": _now(), "run_id": effective_run_id, "month": month,
                            "status": "retry", "attempt": attempt, "error": last_error,
                        },
                    )
                    if attempt < retries:
                        time.sleep(min(2**attempt, 8))
                        driver.get(TOPIS_URL)
                        By, Select, wait = _open_daily_volume_tab(driver, wait_seconds)
            else:
                summary["failed"].append({"month": month, "error": last_error})
                event = {
                    "at": _now(), "run_id": effective_run_id, "month": month,
                    "status": "failed", "error": last_error,
                }
                manifest_rows.append(event)
                pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
                _append_jsonl(log_path, event)
    finally:
        if driver is not None:
            driver.quit()

    summary["finished_at"] = _now()
    summary["raw_dir"] = str(raw_dir)
    summary["screenshots"] = []
    return summary


def _find_sheet(sheet_names: list[str], pattern: str) -> str:
    matches = [name for name in sheet_names if re.search(pattern, name)]
    if len(matches) != 1:
        raise ValueError(f"시트를 하나로 식별하지 못했습니다: pattern={pattern!r}, sheets={sheet_names}")
    return matches[0]


def _month_sheet_pattern(month: str) -> str:
    """[서연] `1월`과 `01월` 형식의 월별 교통량 시트명을 모두 허용한다."""
    year = month[:4]
    month_number = int(month[5:])
    month_text = rf"0?{month_number}" if month_number < 10 else str(month_number)
    return rf"^{year}년\s*{month_text}월$"


def _normalize_locations(locations: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """[서연] 구형·신형 TOPIS 좌표 시트를 지점·유출입 방향 단위로 통일한다."""
    common = {"지점번호", "지점명칭", "위도", "경도"}
    missing_common = sorted(common - set(locations.columns))
    if missing_common:
        raise ValueError(f"{source_name} 위치 공통 열 누락: {missing_common}")

    if {"유입 방향", "유출방향"}.issubset(locations.columns):
        base = locations[["지점번호", "지점명칭", "위도", "경도", "유입 방향", "유출방향"]].copy()
        inbound = base[["지점번호", "지점명칭", "위도", "경도", "유입 방향"]].rename(
            columns={"유입 방향": "direction_detail"}
        )
        inbound["방향"] = "유입"
        outbound = base[["지점번호", "지점명칭", "위도", "경도", "유출방향"]].rename(
            columns={"유출방향": "direction_detail"}
        )
        outbound["방향"] = "유출"
        normalized = pd.concat([inbound, outbound], ignore_index=True)
    elif {"방향", "유출입 방향"}.issubset(locations.columns):
        normalized = locations[["지점번호", "방향", "지점명칭", "위도", "경도", "유출입 방향"]].copy()
        # 신형 파일은 두 번째(유출) 행의 병합 셀 때문에 지점번호가 비어 있다.
        normalized["지점번호"] = normalized["지점번호"].ffill()
        normalized = normalized.rename(columns={"유출입 방향": "direction_detail"})
    else:
        raise ValueError(
            f"{source_name} 위치 방향 열 형식을 식별하지 못했습니다: {list(locations.columns)}"
        )

    normalized["지점번호"] = normalized["지점번호"].astype(str).str.strip()
    normalized["방향"] = normalized["방향"].astype(str).str.strip()
    normalized = normalized[
        normalized["지점번호"].str.fullmatch(r"[A-Z]-\d{2}", na=False)
        & normalized["방향"].isin(["유입", "유출"])
    ].copy()
    if normalized.duplicated(["지점번호", "방향"]).any():
        raise ValueError(f"{source_name} 위치 시트에 지점번호·방향 중복이 있습니다.")
    return normalized


def _read_month_file(path: Path, month: str) -> pd.DataFrame:
    """월별 XLSX를 90일·5시간대의 긴 형식으로 변환한다."""
    workbook = pd.ExcelFile(path, engine="openpyxl")
    volume_sheet = _find_sheet(workbook.sheet_names, _month_sheet_pattern(month))
    location_sheet = _find_sheet(workbook.sheet_names, r"주소\s*및\s*좌표")

    volume = pd.read_excel(workbook, sheet_name=volume_sheet, dtype=object)
    locations = pd.read_excel(workbook, sheet_name=location_sheet, dtype=object)
    volume.columns = [str(column).strip() for column in volume.columns]
    locations.columns = [str(column).strip() for column in locations.columns]

    base_required = {"일자", "지점명", "지점번호", "방향", "구분"}
    missing = sorted(base_required - set(volume.columns))
    if missing:
        raise ValueError(f"{path.name} 교통량 열 누락: {missing}")
    hour_columns = [f"{hour}시" for hour in VOLUME_TO_TOPIS_END_HOUR]
    missing_hours = [column for column in hour_columns if column not in volume.columns]
    if missing_hours:
        raise ValueError(f"{path.name} 시간 열 누락: {missing_hours}")

    locations = _normalize_locations(locations, path.name)

    identifier_columns = ["일자", "요일", "지점명", "지점번호", "방향", "구분"]
    identifier_columns = [column for column in identifier_columns if column in volume.columns]
    long = volume.melt(
        id_vars=identifier_columns,
        value_vars=hour_columns,
        var_name="source_hour_label",
        value_name="traffic_volume",
    )
    long["date"] = pd.to_datetime(long["일자"].astype(str).str.replace(r"\.0$", "", regex=True), format="%Y%m%d", errors="raise")
    long["volume_start_hour"] = long["source_hour_label"].str.extract(r"(\d+)", expand=False).astype(int)
    long["hour"] = long["volume_start_hour"].map(VOLUME_TO_TOPIS_END_HOUR).astype(int)
    long["traffic_volume"] = pd.to_numeric(long["traffic_volume"], errors="coerce")
    long["지점번호"] = long["지점번호"].astype(str).str.strip()
    long = long.merge(locations, on=["지점번호", "방향"], how="left", validate="many_to_one")
    long["road_name"] = long["지점명"].astype(str).str.replace(r"\(.*$", "", regex=True).str.strip()
    long["commute_period"] = long["hour"].map(lambda value: "morning" if value <= 9 else "evening")
    long["source_month"] = month
    long["source_file"] = path.name
    long = long.rename(
        columns={
            "지점번호": "spot_id",
            "지점명": "spot_name",
            "방향": "io_type",
            "위도": "latitude",
            "경도": "longitude",
        }
    )
    return long


def summarize_traffic_volume_missing(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """[서연] 전처리 결과의 교통량 결측을 날짜·지점·시간대별로 요약한다."""
    required = {
        "date", "split", "spot_id", "spot_name", "io_type", "hour",
        "traffic_volume", "source_month", "source_file",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"교통량 결측 분석 열 누락: {missing_columns}")

    data = frame.copy()
    data["is_missing"] = data["traffic_volume"].isna()

    def grouped(columns: list[str]) -> pd.DataFrame:
        result = (
            data.groupby(columns, dropna=False)
            .agg(rows=("is_missing", "size"), missing_rows=("is_missing", "sum"))
            .reset_index()
        )
        result["missing_pct"] = (result["missing_rows"] / result["rows"] * 100).round(3)
        return result.sort_values(["missing_pct", "missing_rows"], ascending=False).reset_index(drop=True)

    total_rows = len(data)
    missing_rows = int(data["is_missing"].sum())
    summary = pd.DataFrame([{
        "rows": total_rows,
        "missing_rows": missing_rows,
        "missing_pct": round(missing_rows / total_rows * 100, 3) if total_rows else 0.0,
        "dates": int(data["date"].nunique()),
        "spots": int(data["spot_id"].nunique()),
    }])
    missing_sample = data.loc[
        data["is_missing"],
        ["date", "split", "spot_id", "spot_name", "io_type", "hour", "source_month", "source_file"],
    ].head(100)
    return {
        "summary": summary,
        "by_split": grouped(["split"]),
        "by_source_month": grouped(["source_month"]),
        "by_date": grouped(["date", "split"]),
        "by_hour": grouped(["hour"]),
        "by_spot": grouped(["spot_id", "spot_name", "io_type"]),
        "missing_sample": missing_sample.reset_index(drop=True),
    }


def preprocess_downloads(
    *,
    root: Path | None = None,
    run_dir: Path | None = None,
    run_id: str | None = None,
    version: str = "v01",
) -> dict[str, object]:
    """다운로드 파일에서 90일·5시간대만 추출해 train/eval Parquet로 저장한다."""
    base = (root or data_root()).resolve()
    raw_dir = base / RAW_SUBDIR
    processed_dir = base / PROCESSED_SUBDIR
    record_dir = run_dir or (base / LOG_SUBDIR)
    record_dir.mkdir(parents=True, exist_ok=True)
    effective_run_id = run_id or datetime.now(KST).strftime("%Y%m%dT%H%M%S")
    log_path = record_dir / f"yang_seoyeon_preprocess_{effective_run_id}_(중단).json"
    validation_path = record_dir / f"yang_seoyeon_validation_{effective_run_id}_(중단).csv"
    days = load_target_days()
    months = required_months(days)
    files = {
        month: raw_dir / f"yang_seoyeon_topis_traffic_volume_{month}.xlsx"
        for month in months
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("필요한 월별 파일이 없습니다:\n" + "\n".join(missing))

    frames = [_read_month_file(path, month) for month, path in files.items()]
    volume = pd.concat(frames, ignore_index=True)
    day_meta = days[["date", "split", "role", "season", "season_name"]].copy()
    selected = volume.merge(day_meta, on="date", how="inner", validate="many_to_one")
    selected = selected[OUTPUT_COLUMNS].sort_values(["date", "spot_id", "io_type", "hour"]).reset_index(drop=True)

    if selected.empty:
        raise RuntimeError("90일과 일치하는 교통량 행이 0개입니다.")
    if not set(selected["hour"].unique()).issubset({8, 9, 18, 19, 20}):
        raise ValueError("TOPIS 끝 시각 변환 결과에 예상 밖 시간이 있습니다.")
    duplicate_keys = ["date", "spot_id", "io_type", "hour"]
    if selected.duplicated(duplicate_keys).any():
        raise ValueError("날짜·지점·방향·시간 기준 중복 행이 있습니다.")

    processed_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    for split in ("train", "eval"):
        output = processed_dir / (
            f"yang_seoyeon_traffic_volume_{split}_{effective_run_id}_{version}_(성공).parquet"
        )
        if output.exists():
            raise FileExistsError(output)
        part = selected[selected["split"] == split].copy()
        part.to_parquet(output, index=False)
        output_paths[split] = str(output)

    validation = (
        selected.groupby("split", observed=True)
        .agg(
            rows=("date", "size"),
            days=("date", "nunique"),
            spots=("spot_id", "nunique"),
            traffic_volume_missing=("traffic_volume", lambda values: int(values.isna().sum())),
            coordinate_missing=("latitude", lambda values: int(values.isna().sum())),
        )
        .reset_index()
    )
    validation["duplicate_keys"] = int(selected.duplicated(duplicate_keys).sum())
    validation.to_csv(validation_path, index=False, encoding="utf-8-sig")

    summary: dict[str, object] = {
        "finished_at": _now(),
        "input_months": months,
        "target_days": int(days["date"].nunique()),
        "rows": int(len(selected)),
        "rows_by_split": selected["split"].value_counts().to_dict(),
        "unique_spots": int(selected["spot_id"].nunique()),
        "date_min": selected["date"].min().date().isoformat(),
        "date_max": selected["date"].max().date().isoformat(),
        "traffic_volume_nan": int(selected["traffic_volume"].isna().sum()),
        "coordinate_nan": int(selected[["latitude", "longitude"]].isna().any(axis=1).sum()),
        "outputs": output_paths,
        "validation_path": str(validation_path),
        "preprocess_log_path": str(log_path),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def validate_test_month(
    month: str,
    *,
    root: Path | None = None,
    run_dir: Path,
    run_id: str,
) -> dict[str, object]:
    """1개월 시험 파일의 구조와 해당 월 대상 날짜·5시간대를 검증한다."""
    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError(f"시험 연월은 YYYY-MM 형식이어야 합니다: {month}")
    base = (root or data_root()).resolve()
    path = base / RAW_SUBDIR / f"yang_seoyeon_topis_traffic_volume_{month}.xlsx"
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    frame = _read_month_file(path, month)
    days = load_target_days()
    month_days = days[days["date"].dt.strftime("%Y-%m") == month].copy()
    if month_days.empty:
        raise ValueError(f"90일 목록에 {month} 날짜가 없어 시험 월로 사용할 수 없습니다.")
    selected = frame.merge(
        month_days[["date", "split", "role", "season", "season_name"]],
        on="date",
        how="inner",
        validate="many_to_one",
    )
    duplicate_keys = ["date", "spot_id", "io_type", "hour"]
    expected_hours = {8, 9, 18, 19, 20}
    actual_hours = set(selected["hour"].dropna().astype(int).unique())
    validation = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "test_month": month,
                "source_file": path.name,
                "source_bytes": path.stat().st_size,
                "source_rows_5hours_all_days": len(frame),
                "target_days_in_month": int(month_days["date"].nunique()),
                "selected_rows": len(selected),
                "unique_spots": int(selected["spot_id"].nunique()),
                "hours": ",".join(map(str, sorted(actual_hours))),
                "expected_hours_ok": actual_hours == expected_hours,
                "traffic_volume_missing": int(selected["traffic_volume"].isna().sum()),
                "coordinate_missing": int(
                    selected[["latitude", "longitude"]].isna().any(axis=1).sum()
                ),
                "duplicate_keys": int(selected.duplicated(duplicate_keys).sum()),
            }
        ]
    )
    validation_path = run_dir / f"yang_seoyeon_test_validation_{month}_{run_id}_(중단).csv"
    sample_path = run_dir / f"yang_seoyeon_test_sample_{month}_{run_id}_(중단).csv"
    validation.to_csv(validation_path, index=False, encoding="utf-8-sig")
    selected.head(100).to_csv(sample_path, index=False, encoding="utf-8-sig")
    if selected.empty:
        raise RuntimeError("시험 월의 90일 대상 교통량이 0행입니다.")
    if actual_hours != expected_hours:
        raise ValueError(f"시험 월의 시간 변환이 다릅니다: {sorted(actual_hours)}")
    if validation.loc[0, "duplicate_keys"] != 0:
        raise ValueError("시험 월에 날짜·지점·방향·시간 중복이 있습니다.")
    return {
        "rows": int(len(selected)),
        "target_days": int(month_days["date"].nunique()),
        "unique_spots": int(selected["spot_id"].nunique()),
        "validation_path": str(validation_path),
        "sample_path": str(sample_path),
    }


def _finalize_run_dir(run_dir: Path, status: str) -> Path:
    """실행 기록 파일과 폴더를 성공·실패 상태로 확정한다."""
    if status not in {"성공", "실패"}:
        return run_dir
    for path in list(run_dir.iterdir()):
        if path.is_file() and "(중단)" in path.name:
            _rename_status(path, "중단", status)
    return _rename_status(run_dir, "중단", status)


def _rewrite_run_path_references(
    value: object,
    *,
    old_run_dir: Path,
    final_run_dir: Path,
    status: str,
) -> object:
    """요약 내부의 실행 폴더 경로를 최종 상태 경로로 바꾼다."""
    if isinstance(value, dict):
        return {
            key: _rewrite_run_path_references(
                item,
                old_run_dir=old_run_dir,
                final_run_dir=final_run_dir,
                status=status,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_run_path_references(
                item,
                old_run_dir=old_run_dir,
                final_run_dir=final_run_dir,
                status=status,
            )
            for item in value
        ]
    if isinstance(value, str) and str(old_run_dir) in value:
        return value.replace(str(old_run_dir), str(final_run_dir)).replace(
            "(중단)", f"({status})"
        )
    return value


def _write_final_summary(
    summary: dict[str, object],
    *,
    summary_path: Path,
    old_run_dir: Path,
    final_run_dir: Path,
    status: str,
) -> dict[str, object]:
    """이름 변경이 끝난 후 summary의 모든 경로를 최종 위치로 다시 기록한다."""
    finalized = _rewrite_run_path_references(
        summary,
        old_run_dir=old_run_dir,
        final_run_dir=final_run_dir,
        status=status,
    )
    if not isinstance(finalized, dict):
        raise TypeError("최종 summary는 dict여야 합니다.")
    finalized["run_dir"] = str(final_run_dir)
    final_summary_path = final_run_dir / summary_path.name.replace(
        "(중단)", f"({status})"
    )
    finalized["summary_path"] = str(final_summary_path)
    final_summary_path.write_text(
        json.dumps(finalized, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return finalized


def run_test_month(
    month: str,
    *,
    root: Path | None = None,
    headless: bool = False,
    retries: int = 3,
    wait_seconds: int = 60,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """필요 연월 중 1개월만 수집하고 기록·구조·시간 변환을 시험한다."""
    allowed_months = required_months()
    if month not in allowed_months:
        raise ValueError(f"시험 월은 현재 필요한 26개월 중 하나여야 합니다: {month}")
    base = (root or data_root()).resolve()
    run_id = datetime.now(KST).strftime("%Y%m%dT%H%M%S")
    run_dir, version = _unique_run_dir(base, run_id, label=f"test_{month}")
    config_path = run_dir / f"yang_seoyeon_run_config_{run_id}_(중단).json"
    summary_path = run_dir / f"yang_seoyeon_summary_{run_id}_(중단).json"
    config = {
        "run_id": run_id,
        "version": version,
        "run_mode": "one_month_test",
        "test_month": month,
        "status_at_start": "중단",
        "started_at": _now(),
        "entrypoint": "crawling/traffic_volume.py::run_test_month",
        "project_root": str(ROOT),
        "data_root": str(base),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "settings": {"headless": headless, "retries": retries, "wait_seconds": wait_seconds},
        "code_provenance": git_provenance(),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    download_summary: dict[str, object] = {}
    test_validation: dict[str, object] = {}
    try:
        progress(f"[1개월 시험 시작] {month}: {run_dir}")
        download_summary = download_required_months(
            root=base,
            months=[month],
            headless=headless,
            retries=retries,
            wait_seconds=wait_seconds,
            run_dir=run_dir,
            run_id=run_id,
            progress=progress,
        )
        if download_summary["failed"]:
            raise RuntimeError(f"시험 다운로드 실패: {download_summary['failed']}")
        test_validation = validate_test_month(
            month,
            root=base,
            run_dir=run_dir,
            run_id=run_id,
        )
        final_status = "성공"
        error: dict[str, str] | None = None
    except KeyboardInterrupt:
        interrupted = {
            "run_id": run_id, "version": version, "run_mode": "one_month_test",
            "test_month": month, "status": "중단", "finished_at": _now(),
            "error": {"type": "KeyboardInterrupt", "message": "사용자 또는 실행 환경에 의해 중단됨"},
            "counts": {
                "downloaded": len(download_summary.get("downloaded", [])),
                "skipped": len(download_summary.get("skipped", [])),
                "failed": len(download_summary.get("failed", [])),
                "retries": int(download_summary.get("retries", 0)),
            },
        }
        summary_path.write_text(json.dumps(interrupted, ensure_ascii=False, indent=2), encoding="utf-8")
        progress(f"[1개월 시험 중단] 기록 보존: {run_dir}")
        raise
    except Exception as exc:
        final_status = "실패"
        error = {"type": type(exc).__name__, "message": str(exc)}

    summary = {
        "run_id": run_id, "version": version, "run_mode": "one_month_test",
        "test_month": month, "status": final_status, "finished_at": _now(),
        "counts": {
            "downloaded": len(download_summary.get("downloaded", [])),
            "skipped": len(download_summary.get("skipped", [])),
            "failed": len(download_summary.get("failed", [])),
            "retries": int(download_summary.get("retries", 0)),
        },
        "rows": test_validation.get("rows"),
        "download": download_summary,
        "test_validation": test_validation,
        "error": error,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    final_dir = _finalize_run_dir(run_dir, final_status)
    summary = _write_final_summary(
        summary,
        summary_path=summary_path,
        old_run_dir=run_dir,
        final_run_dir=final_dir,
        status=final_status,
    )
    progress(f"[1개월 시험 {final_status}] {final_dir}")
    if final_status == "실패":
        raise RuntimeError(f"1개월 시험 실패 기록: {final_dir}; 원인: {error}")
    return summary


def run_pipeline(
    *,
    root: Path | None = None,
    headless: bool = False,
    retries: int = 3,
    wait_seconds: int = 60,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """사용자가 명시적으로 호출할 때만 Selenium 수집과 전처리를 실행한다."""
    base = (root or data_root()).resolve()
    run_id = datetime.now(KST).strftime("%Y%m%dT%H%M%S")
    run_dir, version = _unique_run_dir(base, run_id)
    config_path = run_dir / f"yang_seoyeon_run_config_{run_id}_(중단).json"
    summary_path = run_dir / f"yang_seoyeon_summary_{run_id}_(중단).json"
    months = required_months()
    config = {
        "run_id": run_id,
        "version": version,
        "status_at_start": "중단",
        "started_at": _now(),
        "entrypoint": "crawling/traffic_volume.py::run_pipeline",
        "project_root": str(ROOT),
        "data_root": str(base),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "settings": {
            "headless": headless,
            "retries": retries,
            "wait_seconds": wait_seconds,
            "target_days": 90,
            "target_months": months,
            "topis_end_hours": [8, 9, 18, 19, 20],
        },
        "code_provenance": git_provenance(),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    download_summary: dict[str, object] = {}
    preprocess_summary: dict[str, object] = {}
    try:
        progress(f"[실행 시작] {run_dir}")
        download_summary = download_required_months(
            root=base,
            months=months,
            headless=headless,
            retries=retries,
            wait_seconds=wait_seconds,
            run_dir=run_dir,
            run_id=run_id,
            progress=progress,
        )
        if download_summary["failed"]:
            raise RuntimeError(f"다운로드 실패 월: {download_summary['failed']}")
        preprocess_summary = preprocess_downloads(
            root=base,
            run_dir=run_dir,
            run_id=run_id,
            version=version,
        )
        final_status = "성공"
        error: dict[str, str] | None = None
    except KeyboardInterrupt:
        interrupted = {
            "run_id": run_id,
            "version": version,
            "status": "중단",
            "finished_at": _now(),
            "error": {"type": "KeyboardInterrupt", "message": "사용자 또는 실행 환경에 의해 중단됨"},
            "counts": {
                "downloaded": len(download_summary.get("downloaded", [])),
                "skipped": len(download_summary.get("skipped", [])),
                "failed": len(download_summary.get("failed", [])),
                "retries": int(download_summary.get("retries", 0)),
            },
        }
        summary_path.write_text(json.dumps(interrupted, ensure_ascii=False, indent=2), encoding="utf-8")
        progress(f"[실행 중단] 기록 보존: {run_dir}")
        raise
    except Exception as exc:
        final_status = "실패"
        error = {"type": type(exc).__name__, "message": str(exc)}

    summary = {
        "run_id": run_id,
        "version": version,
        "status": final_status,
        "finished_at": _now(),
        "counts": {
            "downloaded": len(download_summary.get("downloaded", [])),
            "skipped": len(download_summary.get("skipped", [])),
            "failed": len(download_summary.get("failed", [])),
            "retries": int(download_summary.get("retries", 0)),
        },
        "rows": preprocess_summary.get("rows"),
        "download": download_summary,
        "preprocess": preprocess_summary,
        "error": error,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    final_dir = _finalize_run_dir(run_dir, final_status)
    summary = _write_final_summary(
        summary,
        summary_path=summary_path,
        old_run_dir=run_dir,
        final_run_dir=final_dir,
        status=final_status,
    )
    progress(f"[실행 {final_status}] {final_dir}")
    if final_status == "실패":
        raise RuntimeError(f"실행 실패 기록: {final_dir}; 원인: {error}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TOPIS 월별 교통량 Selenium 수집·전처리")
    parser.add_argument(
        "command", choices=("months", "download", "preprocess", "test", "run", "history")
    )
    parser.add_argument("--data-root", type=Path, default=None, help="기본값: SEOUL_TRAFFIC_DATA_ROOT 또는 D 드라이브")
    parser.add_argument("--headless", action="store_true", help="다운로드 Chrome을 화면에 표시하지 않음")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--wait-seconds", type=int, default=60)
    parser.add_argument("--month", help="1개월 시험 연월(YYYY-MM), test 명령에서 필수")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve() if args.data_root else data_root()
    if args.command == "months":
        days = load_target_days()
        print(f"대상 날짜: {len(days)}일")
        print(f"필요 연월: {len(required_months(days))}개")
        print("\n".join(required_months(days)))
        return
    if args.command == "history":
        history = list_run_history(root)
        print(history.to_string(index=False) if not history.empty else "실행 기록이 없습니다.")
        return
    if args.command == "test":
        if not args.month:
            raise ValueError("test 명령에는 --month YYYY-MM이 필요합니다.")
        summary = run_test_month(
            args.month,
            root=root,
            headless=args.headless,
            retries=args.retries,
            wait_seconds=args.wait_seconds,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.command == "run":
        summary = run_pipeline(
            root=root,
            headless=args.headless,
            retries=args.retries,
            wait_seconds=args.wait_seconds,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.command == "download":
        summary = download_required_months(
            root=root,
            headless=args.headless,
            retries=args.retries,
            wait_seconds=args.wait_seconds,
        )
    else:
        summary = preprocess_downloads(root=root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
