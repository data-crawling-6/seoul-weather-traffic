"""양서연 담당 표준링크 다운로드와 링크 Vertex 수집.

표준링크 매핑정보는 서울 열린데이터광장 화면을 Selenium으로
조작해 다운로드한다. 링크 Vertex는 공식 ``LinkVerInfo`` XML
API로만 수집한다. import만으로는 네트워크, 브라우저, 파일 저장이
발생하지 않는다.
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote

import pandas as pd
import requests


STANDARD_LINK_URL = "https://data.seoul.go.kr/dataList/OA-15061/S/1/datasetView.do"
VERTEX_SERVICE = "LinkVerInfo"
VERTEX_COLUMNS = ["link_id", "vertex_order", "grs80tm_x", "grs80tm_y"]


def _selenium() -> dict[str, Any]:
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError as exc:
        raise RuntimeError("selenium이 없습니다. 임의로 설치하지 마세요.") from exc
    return {"webdriver": webdriver, "By": By, "WebDriverWait": WebDriverWait}


def build_download_driver(download_dir: Path, *, headless: bool = False) -> Any:
    """로그인된 개인 프로필을 쓰지 않는 다운로드 전용 Chrome을 만든다."""
    selenium = _selenium()
    target = Path(download_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    options = selenium["webdriver"].ChromeOptions()
    options.add_argument("--lang=ko-KR")
    options.add_argument("--window-size=1440,1000")
    options.add_argument("--disable-notifications")
    options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": str(target),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        },
    )
    if headless:
        options.add_argument("--headless=new")
    return selenium["webdriver"].Chrome(options=options)


def _wait_for_download(download_dir: Path, before: set[Path], wait_seconds: int) -> Path:
    deadline = time.monotonic() + wait_seconds
    directory = Path(download_dir)
    while time.monotonic() < deadline:
        current = set(directory.glob("*"))
        completed = [
            path
            for path in current - before
            if path.is_file() and path.suffix.lower() not in {".crdownload", ".tmp"}
        ]
        if completed and not list(directory.glob("*.crdownload")):
            return max(completed, key=lambda path: path.stat().st_mtime)
        time.sleep(0.5)
    raise TimeoutError("제한 시간 안에 다운로드 완료를 확인하지 못했습니다.")


def download_standard_link_mapping(
    download_dir: Path,
    *,
    headless: bool = False,
    wait_seconds: int = 30,
    progress: Callable[[str], None] = print,
) -> Path:
    """공식 데이터셋 화면에서 표준링크 매핑 파일을 내려받는다.

    화면 구조가 달라지면 선택자 실패로 중단하며 API로 자동 대체하지
    않는다. 예외가 발생해도 driver는 항상 종료한다.
    """
    selenium = _selenium()
    By = selenium["By"]
    WebDriverWait = selenium["WebDriverWait"]
    target = Path(download_dir).resolve()
    before = set(target.glob("*")) if target.exists() else set()
    driver = None
    try:
        progress("[표준링크] 다운로드 전용 Chrome 시작")
        driver = build_download_driver(target, headless=headless)
        driver.get(STANDARD_LINK_URL)

        def find_download(current: Any) -> Any | None:
            for element in current.find_elements(By.CSS_SELECTOR, "a, button"):
                if not element.is_displayed() or not element.is_enabled():
                    continue
                label = " ".join(
                    filter(
                        None,
                        [
                            (element.text or "").strip(),
                            element.get_attribute("title") or "",
                            element.get_attribute("aria-label") or "",
                            element.get_attribute("href") or "",
                        ],
                    )
                ).lower()
                if ("xlsx" in label or "엑셀" in label) and (
                    "다운로드" in label or "download" in label or ".xlsx" in label
                ):
                    return element
            return None

        element = WebDriverWait(driver, wait_seconds).until(find_download)
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
        driver.execute_script("arguments[0].click();", element)
        downloaded = _wait_for_download(target, before, wait_seconds)
        progress(f"[표준링크] 완료: {downloaded.name}")
        return downloaded
    finally:
        if driver is not None:
            driver.quit()


def get_seoul_api_key(variable: str = "SEOUL_OPEN_DATA_API_KEY") -> str:
    """환경변수에서 API 키를 읽되 값을 출력하지 않는다."""
    key = os.getenv(variable, "").strip()
    if not key:
        raise RuntimeError(f"환경변수 {variable}가 설정되지 않았습니다.")
    return key


def _xml_text(row: ET.Element, name: str) -> str:
    element = row.find(name)
    return "" if element is None or element.text is None else element.text.strip()


def fetch_link_vertices(
    link_ids: Iterable[str],
    *,
    api_key: str | None = None,
    request_timeout: int = 20,
    delay_seconds: float = 0.1,
    progress: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``LinkVerInfo`` XML API로 링크별 Vertex와 실패 로그를 반환한다."""
    key = api_key or get_seoul_api_key()
    ids = list(dict.fromkeys(str(value).strip() for value in link_ids if str(value).strip()))
    if not ids:
        raise ValueError("수집할 link_id가 없습니다.")

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    session = requests.Session()
    try:
        for index, link_id in enumerate(ids, start=1):
            url = (
                "http://openapi.seoul.go.kr:8088/"
                f"{quote(key, safe='')}/xml/{VERTEX_SERVICE}/1/1000/"
                f"{quote(link_id, safe='')}"
            )
            try:
                response = session.get(url, timeout=request_timeout)
                response.raise_for_status()
                root = ET.fromstring(response.content)
                result_code = root.findtext(".//RESULT/CODE", default="").strip()
                result_message = root.findtext(".//RESULT/MESSAGE", default="").strip()
                if result_code and result_code != "INFO-000":
                    raise RuntimeError(f"{result_code}: {result_message}")
                api_rows = root.findall(".//row")
                if not api_rows:
                    raise RuntimeError("Vertex 행이 없습니다.")
                for row in api_rows:
                    rows.append(
                        {
                            "link_id": _xml_text(row, "LINK_ID") or link_id,
                            "vertex_order": pd.to_numeric(
                                _xml_text(row, "VER_SEQ"), errors="coerce"
                            ),
                            "grs80tm_x": pd.to_numeric(
                                _xml_text(row, "GRS80TM_X"), errors="coerce"
                            ),
                            "grs80tm_y": pd.to_numeric(
                                _xml_text(row, "GRS80TM_Y"), errors="coerce"
                            ),
                        }
                    )
            except Exception as exc:
                failures.append(
                    {
                        "link_id": link_id,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
            progress(f"[Vertex API] {index}/{len(ids)} {link_id}")
            if delay_seconds > 0:
                time.sleep(delay_seconds)
    finally:
        session.close()

    return (
        pd.DataFrame(rows, columns=VERTEX_COLUMNS),
        pd.DataFrame(failures, columns=["link_id", "error_type", "error_message"]),
    )


def vertices_to_lines(vertices: pd.DataFrame, *, source_epsg: int) -> Any:
    """Vertex를 순서대로 연결해 WGS84 도로 LineString을 만든다.

    서울 API의 좌표계를 출처 문서에서 확인한 후 ``source_epsg``로
    명시해야 한다. 코드가 EPSG를 추정하지 않는다.
    """
    required = set(VERTEX_COLUMNS)
    missing = sorted(required - set(vertices.columns))
    if missing:
        raise ValueError(f"필수 열이 없습니다: {missing}")
    try:
        import geopandas as gpd
        from pyproj import Transformer
        from shapely.geometry import LineString
    except ImportError as exc:
        raise RuntimeError("geopandas, pyproj, shapely가 필요합니다.") from exc

    valid = vertices.dropna(
        subset=["link_id", "vertex_order", "grs80tm_x", "grs80tm_y"]
    ).copy()
    transformer = Transformer.from_crs(source_epsg, 4326, always_xy=True)
    records: list[dict[str, object]] = []
    for link_id, group in valid.groupby("link_id", sort=False):
        ordered = group.sort_values("vertex_order")
        if len(ordered) < 2:
            continue
        lon, lat = transformer.transform(
            ordered["grs80tm_x"].to_numpy(), ordered["grs80tm_y"].to_numpy()
        )
        records.append(
            {
                "link_id": str(link_id),
                "vertex_count": int(len(ordered)),
                "geometry": LineString(zip(lon, lat)),
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
