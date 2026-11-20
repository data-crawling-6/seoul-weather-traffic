"""
기상특보 동적 크롤링 (B 담당 · Selenium)

기상자료개방포털 > 기상예보 > 기상특보 화면에서
  기간 입력 → 지역 트리 펼치기 · '서울·인천·경기도' 체크 → 특보 트리 펼치기 · 대설 · 호우 · 한파 체크
  → 출력 개수 100개 선택 → [조회] → 결과 표가 새로 뜰 때까지 기다리기 → 페이지 넘기기
순서로 목록을 모으고, 각 통보문의 '상세 보기' 내용(발효시각 · 해당지역)을 받는다.
로그인은 필요 없다.

결과
  data/raw/kma_warning/warning_list.csv    목록 (발표시각 · 지역 · 제목 · 상세 키)
  data/raw/kma_warning/warning_detail.csv  상세 (발효시각 · 해당지역 원문)
  logs/kma_warning.log                     실행 기록
  logs/screenshots/*.png                   조회 전 · 후 · 상세 화면 캡처 (PPT 증거용)

실행:  python crawling/kma_warning.py
       python crawling/kma_warning.py --start 2022-10-01 --end 2026-09-30 --headless
이미 받은 상세는 건너뛰므로 중간에 멈춰도 다시 실행하면 이어서 받는다.
"""
import argparse
import logging
import math
import re
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "kma_warning"
LOG_PATH = ROOT / "logs" / "kma_warning.log"
SHOT_DIR = ROOT / "logs" / "screenshots"
URL = "https://data.kma.go.kr/data/weatherReport/wsrList.do?pgmNo=647"
REGION = "서울·인천·경기도"
KINDS = ["대설", "호우", "한파"]
PAGE_SIZE = "100"
DETAIL_JS = """
const [fcTma, no, inpTma, stn, done] = arguments;
fetch('/data/weatherReport/detailLayerPopup.do', {
  method: 'POST',
  headers: {'Content-Type': 'application/x-www-form-urlencoded'},
  body: `code=WRN&fcTma=${fcTma}&mnbyFcNo=${no}&inpTma=${inpTma}&stnId=${stn}`
}).then(r => r.text()).then(html => {
  const d = document.createElement('div'); d.innerHTML = html;
  done(d.innerText);
}).catch(e => done('ERROR ' + e));
"""
ONCLICK_RE = re.compile(r"openDetail\('([^']*)','([^']*)',\s*'([^']*)',\s*'([^']*)'\)")


def make_driver(headless: bool):
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1400,1000")
    driver = webdriver.Chrome(options=opts)  # Selenium 4가 크롬 드라이버를 자동으로 맞춘다
    driver.set_script_timeout(30)
    return driver


def shot(driver, name):
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    driver.save_screenshot(str(SHOT_DIR / f"kma_warning_{name}.png"))


# 이 사이트의 자바스크립트가 Selenium의 '보이는지 검사' 도구(is_displayed)와 충돌한다
# (computedStyleCache.set is not a function). 그래서 element_to_be_clickable 대신
# presence(요소가 생겼는지)로 기다리고, 속성은 get_dom_attribute / get_property 로 읽는다.
def check_tree(driver, wait, tree_id, switch_id, titles):
    """트리를 펼치고, 제목이 titles 로 시작하는 체크박스를 누른다."""
    sw = wait.until(EC.presence_of_element_located((By.ID, switch_id)))
    if "close" in (sw.get_dom_attribute("class") or ""):
        sw.click()
        time.sleep(0.5)
    for t in titles:
        box = wait.until(EC.presence_of_element_located(
            (By.XPATH, f"//ul[@id='{tree_id}']//a[contains(@id,'_check') and starts-with(@title,'{t}')]")))
        if "true" not in (box.get_dom_attribute("class") or ""):
            box.click()
            time.sleep(0.2)
        logging.info("체크: %s", t)


def set_conditions(driver, wait, start, end):
    driver.get(URL)
    wait.until(EC.presence_of_element_located((By.ID, "schForm")))
    shot(driver, "1_before")
    # 날짜: 달력 입력칸이라 값을 직접 넣는다
    for fid, val in (("startDt", start), ("endDt", end)):
        el = driver.find_element(By.ID, fid)
        driver.execute_script("arguments[0].value = arguments[1];", el, val.replace("-", ""))
    check_tree(driver, wait, "ztree1", "ztree1_2_switch", [REGION])
    check_tree(driver, wait, "ztree", "ztree_2_switch", KINDS)
    Select(driver.find_element(By.ID, "selectSchListCnt")).select_by_value(PAGE_SIZE)
    shot(driver, "2_conditions")


def click_search(driver, wait):
    old = driver.find_element(By.ID, "schForm")
    driver.find_element(By.XPATH, "//button[contains(@onclick,'searchData')]").click()
    wait.until(EC.staleness_of(old))  # 폼 제출로 화면이 새로 그려질 때까지
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr a.btn_more")))
    shot(driver, "3_result")


def read_rows(driver):
    rows = []
    for tr in driver.find_elements(By.CSS_SELECTOR, "table tbody tr"):
        links = tr.find_elements(By.CSS_SELECTOR, "a.btn_more")
        if not links:
            continue
        tds = tr.find_elements(By.TAG_NAME, "td")
        m = ONCLICK_RE.search(links[0].get_dom_attribute("onclick"))
        rows.append({
            "issued_at": tds[0].text.strip(), "region": tds[1].text.strip(), "title": tds[2].text.strip(),
            "fcTma": m.group(1), "no": m.group(2), "inpTma": m.group(3), "stn": m.group(4),
        })
    return rows


def go_page(driver, wait, n):
    first = driver.find_element(By.CSS_SELECTOR, "table tbody tr")
    links = [a for a in driver.find_elements(By.CSS_SELECTOR, f"a[onclick^='changePage({n});']")
             if a.text.replace("\n", "").strip() == f"{n}페이지"]
    if links:
        links[0].click()
    else:  # 11쪽부터는 페이지 묶음이 바뀌므로 화면의 changePage 함수를 직접 부른다
        driver.execute_script(f"changePage({n});")
    wait.until(EC.staleness_of(first))
    wait.until(lambda d: d.find_element(By.ID, "pageIndex").get_property("value") == str(n))


def crawl_list(driver, wait):
    total = int(driver.find_element(By.ID, "totCnt").get_property("value"))
    pages = math.ceil(total / int(PAGE_SIZE))
    logging.info("검색 결과 %d건, %d쪽", total, pages)
    rows = read_rows(driver)
    for n in range(2, pages + 1):
        for attempt in range(3):
            try:
                go_page(driver, wait, n)
                break
            except TimeoutException:
                logging.warning("%d쪽 이동 실패, 다시 시도 %d/3", n, attempt + 1)
        rows += read_rows(driver)
        logging.info("%d/%d쪽 · 누적 %d건", n, pages, len(rows))
    if len(rows) != total:
        logging.warning("목록 수 %d건이 검색 결과 %d건과 다름", len(rows), total)
    return pd.DataFrame(rows).drop_duplicates(["inpTma", "no", "stn"])


def crawl_details(driver, lst):
    path = RAW_DIR / "warning_detail.csv"
    done = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=["inpTma", "no", "stn", "text"])
    have = set(zip(done["inpTma"], done["no"], done["stn"]))
    new = []
    # 첫 건은 실제로 '상세 보기'를 눌러 팝업 화면을 남긴다 (PPT 증거)
    try:
        driver.find_element(By.CSS_SELECTOR, "table tbody tr a.btn_more").click()
        time.sleep(1.5)
        shot(driver, "4_detail_popup")
        driver.execute_script("if (window.LayerUtil) LayerUtil.hide();")
    except Exception as e:  # noqa: BLE001
        logging.warning("상세 팝업 캡처 실패: %s", e)

    todo = lst[[k not in have for k in zip(lst["inpTma"], lst["no"], lst["stn"])]]
    logging.info("상세 받을 것 %d건 (이미 있음 %d건)", len(todo), len(lst) - len(todo))
    for i, r in enumerate(todo.itertuples(), 1):
        for attempt in range(3):
            text = driver.execute_async_script(DETAIL_JS, r.fcTma, r.no, r.inpTma, r.stn)
            if text and not text.startswith("ERROR") and "발효시각" in text:
                break
            logging.warning("상세 %s 실패, 다시 시도 %d/3", r.inpTma, attempt + 1)
            time.sleep(2)
        new.append({"inpTma": r.inpTma, "no": r.no, "stn": r.stn, "text": text})
        if i % 100 == 0:  # 100건마다 저장 (중간에 멈춰도 이어받기)
            pd.concat([done, pd.DataFrame(new)]).to_csv(path, index=False, encoding="utf-8-sig")
            logging.info("상세 %d/%d건", i, len(todo))
        time.sleep(0.2)  # 서버 부담 줄이기
    out = pd.concat([done, pd.DataFrame(new)], ignore_index=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-10-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--headless", action="store_true", help="브라우저 창을 띄우지 않고 실행")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()])
    logging.info("특보 수집 시작 %s ~ %s · %s · %s", args.start, args.end, REGION, ",".join(KINDS))

    driver = make_driver(args.headless)
    wait = WebDriverWait(driver, 20)
    try:
        set_conditions(driver, wait, args.start, args.end)
        click_search(driver, wait)
        lst = crawl_list(driver, wait)
        lst.to_csv(RAW_DIR / "warning_list.csv", index=False, encoding="utf-8-sig")
        logging.info("목록 저장 %d건", len(lst))
        det = crawl_details(driver, lst)
        logging.info("상세 저장 %d건 · 끝", len(det))
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
