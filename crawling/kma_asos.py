"""
ASOS 서울(108) 시간자료 수집 (B 담당)

공공데이터포털 '기상청_지상(종관, ASOS) 시간자료 조회서비스' API로
월 단위로 받아서 data/raw/asos/asos_108_YYYYMM.csv 로 저장한다.

- 이미 받은 달은 건너뛴다 (이어받기)
- 실패하면 최대 3번 다시 시도한다 (재시도)
- 받은 행 수가 그 달의 시간 수와 다르면 로그에 경고를 남긴다 (파일 검증)
- 실행 기록은 logs/kma_asos.log 에 남긴다 (발표 증거용)

실행:  python crawling/kma_asos.py
       python crawling/kma_asos.py --start 2022-11 --end 2026-08

인증키: 프로젝트 맨 위 api_keys.txt 에  DATA_GO_KR_KEY=발급받은키  한 줄을 넣는다.
        (공공데이터포털의 '일반 인증키(Decoding)' 값을 쓴다. 이 파일은 깃에 올라가지 않는다.)

API로 받기 어려우면 기상자료개방포털에서 받은 CSV를 data/raw/asos_manual/ 에 넣어도 된다.
분석 코드(src/weather.py)가 두 형식을 모두 읽는다.
"""
import argparse
import calendar
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "asos"
LOG_PATH = ROOT / "logs" / "kma_asos.log"
URL = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
STATION = "108"  # 서울(종로구 송월동)


def load_key(name="DATA_GO_KR_KEY"):
    """api_keys.txt 에서 KEY=VALUE 형식으로 키를 읽는다."""
    path = ROOT / "api_keys.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. api_keys.example.txt 를 복사해서 키를 넣으세요.")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == name and v.strip():
            return v.strip()
    raise KeyError(f"api_keys.txt 에 {name}= 값이 없습니다.")


def month_range(start, end):
    """'2022-11' ~ '2026-08' 사이의 (연, 월) 목록"""
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def fetch_month(key, year, month, retries=3):
    last_day = calendar.monthrange(year, month)[1]
    params = {
        "serviceKey": key,
        "pageNo": 1,
        "numOfRows": 999,  # 한 달은 최대 744시간이라 한 페이지로 받는다
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "HR",
        "startDt": f"{year}{month:02d}01",
        "startHh": "00",
        "endDt": f"{year}{month:02d}{last_day:02d}",
        "endHh": "23",
        "stnIds": STATION,
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(URL, params=params, timeout=30)
            r.raise_for_status()
            body = r.json()["response"]
            code = body["header"]["resultCode"]
            if code != "00":
                raise RuntimeError(f"API 오류 {code}: {body['header'].get('resultMsg')}")
            items = body["body"]["items"]["item"]
            return pd.DataFrame(items), last_day * 24
        except Exception as e:  # noqa: BLE001
            msg = str(e).replace(key, "***")  # 로그에 인증키가 남지 않게 가린다
            if "403" in msg:
                msg += " → 이 API의 활용신청이 안 됐거나 아직 승인 전일 수 있음"
            logging.warning("%d-%02d 시도 %d/%d 실패: %s", year, month, attempt, retries, msg)
            time.sleep(2 * attempt)
    raise RuntimeError(f"{year}-{month:02d} 수집 실패")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-10")   # 학습 첫 겨울 시작 전 여유 포함
    ap.add_argument("--end", default="2026-09")     # 평가 여름 끝까지
    ap.add_argument("--force", action="store_true", help="이미 받은 달도 다시 받기")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )
    key = load_key()
    logging.info("수집 시작 %s ~ %s", args.start, args.end)

    done = skipped = 0
    for y, m in month_range(args.start, args.end):
        out = RAW_DIR / f"asos_{STATION}_{y}{m:02d}.csv"
        if out.exists() and not args.force:
            skipped += 1
            continue
        df, expected = fetch_month(key, y, m)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        if len(df) != expected:
            logging.warning("%s: %d행 (예상 %d행) · 빠진 시간이 있음", out.name, len(df), expected)
        else:
            logging.info("%s: %d행 저장", out.name, len(df))
        done += 1
        time.sleep(0.5)  # 서버 부담 줄이기
    logging.info("수집 끝: 새로 받음 %d개월, 건너뜀 %d개월 (%s)", done, skipped, datetime.now())


if __name__ == "__main__":
    main()
