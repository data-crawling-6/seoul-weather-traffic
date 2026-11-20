"""
구별 AWS 시간자료 · 과거 단기예보 수집 (B 담당 · 기상청 API허브)

악천후일 · 비교일 90일(data/processed/{train,eval}/weather_days_*.csv)에 대해
  1) 방재기상관측(AWS) 정시자료  : 서울 29개 지점, 하루당 36시간 (전날 13시 ~ 그날 24시)
  2) 단기예보(getVilageFcst)     : 서울 4개 권역 격자, 하루당 9번 발표 (전날 23시 + 그날 02~23시)
를 받는다. 시각은 모두 '끝 시각' 기준이다 (13시 = 12:00~13:00, TOPIS "~13시"와 같은 칸).

나눠 받기 (하루 한도 20,000건 안에서 여유 있게):
  python crawling/kma_apihub.py --test     # 2025-12-04 하루만 (응답 형식 · 속도 확인, 약 5분)
  python crawling/kma_apihub.py --part 1   # 평가 30일 (2025~26 겨울 · 2026 여름)
  python crawling/kma_apihub.py --part 2   # 학습 겨울 (2022~23 · 2023~24 · 2024~25)
  python crawling/kma_apihub.py --part 3   # 학습 여름 (2023 · 2024 · 2025)
  python crawling/kma_apihub.py --part 1 --dry-run   # 호출 수만 세고 받지 않음
  python crawling/kma_apihub.py --parse    # 받은 원본을 표로 정리만 다시

- 이미 받은 시각 · 발표는 건너뛴다 (이어받기). 같은 명령을 다시 치면 이어서 받는다.
- 실패하면 3번까지 다시 시도한다. 인증 오류면 바로 멈추고 활용신청 확인을 안내한다.
- 오늘 호출 수를 logs/kma_apihub_calls.json 에 세서, 하루 19,000건이 되면 멈춘다.
- 로그(logs/kma_apihub.log)와 오류 메시지에서 인증키는 *** 로 가린다.

인증키: api_keys.txt 에  KMA_APIHUB_KEY=발급받은키  (기상청 API허브 마이페이지의 authKey)
활용신청: 지상관측 > 방재기상관측(AWS) > 3.1.6 정시자료(awsh.php),
          예특보 > 단기예보 > 4.3 단기예보조회(getVilageFcst)

결과
  data/raw/kma_aws/aws_YYYYMMDDHH.txt                 AWS 원본 (서울 지점만 남김)
  data/raw/kma_fcst/fcst_YYYYMMDDHH_권역.json          단기예보 원본
  data/processed/{train,eval}/aws_hourly_{split}.csv   지점 × 시각 표 (datetime, stn, 관측값들)
  data/processed/{train,eval}/fcst_{split}.csv         권역 × 발표시각 × 예보시각 × 항목 표
"""
import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
AWS_DIR = ROOT / "data" / "raw" / "kma_aws"
FCST_DIR = ROOT / "data" / "raw" / "kma_fcst"
PROC_DIR = ROOT / "data" / "processed"
LOG_PATH = ROOT / "logs" / "kma_apihub.log"
CALLS_PATH = ROOT / "logs" / "kma_apihub_calls.json"

AWS_URL = "https://apihub.kma.go.kr/api/typ01/url/awsh.php"
FCST_URL = "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst"

# 서울 AWS 지점 (API허브 예시의 서울 지점 묶음). 108 = ASOS 서울(종로)
SEOUL_STNS = [108, 110, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412,
              413, 414, 415, 416, 417, 418, 419, 421, 423, 424, 425, 509, 510, 889]

# 서울 특보 4개 권역의 대표 동네예보 격자 (nx, ny). 서북권은 ASOS 관측소가 있는 종로
REGION_GRIDS = {
    "서북권": (60, 127),  # 종로구
    "동북권": (61, 129),  # 노원구
    "동남권": (61, 126),  # 강남구
    "서남권": (58, 126),  # 강서구
}
BASE_HOURS = [2, 5, 8, 11, 14, 17, 20, 23]  # 단기예보 발표 시각

DAILY_LIMIT = 19000   # API허브 일반회원 하루 20,000건 → 여유를 둔다
PAUSE_SEC = 0.2
RETRIES = 3
TEST_DAY = "2025-12-04"
log = logging.getLogger("kma_apihub")


class AuthError(RuntimeError):
    pass


class LimitReached(RuntimeError):
    pass


# ---------------------------------------------------------------- 준비
def setup_log():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def load_key(name="KMA_APIHUB_KEY"):
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


def load_days(part):
    """part: test / 1(평가) / 2(학습 겨울) / 3(학습 여름) → [(날짜, split)]"""
    if part == "test":
        return [(pd.Timestamp(TEST_DAY), "eval")]
    frames = []
    for split in ("train", "eval"):
        d = pd.read_csv(PROC_DIR / split / f"weather_days_{split}.csv", encoding="utf-8-sig")
        d["split"] = split
        frames.append(d)
    d = pd.concat(frames)
    if part == "1":
        d = d[d["split"] == "eval"]
    elif part == "2":
        d = d[(d["split"] == "train") & d["season"].str.startswith("W")]
    elif part == "3":
        d = d[(d["split"] == "train") & d["season"].str.startswith("S")]
    else:
        raise ValueError(f"--part 는 1, 2, 3 중 하나: {part}")
    return [(pd.Timestamp(x), s) for x, s in sorted(set(zip(d["date"], d["split"])))]


def aws_times(day):
    """그날을 보려면 필요한 AWS 시각 (끝 시각): 전날 13시 ~ 그날 24시(= 다음 날 00시), 36개"""
    start = day - timedelta(hours=11)          # 전날 13:00
    return [start + timedelta(hours=i) for i in range(36)]


def fcst_bases(day):
    """그날을 보려면 필요한 단기예보 발표: 전날 23시 + 그날 02~23시, 9번"""
    return [day - timedelta(hours=1)] + [day + timedelta(hours=h) for h in BASE_HOURS]


# ---------------------------------------------------------------- 호출 수 관리
def load_calls():
    today = datetime.now().strftime("%Y-%m-%d")
    data = json.loads(CALLS_PATH.read_text(encoding="utf-8")) if CALLS_PATH.exists() else {}
    return data, today


def add_call(calls, today):
    calls[today] = calls.get(today, 0) + 1
    if calls[today] % 100 == 0:
        CALLS_PATH.write_text(json.dumps(calls, ensure_ascii=False, indent=1), encoding="utf-8")
    if calls[today] >= DAILY_LIMIT:
        raise LimitReached(f"오늘 {calls[today]:,}건 호출. 하루 한도 근처라 멈춥니다. 내일 같은 명령으로 이어 받으세요.")


def get(session, url, params, key, calls, today):
    """요청 1번 (재시도 포함). 인증키는 오류 메시지에서 가린다."""
    params = dict(params, authKey=key)
    last = None
    for attempt in range(1, RETRIES + 1):
        add_call(calls, today)
        try:
            r = session.get(url, params=params, timeout=(10, 60))
            text = r.text
            if r.status_code in (401, 403) or "인증" in text[:300] or "not authorized" in text[:300].lower():
                raise AuthError(f"HTTP {r.status_code}: {text[:200]}")
            r.raise_for_status()
            time.sleep(PAUSE_SEC)
            return text
        except AuthError:
            raise
        except requests.RequestException as e:
            last = str(e).replace(key, "***")
            log.warning("  재시도 %d/%d: %s", attempt, RETRIES, last[:200])
            time.sleep(3 * attempt)
    raise RuntimeError(f"{RETRIES}번 시도 후 실패: {last}")


# ---------------------------------------------------------------- AWS
def find_header(text):
    """머리말(#) 중 열 이름 줄을 찾는다. STN이 들어 있고 설명(':')이 없는 마지막 줄."""
    header = None
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("#"):
            continue
        toks = s.lstrip("#").split()
        if toks and any(t.upper() == "STN" for t in toks) and not any(t.endswith(":") or t == ":" for t in toks):
            header = toks
    return header


def split_row(line):
    return [t for t in re.split(r"[\s,]+", line.strip().strip(",=")) if t and t != "="]


def parse_aws_text(text):
    """awsh.php 응답(등간격 텍스트) → DataFrame. 열 이름을 못 찾으면 None."""
    header = find_header(text)
    if header is None:
        return None
    rows = [split_row(l)[:len(header)] for l in text.splitlines()
            if l.strip() and not l.strip().startswith("#") and len(split_row(l)) >= len(header)]
    df = pd.DataFrame(rows, columns=header)
    stn_col = next(c for c in header if c.upper() == "STN")
    df = df.rename(columns={stn_col: "STN"})
    df["STN"] = pd.to_numeric(df["STN"], errors="coerce")
    return df


def keep_seoul(text):
    """서울 지점 줄과 머리말만 남긴 텍스트. 열 이름을 못 찾으면 원본 그대로."""
    header = find_header(text)
    if header is None:
        return text, False
    idx = [t.upper() for t in header].index("STN")
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            out.append(line)
            continue
        toks = split_row(s)
        if len(toks) > idx and toks[idx].isdigit() and int(toks[idx]) in SEOUL_STNS:
            out.append(line)
    return "\n".join(out) + "\n", True


def fetch_aws(session, key, t, calls, today):
    path = AWS_DIR / f"aws_{t:%Y%m%d%H}.txt"
    if path.exists() and path.stat().st_size > 0:
        return "skip"
    text = get(session, AWS_URL, {"tm": t.strftime("%Y%m%d%H%M"), "stn": 0, "help": 1}, key, calls, today)
    small, ok = keep_seoul(text)
    path.write_text(small, encoding="utf-8")
    if not ok:
        log.warning("  [AWS] %s: 응답에서 열 이름(STN) 줄을 못 찾아 원본을 그대로 저장했습니다 → %s", t, path.name)
    return "ok"


# ---------------------------------------------------------------- 단기예보
def fetch_fcst(session, key, base, region, calls, today):
    nx, ny = REGION_GRIDS[region]
    path = FCST_DIR / f"fcst_{base:%Y%m%d%H}_{region}.json"
    if path.exists() and path.stat().st_size > 0:
        return "skip"
    items, page, total = [], 1, None
    while True:
        params = {"pageNo": page, "numOfRows": 1000, "dataType": "JSON",
                  "base_date": base.strftime("%Y%m%d"), "base_time": base.strftime("%H00"),
                  "nx": nx, "ny": ny}
        text = get(session, FCST_URL, params, key, calls, today)
        try:
            body = json.loads(text)["response"]
        except (ValueError, KeyError):
            raise RuntimeError(f"단기예보 응답이 JSON이 아닙니다: {text[:200]}")
        code = body["header"]["resultCode"]
        if code == "03":  # NO_DATA
            log.warning("  [예보] %s %s: 자료 없음(NO_DATA)", base, region)
            break
        if code != "00":
            raise RuntimeError(f"단기예보 오류 {code}: {body['header'].get('resultMsg')}")
        b = body["body"]
        total = int(b.get("totalCount", 0))
        got = b.get("items", {}).get("item", []) if b.get("items") else []
        items.extend(got)
        if len(items) >= total or not got:
            break
        page += 1
    path.write_text(json.dumps({"base": f"{base:%Y-%m-%d %H:%M}", "region": region, "nx": nx, "ny": ny,
                                "totalCount": total, "items": items}, ensure_ascii=False), encoding="utf-8")
    return "ok"


# ---------------------------------------------------------------- 실행
def collect(part, dry_run=False):
    days = load_days(part)
    aws_need = sorted({t for d, _ in days for t in aws_times(d)})
    fcst_need = sorted({b for d, _ in days for b in fcst_bases(d)})
    aws_todo = [t for t in aws_need if not (AWS_DIR / f"aws_{t:%Y%m%d%H}.txt").exists()]
    fcst_todo = [(b, r) for b in fcst_need for r in REGION_GRIDS
                 if not (FCST_DIR / f"fcst_{b:%Y%m%d%H}_{r}.json").exists()]
    log.info("=== part %s: %d일 · AWS %d시각(남은 %d) · 예보 %d발표 × %d권역(남은 %d) ===",
             part, len(days), len(aws_need), len(aws_todo), len(fcst_need), len(REGION_GRIDS), len(fcst_todo))
    if dry_run:
        log.info("dry-run: 호출 예상 약 %d건 (예보는 페이지가 넘치면 조금 더)", len(aws_todo) + len(fcst_todo))
        return
    key = load_key()
    calls, today = load_calls()
    AWS_DIR.mkdir(parents=True, exist_ok=True)
    FCST_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    t0 = time.time()
    try:
        for i, t in enumerate(aws_todo, 1):
            fetch_aws(session, key, t, calls, today)
            if i % 100 == 0 or i == len(aws_todo):
                log.info("  AWS %d/%d (%.0f초)", i, len(aws_todo), time.time() - t0)
        for i, (b, r) in enumerate(fcst_todo, 1):
            fetch_fcst(session, key, b, r, calls, today)
            if i % 100 == 0 or i == len(fcst_todo):
                log.info("  예보 %d/%d (%.0f초)", i, len(fcst_todo), time.time() - t0)
    except AuthError as e:
        log.error("인증 오류로 멈춥니다: %s", str(e).replace(key, "***")[:300])
        log.error("→ api_keys.txt 의 KMA_APIHUB_KEY 와, API허브에서 이 API의 '활용신청'을 했는지 확인하세요.")
        return
    except LimitReached as e:
        log.warning(str(e))
    finally:
        CALLS_PATH.write_text(json.dumps(calls, ensure_ascii=False, indent=1), encoding="utf-8")
        log.info("오늘 누적 호출 %s건 · 이번 실행 %.0f초", f"{calls.get(today, 0):,}", time.time() - t0)


def parse_all():
    """받은 원본 전체를 split별 표로 정리한다."""
    split_of = {}
    for split in ("train", "eval"):
        d = pd.read_csv(PROC_DIR / split / f"weather_days_{split}.csv", encoding="utf-8-sig")
        for x in d["date"]:
            day = pd.Timestamp(x)
            for t in aws_times(day):
                split_of[("aws", t)] = split
            for b in fcst_bases(day):
                split_of[("fcst", b)] = split

    aws_frames, bad = [], 0
    for p in sorted(AWS_DIR.glob("aws_*.txt")):
        t = datetime.strptime(p.stem[4:], "%Y%m%d%H")
        df = parse_aws_text(p.read_text(encoding="utf-8"))
        if df is None:
            bad += 1
            continue
        df = df[df["STN"].isin(SEOUL_STNS)].copy()
        df.insert(0, "datetime", t)
        df.insert(1, "split", split_of.get(("aws", pd.Timestamp(t)), ""))
        aws_frames.append(df)
    if bad:
        log.warning("AWS %d개 파일에서 열 이름을 못 찾았습니다. 그 파일 머리말을 확인해 주세요.", bad)

    fc_rows = []
    for p in sorted(FCST_DIR.glob("fcst_*.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        base = pd.Timestamp(j["base"])
        for it in j["items"]:
            fc_rows.append({"base": base, "fcst": pd.Timestamp(f"{it['fcstDate']} {it['fcstTime'][:2]}:00"),
                            "region": j["region"], "nx": j["nx"], "ny": j["ny"],
                            "category": it["category"], "value": it["fcstValue"],
                            "split": split_of.get(("fcst", base), "")})

    for split in ("train", "eval"):
        out = PROC_DIR / split
        if aws_frames:
            a = pd.concat(aws_frames)
            a = a[a["split"] == split].drop(columns="split")
            if len(a):
                a.to_csv(out / f"aws_hourly_{split}.csv", index=False, encoding="utf-8-sig")
                log.info("저장: %s (%d행, 지점 %d개)", (out / f"aws_hourly_{split}.csv").relative_to(ROOT),
                         len(a), a["STN"].nunique())
        if fc_rows:
            f = pd.DataFrame(fc_rows)
            f = f[f["split"] == split].drop(columns="split")
            if len(f):
                f.to_csv(out / f"fcst_{split}.csv", index=False, encoding="utf-8-sig")
                log.info("저장: %s (%d행, 발표 %d번)", (out / f"fcst_{split}.csv").relative_to(ROOT),
                         len(f), f["base"].nunique())


def main():
    p = argparse.ArgumentParser(description="구별 AWS · 과거 단기예보 수집 (기상청 API허브)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--test", action="store_true", help=f"{TEST_DAY} 하루만")
    g.add_argument("--part", choices=["1", "2", "3"], help="1 평가 / 2 학습 겨울 / 3 학습 여름")
    g.add_argument("--parse", action="store_true", help="받은 원본을 표로 정리만")
    p.add_argument("--dry-run", action="store_true", help="호출 수만 세기")
    a = p.parse_args()
    setup_log()
    if not a.parse:
        collect("test" if a.test else a.part, a.dry_run)
    if not a.dry_run:
        parse_all()
    if a.test and not a.dry_run:
        show_test_result()


def show_test_result():
    """시험 결과를 화면에 보여 준다 (응답 형식 확인용)."""
    files = sorted(AWS_DIR.glob(f"aws_{TEST_DAY.replace('-', '')}18.txt"))
    if files:
        lines = files[0].read_text(encoding="utf-8").splitlines()
        print("\n----- AWS 원본 예시:", files[0].name, "(앞 25줄) -----")
        print("\n".join(lines[:25]))
        df = parse_aws_text("\n".join(lines))
        print("→ 표로 읽기:", "실패 (열 이름 줄을 못 찾음)" if df is None else f"{len(df)}개 지점, 열 {list(df.columns)}")
    fc = sorted(FCST_DIR.glob(f"fcst_{TEST_DAY.replace('-', '')}14_*.json"))
    for f in fc[:1]:
        j = json.loads(f.read_text(encoding="utf-8"))
        cats = sorted({it["category"] for it in j["items"]})
        times = sorted({it["fcstDate"] + it["fcstTime"] for it in j["items"]})
        print("\n----- 예보 예시:", f.name, "-----")
        print(f"항목 {cats}, 예보 시각 {len(times)}개 ({times[0] if times else '-'} ~ {times[-1] if times else '-'}), 행 {len(j['items'])}")
        snow = [it for it in j["items"] if it["category"] in ("PCP", "SNO", "PTY") and it["fcstDate"] == TEST_DAY.replace("-", "")]
        for it in snow[:12]:
            print(" ", it["fcstDate"], it["fcstTime"], it["category"], it["fcstValue"])


if __name__ == "__main__":
    main()
