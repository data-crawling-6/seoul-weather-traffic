"""
한강 수위 수집 (B 담당 · 한강홍수통제소 오픈 API)

여름 하천변 도로(올림픽대로 · 강변북로 · 동부간선 · 서부간선 등) 통제와 지연을 설명하려고
서울 안 수위 관측소의 10분 수위를 여름 4년(2023~2026, 5~9월) 받아서 시간 단위 표로 만든다.
시각은 다른 표와 같이 '끝 시각' 기준이다 (13시 = 12:10~13:00 의 10분 자료).

실행 순서:
  python crawling/hrfco_waterlevel.py --stations   # 1) 관측소 목록만 받아서 서울 관측소 확인 (1건)
  python crawling/hrfco_waterlevel.py --test       # 2) 한강대교 2026년 7월 1건만 받아서 응답 확인
  python crawling/hrfco_waterlevel.py              # 3) 서울 관측소 전부 × 20개월 (남은 것만)
  python crawling/hrfco_waterlevel.py --workers 3  #    동시 3개 (느릴 때)
  python crawling/hrfco_waterlevel.py --dry-run    #    호출 수만 세고 받지 않음
  python crawling/hrfco_waterlevel.py --parse      # 받은 원본을 표로 정리만 다시

- 관측소: 관측소 목록에서 주소가 '서울'로 시작하는 곳 전부 (주소가 없으면 서울 범위 좌표로 고른다).
  특정 관측소만 받으려면 --codes 1018683 1018680 처럼 준다.
- 한 번에 관측소 1곳 × 한 달씩 요청한다. 이미 받은 달은 건너뛴다 (이어받기).
- 실패하면 3번까지 다시 시도하고, 그래도 실패한 건은 건너뛴다. 다시 실행하면 그것만 받는다.
- 인증 오류로 보이면 바로 멈춘다. 로그(logs/ljh/hrfco_waterlevel.log)에서 인증키는 *** 로 가린다.

인증키: api_keys.txt 에  HRFCO_KEY=발급받은키

결과
  data/raw/hrfco/stations.json                           관측소 목록 원본
  data/raw/hrfco/wl_{관측소코드}_{YYYYMM}.json             10분 수위 원본 (한 달씩)
  data/processed/waterlevel_stations.csv                 받은 관측소 · 기준수위(관심 · 주의 · 경보 · 심각)
  data/processed/{train,eval}/waterlevel_hourly_{split}.csv
      datetime(끝 시각) · wlobscd · obsnm · wl_last(정시 수위) · wl_max · wl_mean · wl_rise_3h
      · fw_last(유량) · fw_max · n_obs · over_attwl(관심 이상) · over_wrnwl(주의 이상)
      학습 = 2023 · 2024 · 2025 여름, 평가 = 2026 여름. 연구 기간 5/9~9/23 만 남긴다.
"""
import argparse
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "hrfco"
PROC_DIR = ROOT / "data" / "processed"
LOG_PATH = ROOT / "logs" / "ljh" / "hrfco_waterlevel.log"


YEARS = {2023: "train", 2024: "train", 2025: "train", 2026: "eval"}
MONTHS = [5, 6, 7, 8, 9]
PERIOD = ("05-09", "09-23")           # config/periods.yaml 여름 연구 기간
SEOUL_BOX = (37.41, 37.72, 126.76, 127.19)  # 위도 최소 · 최대, 경도 최소 · 최대
TEST_CODE_NAME = "한강대교"

log = logging.getLogger("hrfco")
_local = threading.local()


class AuthError(Exception):
    pass


def setup_log():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def load_key(name="HRFCO_KEY"):
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


HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) seoul-weather-traffic/1.0",
           "Accept": "application/json"}
BASES = ["http://api.hrfco.go.kr", "https://api.hrfco.go.kr"]  # 하나가 끊기면 다른 쪽으로


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(HEADERS)
    return _local.s


def get_json(key, path, tries=4):
    """BASE/{key}/{path} 를 받아 JSON 으로. 실패하면 http · https 를 번갈아 다시 시도."""
    last = ""
    for i in range(tries):
        url = f"{BASES[i % len(BASES)]}/{key}/{path}"
        try:
            r = session().get(url, timeout=60)
            text = r.text
            if r.status_code in (401, 403):
                raise AuthError(f"HTTP {r.status_code}: {text[:200]}")
            r.raise_for_status()
            try:
                data = r.json()
            except ValueError:
                low = text.lower()
                if "key" in low or "인증" in text:
                    raise AuthError(text[:200])
                raise ValueError(f"JSON 아님: {text[:200]}")
            if "content" not in data:
                msg = json.dumps(data, ensure_ascii=False)[:300]
                if "key" in msg.lower() or "인증" in msg:
                    raise AuthError(msg)
                raise ValueError(f"content 없음: {msg}")
            return data
        except AuthError:
            raise
        except Exception as e:  # noqa: BLE001
            last = str(e).replace(key, "***")
            time.sleep(2 * (i + 1))
    raise RuntimeError(last)


# ---------------------------------------------------------------- 관측소

def dms_to_deg(v):
    """'127-00-12' 같은 도분초 글자나 숫자를 도 단위로."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    parts = re.split(r"[-:\s]+", s)
    try:
        if len(parts) >= 2:
            d, m = float(parts[0]), float(parts[1])
            sec = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
            return d + m / 60 + sec / 3600
        return float(s)
    except ValueError:
        return None


def load_stations(key, refresh=False):
    path = RAW_DIR / "stations.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))["content"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    data = get_json(key, "waterlevel/info.json")
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data["content"]


def seoul_stations(stations):
    lat0, lat1, lon0, lon1 = SEOUL_BOX
    out = []
    for s in stations:
        addr = str(s.get("addr") or "")
        if addr:
            ok = addr.startswith("서울")
        else:
            lat, lon = dms_to_deg(s.get("lat")), dms_to_deg(s.get("lon"))
            ok = lat is not None and lon is not None and lat0 <= lat <= lat1 and lon0 <= lon <= lon1
        if ok:
            out.append(s)
    return out


def num(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def save_station_table(stations):
    rows = [{
        "wlobscd": str(s.get("wlobscd")),
        "obsnm": s.get("obsnm"),
        "agcnm": s.get("agcnm"),
        "addr": s.get("addr"),
        "lat": dms_to_deg(s.get("lat")),
        "lon": dms_to_deg(s.get("lon")),
        "attwl": num(s.get("attwl")),  # 관심
        "wrnwl": num(s.get("wrnwl")),  # 주의
        "almwl": num(s.get("almwl")),  # 경보
        "srswl": num(s.get("srswl")),  # 심각
    } for s in stations]
    df = pd.DataFrame(rows).sort_values("wlobscd")
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC_DIR / "waterlevel_stations.csv", index=False, encoding="utf-8-sig")
    return df


# ---------------------------------------------------------------- 수집

def month_range(y, m):
    last = pd.Period(f"{y}-{m:02d}").days_in_month
    return f"{y}{m:02d}010000", f"{y}{m:02d}{last}2350"


def raw_path(code, y, m):
    return RAW_DIR / f"wl_{code}_{y}{m:02d}.json"


def fetch_month(key, code, y, m):
    sdt, edt = month_range(y, m)
    data = get_json(key, f"waterlevel/list/10M/{code}/{sdt}/{edt}.json")
    raw_path(code, y, m).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return len(data.get("content") or [])


def collect(key, codes, workers=1, dry_run=False):
    jobs = [(c, y, m) for c in codes for y in YEARS for m in MONTHS if not raw_path(c, y, m).exists()]
    total = len(codes) * len(YEARS) * len(MONTHS)
    log.info("관측소 %d곳 × %d개월 = %d건 중 남은 것 %d건", len(codes), len(YEARS) * len(MONTHS), total, len(jobs))
    if dry_run or not jobs:
        return
    t0, done, fails = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_month, key, *j): j for j in jobs}
        for f in as_completed(futs):
            c, y, m = futs[f]
            try:
                n = f.result()
                done += 1
                if n == 0:
                    log.warning("%s %d-%02d: 자료 0건 (관측 안 한 달일 수 있음)", c, y, m)
                if done % 20 == 0 or done == len(jobs):
                    log.info("  %d/%d 완료 (%.0f초)", done, len(jobs), time.time() - t0)
            except AuthError as e:
                log.error("인증 오류로 멈춥니다: %s", str(e).replace(key, "***"))
                log.error("→ api_keys.txt 의 HRFCO_KEY 를 확인하세요.")
                for x in futs:
                    x.cancel()
                raise SystemExit(1)
            except Exception as e:  # noqa: BLE001
                fails.append((c, y, m))
                log.warning("%s %d-%02d 실패 (건너뜀): %s", c, y, m, str(e).replace(key, "***")[:200])
            time.sleep(0.2)
    log.info("끝: 성공 %d건 · 실패 %d건 (%.0f초). 실패한 건은 다시 실행하면 받는다.",
             done, len(fails), time.time() - t0)


# ---------------------------------------------------------------- 정리

def read_raw():
    frames = []
    for p in sorted(RAW_DIR.glob("wl_*_*.json")):
        content = json.loads(p.read_text(encoding="utf-8")).get("content") or []
        if content:
            frames.append(pd.DataFrame(content))
    if not frames:
        return pd.DataFrame(columns=["wlobscd", "ymdhm", "wl", "fw"])
    return pd.concat(frames, ignore_index=True)


def to_hourly(raw, stations):
    """10분 자료 → 끝 시각 기준 시간 표. 13시 = 12:10 ~ 13:00."""
    d = raw.copy()
    d["wlobscd"] = d["wlobscd"].astype(str)
    d["t"] = pd.to_datetime(d["ymdhm"].astype(str).str[:12], format="%Y%m%d%H%M", errors="coerce")
    d["wl"] = pd.to_numeric(d["wl"].astype(str).str.strip(), errors="coerce")
    d["fw"] = pd.to_numeric(d.get("fw", pd.Series(index=d.index, dtype=object)).astype(str).str.strip(),
                            errors="coerce")
    d = d.dropna(subset=["t"]).drop_duplicates(["wlobscd", "t"])
    d["datetime"] = d["t"].dt.ceil("h")

    g = d.groupby(["wlobscd", "datetime"])
    h = g.agg(wl_max=("wl", "max"), wl_mean=("wl", "mean"), fw_max=("fw", "max"),
              n_obs=("wl", "count")).reset_index()
    on_hour = d[d["t"] == d["datetime"]][["wlobscd", "datetime", "wl", "fw"]]
    h = h.merge(on_hour.rename(columns={"wl": "wl_last", "fw": "fw_last"}), on=["wlobscd", "datetime"], how="left")

    # 정시 값이 없으면 그 시간의 마지막 10분 값
    last = d.sort_values("t").groupby(["wlobscd", "datetime"])[["wl", "fw"]].last()
    last = last.rename(columns={"wl": "wl_tail", "fw": "fw_tail"}).reset_index()
    h = h.merge(last, on=["wlobscd", "datetime"], how="left")
    h["wl_last"] = h["wl_last"].fillna(h["wl_tail"])
    h["fw_last"] = h["fw_last"].fillna(h["fw_tail"])
    h = h.drop(columns=["wl_tail", "fw_tail"])

    # 3시간 전 정시 수위와의 차이 (시간 칸이 비어 있으면 NaN)
    h = h.sort_values(["wlobscd", "datetime"])
    prev = h[["wlobscd", "datetime", "wl_last"]].copy()
    prev["datetime"] = prev["datetime"] + pd.Timedelta(hours=3)
    h = h.merge(prev.rename(columns={"wl_last": "wl_3h_ago"}), on=["wlobscd", "datetime"], how="left")
    h["wl_rise_3h"] = h["wl_last"] - h["wl_3h_ago"]
    h = h.drop(columns=["wl_3h_ago"])

    st = stations[["wlobscd", "obsnm", "attwl", "wrnwl"]].copy()
    st["wlobscd"] = st["wlobscd"].astype(str)
    for c in ("attwl", "wrnwl"):
        st[c] = pd.to_numeric(st[c], errors="coerce")
    h = h.merge(st, on="wlobscd", how="left")
    h["over_attwl"] = (h["wl_max"] >= h["attwl"]).where(h["attwl"].notna())
    h["over_wrnwl"] = (h["wl_max"] >= h["wrnwl"]).where(h["wrnwl"].notna())
    h = h.drop(columns=["attwl", "wrnwl"])

    # 연구 기간만 (끝 시각 기준: 5/9 01시 ~ 9/24 00시)
    y = h["datetime"].dt.year.astype(str)
    start = pd.to_datetime(y + "-" + PERIOD[0])
    end = pd.to_datetime(y + "-" + PERIOD[1]) + pd.Timedelta(days=1)
    h = h[(h["datetime"] > start) & (h["datetime"] <= end)].copy()
    h["split"] = h["datetime"].dt.year.map(YEARS)
    cols = ["datetime", "wlobscd", "obsnm", "wl_last", "wl_max", "wl_mean", "wl_rise_3h",
            "fw_last", "fw_max", "n_obs", "over_attwl", "over_wrnwl", "split"]
    return h[cols].sort_values(["datetime", "wlobscd"]).reset_index(drop=True)


def parse_all():
    st_path = PROC_DIR / "waterlevel_stations.csv"
    if not st_path.exists():
        raise FileNotFoundError("먼저 --stations 또는 수집을 실행하세요 (관측소 표가 없음).")
    stations = pd.read_csv(st_path, dtype={"wlobscd": str})
    raw = read_raw()
    if raw.empty:
        log.warning("받은 원본이 없습니다.")
        return
    h = to_hourly(raw, stations)
    for split, part in h.groupby("split"):
        out = PROC_DIR / split / f"waterlevel_hourly_{split}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        part.drop(columns=["split"]).to_csv(out, index=False, encoding="utf-8-sig")
        log.info("%s: %d행 · 관측소 %d곳 · %s ~ %s → %s", split, len(part), part["wlobscd"].nunique(),
                 part["datetime"].min(), part["datetime"].max(), out.relative_to(ROOT))
    full = h.groupby("wlobscd")["n_obs"].apply(lambda s: (s == 6).mean())
    for code, r in full.items():
        if r < 0.9:
            nm = stations.loc[stations["wlobscd"] == code, "obsnm"]
            log.warning("관측소 %s(%s): 10분 자료 6개가 다 찬 시간이 %.0f%%뿐", code,
                        nm.iloc[0] if len(nm) else "?", r * 100)


# ---------------------------------------------------------------- 실행

def main():
    p = argparse.ArgumentParser(description="한강홍수통제소 수위 수집")
    p.add_argument("--stations", action="store_true", help="관측소 목록만 받아서 서울 관측소 보여 주기")
    p.add_argument("--test", action="store_true", help=f"{TEST_CODE_NAME} 2026년 7월 1건만")
    p.add_argument("--codes", nargs="+", help="받을 관측소 코드 (기본: 서울 관측소 전부)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--parse", action="store_true", help="받은 원본을 표로 정리만 다시")
    a = p.parse_args()
    setup_log()

    if a.parse:
        parse_all()
        return

    key = load_key()
    try:
        stations = load_stations(key, refresh=a.stations)
    except AuthError as e:
        log.error("인증 오류로 멈춥니다: %s", str(e).replace(key, "***"))
        log.error("→ api_keys.txt 의 HRFCO_KEY 를 확인하세요.")
        raise SystemExit(1)
    except RuntimeError as e:
        log.error("관측소 목록을 못 받았습니다 (http · https 모두 실패): %s", str(e)[:300])
        log.error("→ 사용URL에 등록한 IP와 지금 IP가 같은지, 등록 변경이 반영됐는지 확인하세요.")
        raise SystemExit(1)
    seoul = seoul_stations(stations)
    if a.codes:
        want = set(a.codes)
        seoul = [s for s in stations if str(s.get("wlobscd")) in want]
    table = save_station_table(seoul)

    if a.stations:
        log.info("전체 관측소 %d곳 중 서울 %d곳:", len(stations), len(table))
        for r in table.itertuples():
            log.info("  %s  %-12s %s  (관심 %s · 주의 %s)", r.wlobscd, r.obsnm, r.addr, r.attwl, r.wrnwl)
        return

    if a.test:
        hit = [s for s in stations if str(s.get("obsnm", "")).startswith(TEST_CODE_NAME)]
        if not hit:
            log.error("%s 관측소를 못 찾았습니다. --stations 로 이름을 확인하세요.", TEST_CODE_NAME)
            raise SystemExit(1)
        code = str(hit[0]["wlobscd"])
        n = fetch_month(key, code, 2026, 7)
        sample = json.loads(raw_path(code, 2026, 7).read_text(encoding="utf-8"))["content"][:3]
        log.info("%s(%s) 2026-07: %d건. 앞 3건: %s", TEST_CODE_NAME, code, n, sample)
        return

    codes = [str(s["wlobscd"]) for s in seoul]
    collect(key, codes, workers=a.workers, dry_run=a.dry_run)
    if not a.dry_run:
        parse_all()


if __name__ == "__main__":
    main()
