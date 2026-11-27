"""
24시간 · 주말 도로 표 (B 담당)

C 링크 기본표(v02, 24시간 격자)로 연구 기간 '모든 요일 · 24시간'의 도로 × 방향 통행시간 비율과
정체 여부를 만든다. 기존 공통 표(v04 · baseline · road_congestion_length)는 평일 출퇴근 5시간만 있어서,
24시간 비교(Q1 · Q3, 노션 분석 설계 6.3)와 생존분석(특보 뒤 몇 시간 만에 막히나, 9.2 ③)에 쓸 표를 따로 둔다.

정의 (분석 설계 3.4 · 4.1 · 6.3을 24시간으로 넓힌 것)
- 요일 유형   : 평일 / 토 / 일 (공휴일은 뺀다). 6.3 (2) "평소값도 시간대 · 요일 유형마다 따로"
- 도로 통행시간: 링크 통행시간 합. 링크가 하나라도 빠지면 빈값 (v04 규칙과 같음)
- 도로 평소값 : 같은 시즌 · 도로 · 방향 · 요일 유형 · 시간의 맑은 시간 중앙값 (맑음 = event_id · impact 없음)
- 비율       : 도로 통행시간 ÷ 도로 평소값
- 링크 평소값 : 같은 시즌 · 링크 · 요일 유형 · 시간의 맑은 시간 중앙값
- 정체       : 링크 비율 > 1.5 인 링크 길이가 도로 길이의 30% 이상 (periods.yaml thresholds, link_baseline.py 와 같은 규칙)
- 평소값 표본이 MIN_N 미만인 칸은 비율 · 정체를 빈값으로 둔다

출퇴근 시간 평일 행의 비율은 정답용 평소값(baseline_*.csv)과 거의 같지만 같지는 않다
(그쪽은 요일별 평소값, 여기는 평일을 묶은 평소값). 모델 정답에는 쓰지 않는다.

입력   data/derived/topis_weather/link_hourly_20261122_v02.parquet (팀 공유 드라이브)
실행   python src/analysis/hourly_24h.py            # train_dev · holdout 둘 다 만든다 (holdout 은 만들기만 하고 분석에서 열지 않음)
결과   data/derived/topis_weather/road_hourly_24h_{train_dev,holdout}_v01.parquet
       logs/ljh/hourly_24h/hourly_24h.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.io_utils import SPLIT_DIR, load_weather_labels  # noqa: E402
from src.labels.link_baseline import load_thresholds  # noqa: E402

DEFAULT_LINK = ROOT / "data" / "derived" / "topis_weather" / "link_hourly_20261122_v02.parquet"
OUT_DIR = ROOT / "data" / "derived" / "topis_weather"
LOG_DIR = ROOT / "logs" / "ljh" / "hourly_24h"
VERSION = "v01"
MIN_N = 5
COLUMNS = ["link_id", "road_id", "road_name", "direction", "function_type", "date", "hour",
           "season", "split", "in_period", "is_holiday", "distance_m", "link_travel_time_seconds"]
DAYTYPE = {0: "평일", 1: "평일", 2: "평일", 3: "평일", 4: "평일", 5: "토", 6: "일"}


def out_path(split_dir: str) -> Path:
    return OUT_DIR / f"road_hourly_24h_{split_dir}_{VERSION}.parquet"


def weather_group(w: pd.DataFrame) -> pd.Series:
    """맑음 / 약한 비·눈 / 악천후 (분석 설계 3.3, link_baseline.py 와 같은 규칙)."""
    return pd.Series(np.where(w["impact"].notna(), "악천후",
                              np.where(w["event_id"].isna(), "맑음", "약한 비·눈")), index=w.index)


def load_season(path: Path, season: str) -> pd.DataFrame:
    flt = (ds.field("season") == season) & (ds.field("in_period") == True) & (ds.field("is_holiday") == False)  # noqa: E712
    d = ds.dataset(path).to_table(columns=COLUMNS, filter=flt).to_pandas()
    for c in ["link_id", "road_id", "road_name", "direction", "function_type", "season", "split"]:
        d[c] = d[c].astype("category")
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["hour"] = d["hour"].astype("int8")
    d["daytype"] = d["date"].dt.dayofweek.map(DAYTYPE).astype("category")
    return d.drop(columns=["in_period", "is_holiday"])


def build_season(d: pd.DataFrame, labels: pd.DataFrame, link_ratio: float, length_ratio: float) -> pd.DataFrame:
    w = labels.copy()
    w["weather_group"] = weather_group(w)
    d = d.merge(w[["date", "hour", "event_id", "impact", "weather_group"]], on=["date", "hour"], how="left",
                validate="many_to_one")
    if d["weather_group"].isna().any():
        raise ValueError("weather_hourly 에 없는 date · hour 가 있습니다.")
    clear = d["weather_group"].eq("맑음")

    # 링크 평소값 → 링크 비율 → 정체 길이
    lk = ["link_id", "daytype", "hour"]
    lb = (d[clear & d["link_travel_time_seconds"].notna()]
          .groupby(lk, observed=True)["link_travel_time_seconds"].agg(link_base="median", link_n="size").reset_index())
    d = d.merge(lb, on=lk, how="left")
    d["link_ratio"] = d["link_travel_time_seconds"] / d["link_base"].where(d["link_n"] >= MIN_N)
    valid = d["link_ratio"].notna()
    d["cong_m"] = d["distance_m"].where(valid & (d["link_ratio"] > link_ratio), 0.0)
    d["miss_m"] = d["distance_m"].where(d["link_travel_time_seconds"].isna(), 0.0)
    d["has_tt"] = d["link_travel_time_seconds"].notna()

    keys = ["season", "split", "road_id", "road_name", "direction", "function_type", "date", "hour", "daytype"]
    g = (d.groupby(keys, observed=True)
         .agg(n_links=("link_id", "size"), n_obs=("has_tt", "sum"), n_valid_ratio=("link_ratio", "count"),
              tt_s=("link_travel_time_seconds", "sum"), total_m=("distance_m", "sum"),
              congested_m=("cong_m", "sum"), missing_m=("miss_m", "sum"),
              event_id=("event_id", "first"), impact=("impact", "first"), weather_group=("weather_group", "first"))
         .reset_index())
    g["tt_s"] = g["tt_s"].where(g["n_obs"] == g["n_links"])  # 링크가 하나라도 빠지면 빈값

    # 도로 평소값 (맑은 시간 도로 통행시간 중앙값) → 비율
    rk = ["road_id", "direction", "daytype", "hour"]
    rb = (g[g["weather_group"].eq("맑음") & g["tt_s"].notna()]
          .groupby(rk, observed=True)["tt_s"].agg(baseline_tt_s="median", baseline_n="size").reset_index())
    g = g.merge(rb, on=rk, how="left")
    g["ratio"] = g["tt_s"] / g["baseline_tt_s"].where(g["baseline_n"] >= MIN_N)

    g["congestion_length_ratio"] = g["congested_m"] / g["total_m"]
    g["missing_length_ratio"] = g["missing_m"] / g["total_m"]
    g["is_complete"] = g["n_valid_ratio"] == g["n_links"]
    g["is_congested"] = (g["congestion_length_ratio"] >= length_ratio).astype("Int8").mask(~g["is_complete"])
    g["datetime"] = g["date"] + pd.to_timedelta(g["hour"].astype(int), unit="h")
    for c in ["daytype", "weather_group", "impact"]:
        g[c] = g[c].astype("string")
    return g.drop(columns=["n_valid_ratio", "congested_m", "missing_m"])


def main() -> None:
    th = load_thresholds()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--link", type=Path, default=DEFAULT_LINK)
    p.add_argument("--splits", nargs="+", default=["train", "eval"], choices=["train", "eval"])
    a = p.parse_args()

    import yaml
    cfg = yaml.safe_load((ROOT / "config" / "periods.yaml").read_text(encoding="utf-8"))
    labels = load_weather_labels()
    log = {"run_at": datetime.now(KST).isoformat(timespec="seconds"), "input": a.link.name,
           "link_ratio": th["congestion_link_ratio"], "road_length_ratio": th["congestion_road_length_ratio"],
           "min_n": MIN_N, "outputs": {}}
    for split in a.splits:
        parts = []
        for per in cfg["periods"]:
            if per["split"] != split:
                continue
            d = load_season(a.link, per["season"])
            g = build_season(d, labels, th["congestion_link_ratio"], th["congestion_road_length_ratio"])
            print(f"{per['season']}: 링크 {len(d):,}행 → 도로 {len(g):,}행 · 비율 빈값 {g['ratio'].isna().mean():.2%}")
            parts.append(g)
            del d
        out = pd.concat(parts, ignore_index=True)
        for c in ["season", "split", "road_id", "road_name", "direction", "function_type"]:
            out[c] = out[c].astype("string")
        path = out_path(SPLIT_DIR[split])
        out.to_parquet(path, index=False)
        log["outputs"][path.name] = {"rows": len(out), "ratio_missing": round(float(out["ratio"].isna().mean()), 4),
                                     "congested_rows": int(out["is_congested"].sum())}
        print("저장:", path.relative_to(ROOT), f"{len(out):,}행")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "hourly_24h.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
