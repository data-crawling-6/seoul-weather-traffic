"""
구간(링크) 단위 평소값 · 정체 길이 비율 (B 담당)

도로 전체로 합치면 국지 정체가 묻히므로, 링크마다 평소값을 구하고
도로 · 방향 · 시간마다 '막힌 구간이 도로 길이의 몇 %인가'를 계산한다.

정의
- 링크 평소값   : 같은 시즌 · 링크 · 요일 · 시간대의 맑은 시간 링크 통행시간 중앙값
                  (맑음 = event_id · impact 없음, 칸당 맑은 시간이 MIN_N 미만이면 같은 시즌 · 링크 · 시간의 평일 전체로 대신)
- 링크 비율     : 그 시간 링크 통행시간 ÷ 링크 평소값
- 링크 정체     : 링크 비율 > LINK_RATIO (기본 1.5 = 속도가 평소의 2/3 미만, 도로 단위 기준과 같게)
- 정체 길이 비율: 정체 링크 거리 합 ÷ 관측된 링크 거리 합 (도로 · 방향 · 날짜 · 시간)
- 결측 길이 비율: 속도가 없는 링크 거리 합 ÷ 도로 전체 거리 (통제 · 마비 신호로 따로 본다)

입력
  C 릴리스 yang-seoyeon-traffic-weather-v3 의 링크 기본표 parquet
  (yang_seoyeon_topis_weather_link_hourly_20261122_v02.parquet, 약 600MB)
  → 연구 기간 · 평일 · 공휴일 제외 · 출퇴근 시간(8 · 9 · 18 · 19 · 20시)만 읽는다

실행
  python src/link_baseline.py
  python src/link_baseline.py --link 경로.parquet --link-ratio 1.5

결과
  data/processed/{train,eval}/link_baseline_{split}.parquet          링크 평소값
  data/processed/{train,eval}/road_congestion_length_{split}.parquet 도로 · 방향 · 시간별 정체 길이 비율
  outputs/tables/congestion_length_candidates_train.csv              정체 길이 기준 후보 비교 (학습만)
  logs/link_baseline.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LINK = ROOT / "data" / "raw" / "c_traffic" / "yang_seoyeon_topis_weather_link_hourly_20261122_v02.parquet"
PROC_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "outputs" / "tables"

COMMUTE_HOURS = [8, 9, 18, 19, 20]
MIN_N = 5
LENGTH_CANDIDATES = [0.1, 0.2, 0.3, 0.5]   # 정체 길이 비율 기준 후보
COLUMNS = ["link_id", "road_id", "direction", "function_type", "date", "hour", "season", "split",
           "distance_m", "link_travel_time_seconds", "missing_reason", "event_id", "impact"]


def load_links(path: Path) -> pd.DataFrame:
    flt = ((ds.field("in_period") == True) & (ds.field("is_weekday") == True)  # noqa: E712
           & (ds.field("is_holiday") == False) & ds.field("hour").isin(COMMUTE_HOURS))  # noqa: E712
    d = ds.dataset(path).to_table(columns=COLUMNS, filter=flt).to_pandas()
    d["date"] = pd.to_datetime(d["date"])
    d["weekday"] = d["date"].dt.weekday.astype("int8")
    clear = d["event_id"].isna() & d["impact"].isna()
    d["weather_group"] = np.where(d["impact"].notna(), "악천후", np.where(clear, "맑음", "약한 비·눈"))
    return d


def make_link_baseline(d: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "link_id", "weekday", "hour"]
    clear = d[(d["weather_group"] == "맑음") & d["link_travel_time_seconds"].notna()]
    by_day = (clear.groupby(keys)["link_travel_time_seconds"]
              .agg(tt_weekday="median", n_weekday="size").reset_index())
    by_all = (clear.groupby(["season", "link_id", "hour"])["link_travel_time_seconds"]
              .agg(tt_all="median", n_all="size").reset_index())
    meta = d.drop_duplicates(["season", "link_id"])[
        ["season", "split", "link_id", "road_id", "direction", "function_type", "distance_m"]]
    b = (d[keys].drop_duplicates()
         .merge(by_day, on=keys, how="left")
         .merge(by_all, on=["season", "link_id", "hour"], how="left")
         .merge(meta, on=["season", "link_id"], how="left"))
    b["n_weekday"] = b["n_weekday"].fillna(0).astype(int)
    use_day = b["n_weekday"] >= MIN_N
    b["baseline_link_tt_s"] = np.where(use_day, b["tt_weekday"], b["tt_all"])
    b["baseline_n"] = np.where(use_day, b["n_weekday"], b["n_all"])
    b["baseline_level"] = np.where(use_day, "weekday", np.where(b["tt_all"].notna(), "all_weekdays", "none"))
    cols = ["season", "split", "link_id", "road_id", "direction", "function_type", "distance_m",
            "weekday", "hour", "baseline_link_tt_s", "baseline_n", "baseline_level"]
    return b[cols].sort_values(["season", "road_id", "direction", "link_id", "weekday", "hour"]).reset_index(drop=True)


def road_congestion_length(d: pd.DataFrame, b: pd.DataFrame, link_ratio: float) -> pd.DataFrame:
    keys = ["season", "link_id", "weekday", "hour"]
    x = d.merge(b[keys + ["baseline_link_tt_s"]], on=keys, how="left")
    x["link_ratio"] = x["link_travel_time_seconds"] / x["baseline_link_tt_s"]
    valid = x["link_ratio"].notna()
    x["obs_dist"] = x["distance_m"].where(valid, 0.0)
    x["cong_dist"] = x["distance_m"].where(valid & (x["link_ratio"] > link_ratio), 0.0)
    x["miss_dist"] = x["distance_m"].where(x["link_travel_time_seconds"].isna(), 0.0)
    x["is_cong"] = (valid & (x["link_ratio"] > link_ratio)).astype("int16")
    g = (x.groupby(["date", "hour", "season", "split", "road_id", "direction", "function_type", "weather_group"])
         .agg(n_links=("link_id", "size"), n_congested_links=("is_cong", "sum"),
              total_distance_m=("distance_m", "sum"), observed_distance_m=("obs_dist", "sum"),
              congested_distance_m=("cong_dist", "sum"), missing_distance_m=("miss_dist", "sum"),
              max_link_ratio=("link_ratio", "max"))
         .reset_index())
    g["congestion_length_ratio"] = np.where(g["observed_distance_m"] > 0,
                                            g["congested_distance_m"] / g["observed_distance_m"], np.nan)
    g["missing_length_ratio"] = g["missing_distance_m"] / g["total_distance_m"]
    return g.sort_values(["date", "hour", "road_id", "direction"]).reset_index(drop=True)


def compare_candidates(g: pd.DataFrame) -> pd.DataFrame:
    """학습 시즌만: 정체 길이 비율 기준마다 날씨 구분별 해당 시간 비율과 '악천후 ÷ 맑음' 배수."""
    t = g[(g["split"] == "train") & g["congestion_length_ratio"].notna()]
    rows = []
    for th in LENGTH_CANDIDATES:
        hit = t["congestion_length_ratio"] >= th
        rate = hit.groupby(t["weather_group"]).mean()
        rows.append({"candidate": f"length_ratio>={th:.0%}", "rows": len(t), "positives": int(hit.sum()),
                     "rate_clear": rate.get("맑음"), "rate_light": rate.get("약한 비·눈"),
                     "rate_severe": rate.get("악천후"),
                     "lift_severe_vs_clear": rate.get("악천후") / rate.get("맑음")})
    any_cong = t["n_congested_links"] > 0
    rate = any_cong.groupby(t["weather_group"]).mean()
    rows.append({"candidate": "any_link_congested", "rows": len(t), "positives": int(any_cong.sum()),
                 "rate_clear": rate.get("맑음"), "rate_light": rate.get("약한 비·눈"),
                 "rate_severe": rate.get("악천후"), "lift_severe_vs_clear": rate.get("악천후") / rate.get("맑음")})
    miss = g[g["split"] == "train"].groupby("weather_group")["missing_length_ratio"].mean()
    rows.append({"candidate": "mean_missing_length_ratio", "rows": int((g["split"] == "train").sum()),
                 "positives": None, "rate_clear": miss.get("맑음"), "rate_light": miss.get("약한 비·눈"),
                 "rate_severe": miss.get("악천후"), "lift_severe_vs_clear": miss.get("악천후") / miss.get("맑음")})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="구간 단위 평소값 · 정체 길이 비율 (B)")
    p.add_argument("--link", type=Path, default=DEFAULT_LINK, help="C 링크 기본표 parquet")
    p.add_argument("--link-ratio", type=float, default=1.5, help="링크 정체 기준 (통행시간 ÷ 평소값 초과)")
    a = p.parse_args()

    d = load_links(a.link)
    b = make_link_baseline(d)
    g = road_congestion_length(d, b, a.link_ratio)

    for split in ("train", "eval"):
        out = PROC_DIR / split
        out.mkdir(parents=True, exist_ok=True)
        b[b["split"] == split].to_parquet(out / f"link_baseline_{split}.parquet", index=False)
        g[g["split"] == split].to_parquet(out / f"road_congestion_length_{split}.parquet", index=False)
        print(f"저장: {split} 링크 평소값 {(b['split'] == split).sum():,}칸 · 정체 길이 {(g['split'] == split).sum():,}행")

    cand = compare_candidates(g)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cand.to_csv(TABLE_DIR / "congestion_length_candidates_train.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_at": datetime.now(KST).isoformat(timespec="seconds"),
        "input": a.link.name,
        "link_rows": len(d),
        "links": int(d["link_id"].nunique()),
        "link_tt_nan_rows": int(d["link_travel_time_seconds"].isna().sum()),
        "link_ratio_threshold": a.link_ratio,
        "baseline_cells": len(b),
        "baseline_level_counts": b["baseline_level"].value_counts().to_dict(),
        "road_rows": len(g),
        "min_n": MIN_N,
    }
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "link_baseline.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
