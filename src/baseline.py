"""
평소값 · 정체 기준 (B 담당)

C의 도로 통행시간 표(v04)로 도로 × 방향마다 '평소 통행시간'을 만들고,
정체 기준 후보를 학습 기간만 보고 비교한다.

정의 (노션 '데이터 · 연구 기간' 4번 · '모델 · 평가 설계' 3번)
- 평소값   : 같은 시즌 · 도로 · 방향 · 요일 · 시간대의 '맑음' 시간 통행시간 중앙값
             맑음 = event_id 없음 · impact 없음 (B weather_hourly, A · B 합의 11/20)
             그 칸의 맑음 시간이 MIN_N 미만이면 같은 시즌 · 도로 · 방향 · 시간대의 평일 전체로 대신한다
- 비율     : 그 시간 통행시간 ÷ 평소 통행시간 (A의 예측 대상 y)
- 정체 후보: 속도가 평소의 50% · 60% · 2/3 · 77% 미만(비율 > 2.0 · 1.667 · 1.5 · 1.3), 시속 15km 이하(TOPIS 표출 기준)
             기준은 학습 시즌만 보고 정한다 (평가 시즌은 열지 않음)

입력
  C 릴리스 yang-seoyeon-traffic-weather-v3 의 v04 zip을 data/raw/c_traffic/ 에 풀어 둔 parquet
  (v03을 넣어도 링크가 일부라도 빠진 행을 NaN으로 두는 v04 규칙을 여기서 한 번 더 적용하므로 결과가 같다)

실행
  python src/baseline.py                         # 기본 경로의 parquet
  python src/baseline.py --road 경로.parquet

결과
  data/processed/{train,eval}/baseline_{split}.csv          평소값 (시즌 × 도로 × 방향 × 요일 × 시간)
  outputs/tables/congestion_threshold_candidates_train.csv  정체 기준 후보 비교 (학습만)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROAD = ROOT / "data" / "raw" / "c_traffic" / "yang_seoyeon_topis_weather_commute_20261122_v04.parquet"
PROC_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "outputs" / "tables"

MIN_N = 5                       # 요일 칸의 맑음 시간이 이보다 적으면 평일 전체로 대신
WEEKDAY_NAMES = ["월", "화", "수", "목", "금"]
KEYS = ["season", "road_id", "direction", "weekday", "hour"]
CANDIDATES = {                  # 이름: (설명, 정체 판정 함수)
    "speed_lt_50pct": ("속도 < 평소의 50% (비율 > 2.0)", lambda d: d["ratio"] > 1 / 0.5),
    "speed_lt_60pct": ("속도 < 평소의 60% (비율 > 1.667)", lambda d: d["ratio"] > 1 / 0.6),
    "speed_lt_67pct": ("속도 < 평소의 2/3 (비율 > 1.5)", lambda d: d["ratio"] > 1.5),
    "speed_lt_77pct": ("속도 < 평소의 77% (비율 > 1.3)", lambda d: d["ratio"] > 1.3),
    "speed_le_15kmh": ("속도 <= 15km/h (TOPIS 정체)", lambda d: d["road_speed_kmh"] <= 15),
}


def load_road(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d["date"] = pd.to_datetime(d["date"])
    # v04 규칙: 링크가 일부라도 빠지면 관측 링크만 합산되어 통행시간이 짧게(과소) 잡힌다 → NaN
    incomplete = d["observed_link_count"] < d["link_count"]
    d.loc[incomplete, ["travel_time_seconds", "road_speed_kmh"]] = np.nan
    d["weekday"] = d["date"].dt.weekday.astype("int8")
    if (d["weekday"] > 4).any():
        raise ValueError("주말 행이 있습니다. C 표는 평일만 있어야 합니다.")
    clear = d["event_id"].isna() & d["impact"].isna()
    d["weather_group"] = np.where(d["impact"].notna(), "악천후", np.where(clear, "맑음", "약한 비·눈"))
    return d


def make_baseline(d: pd.DataFrame) -> pd.DataFrame:
    meta = d.drop_duplicates(["season", "road_id", "direction"])[
        ["season", "split", "road_id", "road_name", "function_type", "direction", "total_distance_m"]]
    clear = d[(d["weather_group"] == "맑음") & d["travel_time_seconds"].notna()]

    by_day = (clear.groupby(KEYS, observed=True)["travel_time_seconds"]
              .agg(tt_weekday="median", n_weekday="size").reset_index())
    by_all = (clear.groupby(["season", "road_id", "direction", "hour"], observed=True)["travel_time_seconds"]
              .agg(tt_all="median", n_all="size").reset_index())

    # 모든 시즌 × 도로 × 방향 × 요일 × 시간 칸을 만든 뒤 붙인다 (맑음이 0개인 칸도 남기기)
    grid = d[KEYS].drop_duplicates()
    b = (grid.merge(by_day, on=KEYS, how="left")
             .merge(by_all, on=["season", "road_id", "direction", "hour"], how="left"))
    b["n_weekday"] = b["n_weekday"].fillna(0).astype(int)
    use_day = b["n_weekday"] >= MIN_N
    b["baseline_travel_time_s"] = np.where(use_day, b["tt_weekday"], b["tt_all"])
    b["baseline_n"] = np.where(use_day, b["n_weekday"], b["n_all"]).astype("float")
    b["baseline_level"] = np.where(use_day, "weekday", np.where(b["tt_all"].notna(), "all_weekdays", "none"))
    b = b.merge(meta, on=["season", "road_id", "direction"], how="left")
    b["baseline_speed_kmh"] = b["total_distance_m"] / b["baseline_travel_time_s"] * 3.6
    b["weekday_name"] = b["weekday"].map(dict(enumerate(WEEKDAY_NAMES)))
    b["commute_period"] = np.where(b["hour"] <= 12, "morning", "evening")
    cols = ["season", "split", "road_id", "road_name", "function_type", "direction", "weekday", "weekday_name",
            "hour", "commute_period", "baseline_travel_time_s", "baseline_speed_kmh", "baseline_n",
            "baseline_level", "n_weekday", "total_distance_m"]
    return b[cols].sort_values(["season", "road_id", "direction", "weekday", "hour"]).reset_index(drop=True)


def add_ratio(d: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    out = d.merge(b[KEYS + ["baseline_travel_time_s"]], on=KEYS, how="left")
    out["ratio"] = out["travel_time_seconds"] / out["baseline_travel_time_s"]
    return out


def compare_candidates(r: pd.DataFrame) -> pd.DataFrame:
    """학습 시즌만: 후보마다 날씨 구분 · 도로 유형별 정체 비율과 '악천후 ÷ 맑음' 배수."""
    t = r[(r["split"] == "train") & r["ratio"].notna()].copy()
    rows = []
    for name, (desc, rule) in CANDIDATES.items():
        t["cong"] = rule(t)
        for ftype in ["전체", "도시고속도로", "주간선도로"]:
            s = t if ftype == "전체" else t[t["function_type"] == ftype]
            rate = s.groupby("weather_group")["cong"].mean()
            clear_days = s[s["weather_group"] == "맑음"]
            # 맑은 날에도 늘 막히는 도로 · 방향 (맑음 시간의 절반 이상 정체) 수
            always = (clear_days.groupby(["road_id", "direction"])["cong"].mean() >= 0.5).sum()
            rows.append({
                "candidate": name, "description": desc, "function_type": ftype,
                "rows": len(s), "positives": int(s["cong"].sum()),
                "rate_all": s["cong"].mean(),
                "rate_clear": rate.get("맑음", np.nan),
                "rate_light": rate.get("약한 비·눈", np.nan),
                "rate_severe": rate.get("악천후", np.nan),
                "lift_severe_vs_clear": rate.get("악천후", np.nan) / rate.get("맑음", np.nan),
                "road_dirs_congested_on_clear_half": int(always),
            })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="평소값 · 정체 기준 후보 (B)")
    p.add_argument("--road", type=Path, default=DEFAULT_ROAD, help="C 도로 통행시간 parquet (v03 · v04)")
    a = p.parse_args()

    d = load_road(a.road)
    b = make_baseline(d)
    r = add_ratio(d, b)

    for split in ("train", "eval"):
        out = PROC_DIR / split / f"baseline_{split}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        b[b["split"] == split].to_csv(out, index=False, encoding="utf-8-sig")
        print(f"저장: {out.relative_to(ROOT)} ({(b['split'] == split).sum():,}행)")

    cand = compare_candidates(r)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cand_path = TABLE_DIR / "congestion_threshold_candidates_train.csv"
    cand.to_csv(cand_path, index=False, encoding="utf-8-sig")
    print(f"저장: {cand_path.relative_to(ROOT)}")

    summary = {
        "run_at": datetime.now(KST).isoformat(timespec="seconds"),
        "input": a.road.name,
        "road_rows": len(d),
        "travel_time_nan_rows": int(d["travel_time_seconds"].isna().sum()),
        "baseline_cells": len(b),
        "baseline_level_counts": b["baseline_level"].value_counts().to_dict(),
        "ratio_nan_rows": int(r["ratio"].isna().sum()),
        "clear_ratio_median_train": float(r.loc[(r["split"] == "train") & (r["weather_group"] == "맑음"), "ratio"].median()),
        "min_n": MIN_N,
    }
    (ROOT / "logs" / "ljh").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs" / "ljh" / "baseline.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
