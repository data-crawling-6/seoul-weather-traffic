"""
입력용 평소값 prior (A 담당)

예측 모델 정답 y의 분모와 입력 비율에 쓰는 평소값. 예측하는 날 d에 알 수 있는 값만 쓴다.
규칙은 노션 '분석 설계 3.4' (11/25 A · B 합의).

- 기준일 d마다 d-28 ~ d-1 (당일 제외)
- 맑음(event_id · impact 없음) · 평일(공휴일 제외, 월~금 묶음) · 같은 시간대의 중앙값
- 링크 단위와 도로 × 방향 단위. 도로 통행시간 = 그 시각 링크 통행시간 합,
  링크가 하나라도 빠진 시각은 빈값으로 두고 표본에서 뺀다 (C v04와 같은 규칙)
- 24시간 전부 계산 (모델 입력이 t · t-1 · t-2 비율이라)
- 표본이 MIN_N 미만이면 지난 시즌(같은 계절 1년 전) 정답용 평소값으로 대신한다.
  정답용은 요일별 · 출퇴근 시간만 있어서 d와 같은 요일 · 같은 시간 값을 쓰고,
  없으면(첫 시즌 · 출퇴근 외 시간) 있는 표본으로 계산한다
- 결과는 연구 기간 안 평일(공휴일 제외) 날짜만. 평소값 구간(앞 4주)은 계산 재료로만 쓴다

prior_source: window28 (28일 표본 MIN_N 이상) / prev_season / short_window (표본 부족, 있는 것만)

입력
  data/raw/traffic/yang_seoyeon_topis_weather_link_hourly_20261122_v02.parquet (C 릴리스 v3)
  data/processed/{train,eval}/link_baseline_*.parquet, baseline_*.csv (B 정답용 평소값)

실행
  python src/prior_baseline.py
  python src/prior_baseline.py --seasons 2024-25_winter     # 시즌 하나만 시험

결과
  data/processed/{train,eval}/prior_link_{split}.parquet
  data/processed/{train,eval}/prior_road_{split}.parquet
  logs/kmj/prior_baseline.json
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LINK = ROOT / "data" / "raw" / "traffic" / "yang_seoyeon_topis_weather_link_hourly_20261122_v02.parquet"
PROC_DIR = ROOT / "data" / "processed"
WINDOW_DAYS = 28
MIN_N = 8

COLUMNS = ["link_id", "road_id", "direction", "date", "hour", "season", "split", "period_scope",
           "is_weekday", "is_holiday", "link_travel_time_seconds", "event_id", "impact"]


def load_periods() -> pd.DataFrame:
    y = yaml.safe_load((ROOT / "config" / "periods.yaml").read_text(encoding="utf-8"))
    p = pd.DataFrame(y["periods"])
    p["kind"] = p["season"].str.split("_").str[-1]
    p = p.sort_values("start")
    p["prev_season"] = p.groupby("kind")["season"].shift(1)  # 같은 계절의 1년 전
    return p


def load_prev_baselines() -> tuple[pd.DataFrame, pd.DataFrame]:
    link = pd.concat([pd.read_parquet(PROC_DIR / s / f"link_baseline_{s}.parquet") for s in ["train", "eval"]])
    road = pd.concat([pd.read_csv(PROC_DIR / s / f"baseline_{s}.csv", encoding="utf-8-sig") for s in ["train", "eval"]])
    link = link[["season", "link_id", "weekday", "hour", "baseline_link_tt_s"]].rename(
        columns={"baseline_link_tt_s": "prev_tt_s"})
    road = road[["season", "road_id", "direction", "weekday", "hour", "baseline_travel_time_s"]].rename(
        columns={"baseline_travel_time_s": "prev_tt_s"})
    link["link_id"] = link["link_id"].astype(str)
    return link, road


def load_season(path: Path, season: str) -> pd.DataFrame:
    t = ds.dataset(path).to_table(columns=COLUMNS, filter=ds.field("season") == season)
    d = t.to_pandas()
    for c in ["link_id", "road_id", "direction"]:
        d[c] = d[c].astype(str)
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["ok_day"] = d["is_weekday"] & ~d["is_holiday"]
    d["clear"] = d["event_id"].isna() & d["impact"].isna()
    return d


def road_hourly(d: pd.DataFrame) -> pd.DataFrame:
    """도로 × 방향 × 날짜 × 시간 통행시간. 링크가 하나라도 빠지면 NaN."""
    n_links = d.groupby(["road_id", "direction"])["link_id"].nunique().rename("link_count")
    g = (d.groupby(["road_id", "direction", "date", "hour"], observed=True)
         .agg(tt=("link_travel_time_seconds", "sum"), n_obs=("link_travel_time_seconds", "count"),
              ok_day=("ok_day", "first"), clear=("clear", "first"), period_scope=("period_scope", "first"))
         .reset_index().join(n_links, on=["road_id", "direction"]))
    g.loc[g["n_obs"] < g["link_count"], "tt"] = np.nan
    return g.drop(columns=["n_obs", "link_count"])


def rolling_prior(d: pd.DataFrame, keys: list[str], tt_col: str) -> pd.DataFrame:
    """keys × hour마다 d-28 ~ d-1 맑은 평일 표본의 중앙값 · 개수를 연구 기간 평일 날짜에 대해 구한다."""
    day0 = d["date"].min()
    n_days = (d["date"].max() - day0).days + 1
    d = d.assign(day=(d["date"] - day0).dt.days)

    grp = d[keys + ["hour"]].drop_duplicates().reset_index(drop=True)
    grp["g"] = np.arange(len(grp))
    d = d.merge(grp, on=keys + ["hour"])

    # 표본만 남긴 [그룹 × 날짜] 표. 표본이 아니면 NaN
    sample = d[d["ok_day"] & d["clear"] & d[tt_col].notna()]
    grid = np.full((len(grp), n_days), np.nan, dtype="float64")
    grid[sample["g"].to_numpy(), sample["day"].to_numpy()] = sample[tt_col].to_numpy()

    targets = np.sort(d.loc[(d["period_scope"] == "in_period") & d["ok_day"], "day"].unique())
    out = []
    for i in targets:
        w = grid[:, max(0, i - WINDOW_DAYS):i]
        n = np.sum(~np.isnan(w), axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # 표본 0개면 NaN
            med = np.nanmedian(w, axis=1)
        out.append(pd.DataFrame({"g": grp["g"].to_numpy(), "day": i, "window_tt_s": med, "prior_n": n}))
    r = pd.concat(out, ignore_index=True).merge(grp, on="g").drop(columns="g")
    r["date"] = day0 + pd.to_timedelta(r["day"], unit="D")
    return r.drop(columns="day")


def apply_fallback(r: pd.DataFrame, prev: pd.DataFrame, keys: list[str], prev_season: str | None) -> pd.DataFrame:
    r["weekday"] = r["date"].dt.weekday.astype("int64")
    if prev_season is not None:
        p = prev[prev["season"] == prev_season].drop(columns="season")
        r = r.merge(p, on=keys + ["weekday", "hour"], how="left")
    else:
        r["prev_tt_s"] = np.nan
    enough = r["prior_n"] >= MIN_N
    use_prev = ~enough & r["prev_tt_s"].notna()
    r["prior_tt_s"] = np.where(use_prev, r["prev_tt_s"], r["window_tt_s"])
    r["prior_source"] = np.select([enough, use_prev], ["window28", "prev_season"], "short_window")
    return r.drop(columns=["weekday", "prev_tt_s", "window_tt_s"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", type=Path, default=DEFAULT_LINK)
    ap.add_argument("--seasons", nargs="*", help="일부 시즌만 계산 (시험용). 결과 파일은 쓰지 않는다")
    a = ap.parse_args()

    periods = load_periods()
    prev_link, prev_road = load_prev_baselines()
    seasons = a.seasons or periods["season"].tolist()

    link_out, road_out, summary = [], [], {}
    for season in seasons:
        row = periods.set_index("season").loc[season]
        prev_season = row["prev_season"] if pd.notna(row["prev_season"]) else None
        d = load_season(a.link, season)

        lk = rolling_prior(d, ["link_id"], "link_travel_time_seconds")
        lk = lk.merge(d[["link_id", "road_id", "direction"]].drop_duplicates(), on="link_id")
        lk = apply_fallback(lk, prev_link, ["link_id"], prev_season)

        rd = rolling_prior(road_hourly(d), ["road_id", "direction"], "tt")
        rd = apply_fallback(rd, prev_road, ["road_id", "direction"], prev_season)

        for r in (lk, rd):
            r["season"] = season
            r["split"] = row["split"]
        link_out.append(lk)
        road_out.append(rd)
        summary[season] = {
            "split": row["split"], "prev_season": prev_season,
            "dates": int(lk["date"].nunique()),
            "link_rows": len(lk), "road_rows": len(rd),
            "link_source": lk["prior_source"].value_counts().to_dict(),
            "road_source": rd["prior_source"].value_counts().to_dict(),
            "link_prior_nan": int(lk["prior_tt_s"].isna().sum()),
            "road_prior_nan": int(rd["prior_tt_s"].isna().sum()),
        }
        print(season, json.dumps(summary[season], ensure_ascii=False))
        del d

    if a.seasons:
        return

    cols_link = ["season", "split", "date", "hour", "link_id", "road_id", "direction", "prior_tt_s", "prior_n", "prior_source"]
    cols_road = ["season", "split", "date", "hour", "road_id", "direction", "prior_tt_s", "prior_n", "prior_source"]
    link_all = pd.concat(link_out, ignore_index=True)[cols_link]
    road_all = pd.concat(road_out, ignore_index=True)[cols_road]
    for t in (link_all, road_all):
        t["hour"] = t["hour"].astype("int8")
        t["prior_n"] = t["prior_n"].astype("int16")
        t["prior_tt_s"] = t["prior_tt_s"].astype("float32")
        for c in ["season", "split", "prior_source"]:
            t[c] = t[c].astype("category")
    link_all["link_id"] = link_all["link_id"].astype("category")
    for t in (link_all, road_all):
        for c in ["road_id", "direction"]:
            t[c] = t[c].astype("category")

    for split in ["train", "eval"]:
        out_dir = PROC_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)
        lk = link_all[link_all["split"] == split].sort_values(["season", "link_id", "date", "hour"])
        rd = road_all[road_all["split"] == split].sort_values(["season", "road_id", "direction", "date", "hour"])
        lk.to_parquet(out_dir / f"prior_link_{split}.parquet", index=False)
        rd.to_parquet(out_dir / f"prior_road_{split}.parquet", index=False)

    log = {"run_at": datetime.now(KST).isoformat(timespec="seconds"), "input": a.link.name,
           "window_days": WINDOW_DAYS, "min_n": MIN_N, "seasons": summary}
    (ROOT / "logs" / "kmj").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs" / "kmj" / "prior_baseline.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
