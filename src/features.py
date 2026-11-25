"""
예측 모델 입력 변수 (A 담당)

시각 t에 h시간 뒤(1 · 3시간) 도로 × 방향 통행시간이 입력용 평소값의 몇 배인지 예측한다.
노션 '분석 설계 8' 기준. 입력은 t 시점에 알 수 있는 값만 쓴다.

한 행 = 도로 × 방향 × 날짜 × 목표 시각(출퇴근 8 · 9 · 18 · 19 · 20시) × 예측 거리 h
  t = 목표 시각 - h. 시간은 모두 구간 끝 시각 (8 = 07:00~08:00)

정답
  y_ratio           목표 시각 통행시간 ÷ 목표 날짜 입력용 평소값 (prior_road, 분석 설계 8.1)
  tt_target_s       목표 시각 통행시간 (초). 링크가 하나라도 빠지면 빈값 -> y도 빈값

입력 (분석 설계 8.2)
  (1) 도로       road_id, direction, function_type, n_links, total_distance_m, lanes(거리 가중 평균),
                urban_share(도심권 거리 비율), is_riverside(올림픽대로 · 강변북로 · 동부간선도로 · 내부순환로)
  (2) 시간       target_hour, horizon, weekday, month, season_kind, is_vacation_peak
  (3) 최근 상태  ratio_t0 · ratio_t1 · ratio_t2 (t · t-1 · t-2 시점 비율), *_missing,
                city_ratio_t0 · type_ratio_t0 (서울 전체 · 같은 기능유형 도로의 t 시점 비율 중앙값)
  (4) 관측 날씨  서울 ASOS t 시점: temp_c, temp_drop_3h, precip_1h, precip_3h, new_snow_3h_cm,
                snow_depth_cm, precip_type
                한강 수위(하천변 도로만): wl_rise_3h, wl_over_attention, wl_over_warning (주 관측소),
                wl2_* (보조 관측소, 동부간선도로만 중랑교). 절대 수위는 관측소마다 기준 높이가 달라 쓰지 않는다
  (5) 특보       warn_snow · warn_rain · warn_cold (t에 발효 중인 서울 권역 수),
                *_upcoming(발표됐지만 발효 전), *_hours(발효 후 지난 시간)

평가용 (입력 금지)
  weather_class(목표 시각 맑음 / 약한 비 · 눈 / 악천후), weather_kind(눈 / 비),
  prior_replaced_target, model_set(train / dev / test)

넣지 않은 것
  event_id · impact: 사건이 끝난 뒤에 정해지는 값이라 입력으로 쓰면 누수 -> 평가 구분용으로만
  구별 AWS · 예보((6)): B가 연구 기간 평일 전체로 다시 받는 중 (노션 8.2, 4단계까지는 빼고 돌린다)
  권역: 도로별 권역은 C 링크 좌표가 나온 뒤에 붙인다 -> 그 전까지 city/type 비율로 대신

입력
  data/raw/traffic/yang_seoyeon_topis_weather_link_hourly_20261122_v02.parquet (24시간 통행시간)
  data/processed/{train,eval}/prior_road_*.parquet, weather_hourly_*.csv, waterlevel_hourly_*.csv
  data/raw/topis/csv/topis_speed_2025-12.csv (링크 차선 수 · 도심/외곽)

실행
  python src/features.py
결과
  data/processed/{train,eval}/features_{split}.parquet
  logs/kmj/features.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import read_topis  # noqa: E402
from src.prior_baseline import DEFAULT_LINK, load_periods, load_season, road_hourly  # noqa: E402

KST = timezone(timedelta(hours=9))
PROC_DIR = ROOT / "data" / "processed"
TARGET_HOURS = [8, 9, 18, 19, 20]
HORIZONS = [1, 3]
LINK_ATTR_MONTH = "2025-12"  # roads.csv metadata_reference_month와 같은 달

# 휴가 절정 주간 (노션 8.2, 11/25 확정)
VACATION_PEAK = ("07-21", "08-15")

# 노션 2.2 (4)(3) · 8.2 하천변 도로 -> 수위 관측소 [주, 보조]
# - 동부간선도로: 서울시 통제 기준은 중랑천 월계1교인데 자료에 없다. 바로 위 상류 월계2교를 주로 쓰고
#   (상류가 먼저 올라 신호가 빠르다) 중랑교를 보조로 넣는다. 월계2교는 2023~2026 여름 빈값 0%
# - 내부순환로: 성동JC~사근~마장 구간이 중랑천 옆이라 2022 · 2024년 통제됨. 2023~2025 여름 호우 때
#   이 구간 4개 링크만 속도 빈 비율이 3~4%(다른 구간 0.4%), 2024-07-18에는 5~10시간 비어 있었다 (11/25 확인)
# 관측소마다 기준 높이가 달라 절대 수위는 쓰지 않고, 3시간 상승폭과 관심 · 주의 수위 초과만 쓴다
RIVERSIDE_STATIONS = {
    "올림픽대로": ["서울시(한강대교)"],
    "강변북로": ["서울시(한강대교)"],
    "동부간선도로": ["서울시(월계2교)", "서울시(중랑교)"],
    "내부순환로": ["서울시(중랑교)"],
}

# 노션 7.2 (11/25). B의 add_model_set()이 생기면 그걸로 바꾼다
MODEL_SETS = {
    "2022-23_winter": "train", "2023-24_winter": "train", "2023_summer": "train", "2024_summer": "train",
    "2024-25_winter": "dev", "2025_summer": "dev",
    "2025-26_winter": "test", "2026_summer": "test",
}


def road_static(links: pd.DataFrame) -> pd.DataFrame:
    """링크 목록(road_id · direction · link_id · distance_m) + TOPIS 링크 속성 -> 도로 × 방향 정보"""
    raw = read_topis(LINK_ATTR_MONTH)
    attr = (raw.assign(link_id=raw["링크아이디"].astype(str))
            .drop_duplicates("link_id")[["link_id", "차선수", "도심/외곽구분"]])
    d = links.merge(attr, on="link_id", how="left")
    d["urban_m"] = d["distance_m"] * (d["도심/외곽구분"] == "도심권")
    d["lane_m"] = d["distance_m"] * d["차선수"]
    g = d.groupby(["road_id", "direction"]).agg(
        n_links=("link_id", "nunique"), total_distance_m=("distance_m", "sum"),
        lane_m=("lane_m", "sum"), urban_m=("urban_m", "sum")).reset_index()
    g["lanes"] = g["lane_m"] / g["total_distance_m"]
    g["urban_share"] = g["urban_m"] / g["total_distance_m"]
    roads = pd.read_csv(ROOT / "config" / "roads.csv", encoding="utf-8-sig")[["road_id", "road_name", "function_type"]]
    g = g.merge(roads, on="road_id", how="left")
    g["is_riverside"] = g["road_name"].isin(RIVERSIDE_STATIONS)
    return g.drop(columns=["lane_m", "urban_m"])


def weather_table() -> pd.DataFrame:
    w = pd.concat([pd.read_csv(PROC_DIR / s / f"weather_hourly_{s}.csv", parse_dates=["datetime", "date"])
                   for s in ["train", "eval"]]).drop_duplicates("datetime").sort_values("datetime")
    w = w.set_index("datetime").asfreq("h")  # 빠진 시각이 있어도 shift가 정확히 1시간이 되게
    out = pd.DataFrame(index=w.index)
    out["temp_c"] = w["temp_c"]
    out["temp_drop_3h"] = w["temp_c"].shift(3) - w["temp_c"]          # 양수면 3시간 동안 떨어짐
    out["precip_1h"] = w["precip_mm"]
    out["precip_3h"] = w["precip_mm"].rolling(3, min_periods=1).sum()   # t-2 ~ t
    out["new_snow_3h_cm"] = w["new_snow_3h_cm"]
    out["snow_depth_cm"] = w["snow_depth_cm"]
    out["precip_type"] = w["precip_type"].fillna("none")
    for k in ["snow", "rain", "cold"]:
        for suf in ["", "_upcoming", "_hours"]:
            out[f"warn_{k}{suf}"] = w[f"warn_{k}{suf}"]
    # 평가용
    out["weather_class"] = np.select([w["impact"].notna(), w["event_id"].notna()], ["악천후", "약한 비·눈"], "맑음")
    label = w["impact"].fillna(w["event_intensity"]).fillna("")
    out["weather_kind"] = np.where(label.str.contains("눈"), "눈", np.where(label != "", "비", ""))
    return out.reset_index()


def water_table() -> pd.DataFrame:
    wl = pd.concat([pd.read_csv(PROC_DIR / s / f"waterlevel_hourly_{s}.csv", parse_dates=["datetime"])
                    for s in ["train", "eval"]])
    wl = wl[wl["obsnm"].isin({s for v in RIVERSIDE_STATIONS.values() for s in v})]
    return wl[["datetime", "obsnm", "wl_rise_3h", "over_attwl", "over_wrnwl"]].rename(
        columns={"over_attwl": "wl_over_attention", "over_wrnwl": "wl_over_warning"}).drop_duplicates(["datetime", "obsnm"])


def season_rows(season: str, prior: pd.DataFrame) -> pd.DataFrame:
    """그 시즌 연구 기간 평일 × 24시간 도로 통행시간과 입력용 평소값 대비 비율"""
    d = load_season(DEFAULT_LINK, season)
    rh = road_hourly(d)
    rh = rh[(rh["period_scope"] == "in_period") & rh["ok_day"]][["road_id", "direction", "date", "hour", "tt"]]
    del d
    p = prior[prior["season"] == season][["road_id", "direction", "date", "hour", "prior_tt_s", "prior_replaced"]]
    rh = rh.merge(p, on=["road_id", "direction", "date", "hour"], how="left")
    rh["ratio"] = rh["tt"] / rh["prior_tt_s"]
    return rh


def build(seasons: list[str]) -> tuple[pd.DataFrame, dict]:
    periods = load_periods().set_index("season")
    prior = pd.concat([pd.read_parquet(PROC_DIR / s / f"prior_road_{s}.parquet") for s in ["train", "eval"]])
    for c in ["season", "road_id", "direction"]:
        prior[c] = prior[c].astype(str)
    prior["date"] = pd.to_datetime(prior["date"])
    weather = weather_table()
    water = water_table()

    out, static, summary = [], None, {}
    for season in seasons:
        row = periods.loc[season]
        rh = season_rows(season, prior)
        if static is None:
            static = road_static(_links_with_distance(season))
        rh = rh.merge(static[["road_id", "direction", "function_type"]], on=["road_id", "direction"], how="left")

        key = ["road_id", "direction", "date", "hour"]
        ratio = rh.set_index(key)["ratio"]
        city = rh.groupby(["date", "hour"])["ratio"].median().rename("city_ratio_t0")
        typ = rh.groupby(["function_type", "date", "hour"])["ratio"].median().rename("type_ratio_t0")

        target = rh[rh["hour"].isin(TARGET_HOURS)].rename(columns={
            "hour": "target_hour", "tt": "tt_target_s", "prior_tt_s": "prior_target_s",
            "prior_replaced": "prior_replaced_target", "ratio": "y_ratio"})
        for h in HORIZONS:
            f = target.assign(horizon=h, t_hour=target["target_hour"] - h)
            for lag in [0, 1, 2]:
                idx = pd.MultiIndex.from_arrays([f["road_id"], f["direction"], f["date"], f["t_hour"] - lag])
                f[f"ratio_t{lag}"] = ratio.reindex(idx).to_numpy()
                f[f"ratio_t{lag}_missing"] = f[f"ratio_t{lag}"].isna()
            f = f.merge(city, left_on=["date", "t_hour"], right_index=True, how="left")
            f = f.merge(typ, left_on=["function_type", "date", "t_hour"], right_index=True, how="left")
            f["t_datetime"] = f["date"] + pd.to_timedelta(f["t_hour"], unit="h")
            f["target_datetime"] = f["date"] + pd.to_timedelta(f["target_hour"], unit="h")
            f = f.merge(weather.drop(columns=["weather_class", "weather_kind"]), left_on="t_datetime",
                        right_on="datetime", how="left").drop(columns="datetime")
            f = f.merge(weather[["datetime", "weather_class", "weather_kind"]], left_on="target_datetime",
                        right_on="datetime", how="left").drop(columns="datetime")
            out.append(f.assign(season=season, split=row["split"], model_set=MODEL_SETS[season]))
        summary[season] = {"targets": int(len(target)), "y_nan": int(target["y_ratio"].isna().sum())}
        print(season, summary[season])

    f = pd.concat(out, ignore_index=True)
    f = f.merge(static.drop(columns=["function_type"]), on=["road_id", "direction"], how="left")

    for i, prefix in [(0, "wl_"), (1, "wl2_")]:
        stations = f["road_name"].map(lambda r: RIVERSIDE_STATIONS.get(r, []))
        station = stations.map(lambda s: s[i] if len(s) > i else None)
        w = water.rename(columns={c: prefix + c[3:] for c in water.columns if c.startswith("wl_")})
        f = f.assign(station=station).merge(w, left_on=["t_datetime", "station"], right_on=["datetime", "obsnm"],
                                            how="left").drop(columns=["datetime", "obsnm", "station"])

    f["weekday"] = f["date"].dt.weekday.astype("int8")
    f["month"] = f["date"].dt.month.astype("int8")
    f["season_kind"] = f["season"].str.split("_").str[-1]
    md = f["date"].dt.strftime("%m-%d")
    f["is_vacation_peak"] = (md >= VACATION_PEAK[0]) & (md <= VACATION_PEAK[1])
    return f, summary


def _links_with_distance(season: str) -> pd.DataFrame:
    import pyarrow.dataset as ds
    t = ds.dataset(DEFAULT_LINK).to_table(columns=["road_id", "direction", "link_id", "distance_m"],
                                          filter=ds.field("season") == season).to_pandas()
    for c in ["road_id", "direction", "link_id"]:
        t[c] = t[c].astype(str)
    return t.groupby(["road_id", "direction", "link_id"], as_index=False)["distance_m"].median()


COLUMNS = [
    # 키 · 평가용
    "season", "split", "model_set", "date", "target_hour", "horizon", "t_hour", "road_id", "road_name", "direction",
    "y_ratio", "tt_target_s", "prior_target_s", "prior_replaced_target", "weather_class", "weather_kind",
    # (1) 도로
    "function_type", "n_links", "total_distance_m", "lanes", "urban_share", "is_riverside",
    # (2) 시간
    "weekday", "month", "season_kind", "is_vacation_peak",
    # (3) 최근 상태
    "ratio_t0", "ratio_t1", "ratio_t2", "ratio_t0_missing", "ratio_t1_missing", "ratio_t2_missing",
    "city_ratio_t0", "type_ratio_t0",
    # (4) 관측 날씨
    "temp_c", "temp_drop_3h", "precip_1h", "precip_3h", "new_snow_3h_cm", "snow_depth_cm", "precip_type",
    "wl_rise_3h", "wl_over_attention", "wl_over_warning", "wl2_rise_3h", "wl2_over_attention", "wl2_over_warning",
    # (5) 특보
    "warn_snow", "warn_snow_upcoming", "warn_snow_hours", "warn_rain", "warn_rain_upcoming", "warn_rain_hours",
    "warn_cold", "warn_cold_upcoming", "warn_cold_hours",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", help="일부 시즌만 (시험용, 파일은 쓰지 않음)")
    a = ap.parse_args()
    seasons = a.seasons or list(MODEL_SETS)
    f, summary = build(seasons)
    f = f[COLUMNS].sort_values(["season", "road_id", "direction", "date", "target_hour", "horizon"])
    if a.seasons:
        print(f.head(3).T.to_string())
        return
    for c in ["season", "split", "model_set", "road_id", "road_name", "direction", "function_type",
              "season_kind", "precip_type", "weather_class", "weather_kind"]:
        f[c] = f[c].astype("category")
    for split in ["train", "eval"]:
        f[f["split"] == split].to_parquet(PROC_DIR / split / f"features_{split}.parquet", index=False)
    log = {"run_at": datetime.now(KST).isoformat(timespec="seconds"), "rows": len(f),
           "target_hours": TARGET_HOURS, "horizons": HORIZONS, "vacation_peak": VACATION_PEAK,
           "riverside_stations": RIVERSIDE_STATIONS, "seasons": summary}
    (ROOT / "logs" / "kmj").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs" / "kmj" / "features.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
