"""
악천후 모드 발동 기준 (B 담당, 노션 일정 '12/2 악천후 모드 발동 기준 → C', 결과 4.3)

무엇을 정하나
  평일 출퇴근 목표 시각(08 · 09 · 18 · 19 · 20시)마다, 미리(1 · 3시간 전) '악천후 모드'를 켤지 정하는 규칙.
  모드가 켜지면 C의 지도 · 서비스 화면이 도로별 예상 지연(A)과 정체 경보(B)를 보여 준다.
  서울 전체에 한 번에 켜는 규칙이다 (예보가 서울 4개 권역 값이라, 도로별로 켜는 건 다음 단계).

입력 (그 시점에 알 수 있던 것만)
  예보  fcst_seoul: 목표 시각을 예보한 발표 가운데 발표 시각 + 1시간 <= t 인 가장 최근 것 (경보 모델과 같은 규칙)
        pcp_max (권역 최대 1시간 강수 mm) · sno_max (권역 최대 1시간 신적설 cm) · 비 · 눈 권역 수
  관측  weather_hourly 의 t 시점 강수 (지금 내리고 있나, 기준선 비교용)
  특보  t 시점 대설 · 호우특보 발효 (기준선 비교용)

정답 (모드가 켜졌어야 하는 시간)
  목표 시각에 악천후가 '내리는 중' (impact 있음 + 비는 관측 시간 · 눈은 snow_phase == falling).
  그친 뒤 영향 시간은 오히려 빨라지는 시간이 섞여서 정답에서 뺀다 (24시간 비교 5-2, 노션 10.2 (2-1)).
  함께 보는 것: 모드가 켜진 시간에 실제로 정체가 많았나 (서울 정체 비율, 켜짐 vs 꺼짐).

규칙 후보와 고르는 기준 (돌리기 전에 정함)
  후보 A (예보만): 예보 강수 pcp_max >= p (p = 0.5 · 1 · 2 · 3 · 5 · 10) 또는 예보 눈(sno_max > 0 또는 눈 권역 >= 1)
  후보 B (예보 + 지금 관측): 후보 A 또는 t 시점에 비 · 눈이 관측됨 (예보가 놓친 비를 관측으로 보탠다)
  같은 성적이면 p 가 큰 쪽(덜 자주 켜짐).
  Train 으로 p 를 고른다: 재현율 80% 이상인 후보 가운데 정밀도가 가장 높은 것 (경보 기준과 같은 원칙).
  80%를 넘는 후보가 없으면 재현율이 가장 높은 후보 (같으면 정밀도 · p 큰 쪽).
  Dev 로 확인한다. Test 는 열지 않는다.
  함께 적는 것: 재현율 · 정밀도 · 한 시즌(평일) 켜진 시간 수 · 헛켜짐 수, 눈 · 비 따로

실행
  python src/analysis/trigger_rule.py
결과
  outputs/tables/trigger_rule_candidates.csv     후보 × h × Train / Dev 성능
  outputs/tables/trigger_rule_chosen.csv         고른 규칙의 Train · Dev 성능 (눈 · 비 따로)
  outputs/tables/trigger_rule_effect.csv         모드 켜짐 / 꺼짐 시간의 서울 정체 비율 · 평균 통행시간 비율
  data/shared/train_dev/trigger_hours_train_dev.csv  날짜 × 목표 시각 × h 의 모드 켜짐 여부 (C 지도 · 시제품용)
  logs/ljh/trigger_rule/trigger_rule.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.io_utils import load_processed  # noqa: E402

KST = timezone(timedelta(hours=9))
TABLE_DIR = ROOT / "outputs" / "tables"
OUT_PATH = ROOT / "data" / "shared" / "train_dev" / "trigger_hours_train_dev.csv"
LOG_PATH = ROOT / "logs" / "ljh" / "trigger_rule" / "trigger_rule.json"
TARGET_HOURS = [8, 9, 18, 19, 20]
HORIZONS = [1, 3]
PCP_CANDIDATES = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
MIN_RECALL = 0.80


def target_grid() -> pd.DataFrame:
    """연구 기간 평일(공휴일 제외) × 목표 시각. 정답 · t 시점 관측 · 특보를 붙인다."""
    w = load_processed("weather_hourly")
    w["datetime"] = pd.to_datetime(w["datetime"])
    w = w.set_index("datetime").sort_index()
    import holidays
    hol = holidays.KR(years=range(2022, 2027))
    tgt = w[w["in_period"] & w["hour"].isin(TARGET_HOURS)].copy()
    tgt = tgt[(tgt.index.dayofweek < 5) & ~pd.Index(tgt.index.date).isin(hol)]
    snow = tgt["snow_phase"].eq("falling")
    rain = tgt["impact"].notna() & tgt["snow_phase"].isna() & tgt["event_id"].notna()
    out = pd.DataFrame({"datetime": tgt.index, "date": tgt.index.normalize(), "target_hour": tgt["hour"].to_numpy(),
                        "season": tgt["season"].to_numpy(), "model_set": tgt["model_set"].to_numpy(),
                        "truth": (snow | rain).to_numpy(),
                        "truth_kind": np.where(snow, "눈", np.where(rain, "비", "")),
                        "impact": tgt["impact"].to_numpy()})
    rows = []
    for h in HORIZONS:
        o = out.copy()
        o["horizon"] = h
        t = o["datetime"] - pd.to_timedelta(h, unit="h")
        o["obs_precip_t"] = w["precip_mm"].reindex(t).to_numpy()
        o["warn_t"] = ((w["warn_snow"].reindex(t).fillna(0) > 0) | (w["warn_rain"].reindex(t).fillna(0) > 0)).to_numpy()
        rows.append(o)
    return pd.concat(rows, ignore_index=True)


def add_forecast(g: pd.DataFrame) -> pd.DataFrame:
    """목표 시각 예보 가운데 발표 + 1시간 <= t 인 가장 최근 발표 (경보 모델 alert_model.add_forecast 와 같은 규칙)."""
    fc = load_processed("fcst_seoul")
    fc["base"], fc["fcst"] = pd.to_datetime(fc["base"]), pd.to_datetime(fc["fcst"])
    g = g.copy()
    g["cutoff"] = g["datetime"] - pd.to_timedelta(g["horizon"] + 1, unit="h")
    m = pd.merge_asof(g.sort_values("cutoff"), fc.sort_values("base")[
        ["fcst", "base", "lead_h", "pcp_max", "sno_max", "pty_rain_regions", "pty_snow_regions"]],
        left_on="cutoff", right_on="base", left_by="datetime", right_by="fcst", direction="backward")
    if (m["base"] > m["cutoff"]).any():
        raise RuntimeError("t 시점 뒤에 나온 예보가 들어갔습니다.")
    m["fc_missing"] = m["base"].isna() | m["pcp_max"].isna()
    return m.drop(columns=["fcst", "cutoff"]).sort_values(["horizon", "datetime"]).reset_index(drop=True)


def rule_on(g: pd.DataFrame, p: float, with_obs: bool = False) -> np.ndarray:
    """예보 강수 pcp_max >= p 또는 예보 눈(신적설 > 0 또는 눈 권역 >= 1). with_obs 면 t 시점 관측 강수도 보탠다."""
    rain = g["pcp_max"].fillna(0) >= p
    snow = (g["sno_max"].fillna(0) > 0) | (g["pty_snow_regions"].fillna(0) >= 1)
    on = rain | snow
    if with_obs:
        on = on | g["obs_precip_t"].fillna(0).gt(0)
    return on.to_numpy()


def score(g: pd.DataFrame, on: np.ndarray) -> dict:
    y = g["truth"].to_numpy()
    tp, fp, fn = int((on & y).sum()), int((on & ~y).sum()), int((~on & y).sum())
    seasons = g["season"].nunique()
    out = {"recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1), "tp": tp, "fp": fp, "fn": fn,
           "on_hours_per_season": (tp + fp) / max(seasons, 1), "false_on_per_season": fp / max(seasons, 1)}
    for kind in ["눈", "비"]:
        k = g["truth_kind"].eq(kind).to_numpy()
        out[f"recall_{kind}"] = float((on & k).sum() / max(k.sum(), 1))
    return out


def main() -> None:
    g = add_forecast(target_grid())
    rows = []
    for h in HORIZONS:
        for ms in ["train", "dev"]:
            part = g[(g["horizon"] == h) & (g["model_set"] == ms)]
            for p in PCP_CANDIDATES:
                for with_obs in [False, True]:
                    name = f"예보 강수 >= {p}mm 또는 예보 눈" + (" 또는 지금 관측" if with_obs else "")
                    rows.append({"rule": name, "pcp": p, "with_obs": with_obs, "horizon": h, "model_set": ms,
                                 **score(part, rule_on(part, p, with_obs))})
            rows.append({"rule": "기준선: 지금 비 · 눈 관측", "pcp": np.nan, "horizon": h, "model_set": ms,
                         **score(part, part["obs_precip_t"].fillna(0).gt(0).to_numpy())})
            rows.append({"rule": "기준선: 특보 발효 중", "pcp": np.nan, "horizon": h, "model_set": ms,
                         **score(part, part["warn_t"].to_numpy())})
    cand = pd.DataFrame(rows)

    # Train 에서 고른다: 재현율 80% 이상 중 정밀도 최대 (없으면 재현율 최대). h 마다
    chosen = {}
    for h in HORIZONS:
        c = cand[(cand["horizon"] == h) & (cand["model_set"] == "train") & cand["pcp"].notna()]
        ok = c[c["recall"] >= MIN_RECALL]
        pool = ok.sort_values(["precision", "pcp"], ascending=False) if len(ok) else c.sort_values(["recall", "precision", "pcp"], ascending=False)
        best = pool.iloc[0]
        chosen[h] = (float(best["pcp"]), bool(best["with_obs"]))
    cand["chosen"] = [r["pcp"] == chosen[r["horizon"]][0] and r.get("with_obs") == chosen[r["horizon"]][1]
                      for _, r in cand.iterrows()]

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cand.to_csv(TABLE_DIR / "trigger_rule_candidates.csv", index=False, encoding="utf-8-sig")
    keep = cand[cand["chosen"] | cand["pcp"].isna()]
    keep.to_csv(TABLE_DIR / "trigger_rule_chosen.csv", index=False, encoding="utf-8-sig")

    g["mode_on"] = [bool(x) for x in np.concatenate([rule_on(g[g["horizon"] == h], *chosen[h]) for h in HORIZONS])]
    out = g[["date", "target_hour", "horizon", "model_set", "base", "lead_h", "pcp_max", "sno_max",
             "pty_rain_regions", "pty_snow_regions", "fc_missing", "mode_on", "truth", "truth_kind"]].copy()
    out = out.rename(columns={"base": "forecast_base", "lead_h": "forecast_lead_h", "truth": "bad_weather_falling",
                              "truth_kind": "bad_weather_kind"})
    # 켜진 시간에 실제로 더 막혔나: 서울 정체 비율(180개 도로 · 방향 중 정체 비율, 24시간 표)을 켜짐 / 꺼짐으로
    from src.analysis import hourly_24h as H24
    if H24.out_path("train_dev").exists():
        hh = pd.read_parquet(H24.out_path("train_dev"), columns=["date", "hour", "is_congested", "ratio"])
        city = hh.groupby(["date", "hour"]).agg(city_congested=("is_congested", "mean"), city_ratio=("ratio", "mean")).reset_index()
        city["date"] = pd.to_datetime(city["date"])
        out = out.merge(city.rename(columns={"hour": "target_hour"}), on=["date", "target_hour"], how="left")
        eff = (out.groupby(["horizon", "model_set", "mode_on"])[["city_congested", "city_ratio"]].mean()
               .reset_index())
        eff.to_csv(TABLE_DIR / "trigger_rule_effect.csv", index=False, encoding="utf-8-sig")
        print(eff.round(4).to_string(index=False))
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    log = {"run_at": datetime.now(KST).isoformat(timespec="seconds"),
           "chosen": {h: {"pcp_mm": p, "with_observation": o} for h, (p, o) in chosen.items()},
           "rule": "예보 강수(권역 최대 1시간) >= p mm 또는 예보 눈(신적설 > 0 또는 눈 권역 >= 1), 발표 + 1시간 <= t",
           "rows": len(out), "forecast_missing_rate": round(float(g["fc_missing"].mean()), 4),
           "truth_rate": {ms: round(float(g.loc[g["model_set"] == ms, "truth"].mean()), 4) for ms in ["train", "dev"]}}
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(keep.round(3).to_string(index=False))
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
