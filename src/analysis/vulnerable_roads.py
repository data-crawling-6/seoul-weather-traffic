"""
반복 취약 도로 목록 (B 담당, 노션 분석 설계 10.2 · 9.2 ⑦)

정의
- 사건 비율  : 악천후 사건(impact 있는 시간)이 평일 출퇴근 시간에 걸친 시간마다
               도로 · 방향 통행시간 ÷ 정답용 평소값(baseline_*.csv)을 구하고, 그 사건 안의 중앙값을 낸다
- 상위 20%   : 그 사건에서 사건 비율이 있는 도로 · 방향(최대 180개) 중 백분위 순위 80% 이상
- 반복 취약  : 사건 가운데 절반 이상에서 상위 20%에 들고, 사건 비율 중앙값이 최소 비율 이상인 도로 · 방향
- 눈 · 비는 따로 판정한다 (weather_events 의 type)

진행 (분석 설계 7.3)
1. Train 사건으로 목록 후보를 만든다 (최소 비율 후보마다)
2. Dev 사건에서 '목록이 상위 20%에 드는 비율'(적중률)을 보고 최소 비율을 고른다
   - 무작위 기댓값 20% 대비 배수(lift)가 가장 큰 후보, 목록이 MIN_LIST 개 미만인 후보는 뺀다, 같으면 목록이 큰 쪽
   - 순열검정 p: 같은 개수를 무작위로 골랐을 때 적중률이 이만큼 이상일 확률
3. 고른 최소 비율로 Train + Dev 사건 전체로 다시 만든다 → C 에게 넘기는 최종 목록
4. Test 확인(학습 3년 목록이 Test 시즌에서도 상위 20%에 드나)은 check_test() 로 마지막에 한 번만 (기본 실행에서 하지 않음)
   '반복된다' = lift 2배 이상 그리고 순열검정 p < 0.05 (분석 설계 10.2 (3))

보조 확인
- 맑은 날 대조 (load_ratios(control=True)): 사건마다 같은 시즌 · 같은 요일 · 같은 시각을 1~6주 옮겨 모두 맑은 시간을
  '가짜 사건'으로 만든다. 날씨 목록이 맑은 날에도 상위 20%에 많이 들면, 날씨 때문이 아니라 원래 흔들리는 도로라는 뜻이다
- 시즌 하나씩 빼기 (leave_one_season_out): 한 시즌을 빼고 나머지로 목록을 만든 뒤 뺀 시즌에서 확인 (7.3 (5))

입력
  data/derived/topis_weather/commute_20261122_v04.parquet  (C 도로 통행시간 표, 팀 공유 드라이브)
  data/shared/train_dev/baseline_train_dev.csv · weather_events_train_dev.csv · weather_hourly (날씨 판정)
실행
  python src/analysis/vulnerable_roads.py
결과
  data/shared/train_dev/vulnerable_roads_train_dev.csv          최종 목록 (Train + Dev 로 만든 것, 전 도로 · 방향 행 + is_vulnerable)
  outputs/tables/vulnerable_event_ratios_train_dev.csv           사건 × 도로 · 방향 비율 · 순위
  outputs/tables/vulnerable_min_ratio_dev.csv                    최소 비율 후보별 Dev 적중률 · lift · 순열 p
  outputs/tables/vulnerable_clear_control.csv                    맑은 날 대조
  outputs/tables/vulnerable_leave_one_season_out.csv             시즌 하나씩 빼기
  logs/ljh/vulnerable_roads/vulnerable_roads.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.io_utils import add_model_set, attach_weather_labels  # noqa: E402

ROAD = ROOT / "data" / "derived" / "topis_weather" / "commute_20261122_v04.parquet"
SHARED = ROOT / "data" / "shared"
TABLE_DIR = ROOT / "outputs" / "tables"
LOG_DIR = ROOT / "logs" / "ljh" / "vulnerable_roads"

TOP_SHARE = 0.20          # 상위 20%
HIT_SHARE = 0.50          # 사건 절반 이상
MIN_RATIO_CANDIDATES = [1.00, 1.02, 1.05, 1.08, 1.10, 1.15, 1.20]
MIN_LIST = 5              # 목록이 이보다 작으면 최소 비율 후보에서 뺀다
N_PERM = 10000
SEED = 20261127
KIND_NAME = {"snow": "눈", "rain": "비"}


def _split_dir(split: str) -> str:
    return {"train": "train_dev", "eval": "holdout"}[split]


CONTROL_WEEKS = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6]


def _control_hours(v: pd.DataFrame) -> pd.DataFrame:
    """사건마다 악천후 출퇴근 시간을 k주 옮긴 '맑은 날 가짜 사건' (같은 시즌 · 요일 · 시각, 모두 맑음인 가장 가까운 주)."""
    v = v.copy()
    v["date"] = pd.to_datetime(v["date"])
    hours = v[["date", "hour", "season", "event_id", "impact"]].drop_duplicates(["date", "hour"])
    clear = set(map(tuple, hours.loc[hours["event_id"].isna() & hours["impact"].isna(), ["date", "hour"]].to_numpy()))
    season_of = dict(zip(zip(hours["date"], hours["hour"]), hours["season"]))
    rows = []
    for ev, g in hours[hours["impact"].notna()].groupby("event_id"):
        pairs = list(zip(g["date"], g["hour"]))
        for k in CONTROL_WEEKS:
            moved = [(d + pd.Timedelta(7 * k, unit="D"), hr) for d, hr in pairs]
            if all(m in clear and season_of.get(m) == season_of[p] for m, p in zip(moved, pairs)):
                rows += [{"date": m[0], "hour": m[1], "control_event_id": ev, "shift_weeks": k} for m in moved]
                break
    return pd.DataFrame(rows)


def load_ratios(split: str = "train", control: bool = False) -> pd.DataFrame:
    """v04 평일 출퇴근 + 정답용 평소값 → 악천후 시간의 도로 · 방향 비율 (사건 · 종류 · model_set 붙임).

    control=True: 같은 사건 번호로 '맑은 날 가짜 사건'(_control_hours)의 비율을 돌려준다. 사건 종류 · 시즌은 원래 사건 그대로.
    """
    v = pd.read_parquet(ROAD, columns=["date", "hour", "season", "split", "road_id", "road_name", "function_type",
                                       "direction", "travel_time_seconds"])
    v = v[v["split"] == split].copy()
    attach_weather_labels(v)
    if control:
        ctrl = _control_hours(v)
        v["date"] = pd.to_datetime(v["date"])
        v = v.merge(ctrl, on=["date", "hour"], how="inner")
        v["event_id"] = v["control_event_id"].astype(float)
        v = v.drop(columns=["control_event_id"])
    else:
        v = v[v["impact"].notna()].copy()
    v["weekday"] = pd.to_datetime(v["date"]).dt.dayofweek
    b = pd.read_csv(SHARED / _split_dir(split) / f"baseline_{_split_dir(split)}.csv", encoding="utf-8-sig",
                    usecols=["season", "road_id", "direction", "weekday", "hour", "baseline_travel_time_s"])
    v = v.merge(b, on=["season", "road_id", "direction", "weekday", "hour"], how="left", validate="many_to_one")
    v["ratio"] = v["travel_time_seconds"] / v["baseline_travel_time_s"]
    ev = pd.read_csv(SHARED / _split_dir(split) / f"weather_events_{_split_dir(split)}.csv", encoding="utf-8-sig",
                     usecols=["event_id", "type", "intensity", "start"])
    v = v.merge(ev, on="event_id", how="left", validate="many_to_one")
    return add_model_set(v, include_test=(split == "eval"))


def event_ratios(v: pd.DataFrame) -> pd.DataFrame:
    """사건 × 도로 · 방향 비율 중앙값과 그 사건 안의 백분위 순위."""
    keys = ["model_set", "season", "type", "event_id", "start", "intensity"]
    e = (v.groupby(keys + ["road_id", "road_name", "function_type", "direction"], dropna=False)
         .agg(event_ratio=("ratio", "median"), n_hours=("ratio", "count")).reset_index())
    e = e[e["n_hours"] > 0].copy()
    e["pct_rank"] = e.groupby("event_id")["event_ratio"].rank(pct=True, method="average")
    e["n_roads"] = e.groupby("event_id")["event_ratio"].transform("size")
    e["top20"] = e["pct_rank"] > 1 - TOP_SHARE
    e["event_hours"] = e.groupby("event_id")["n_hours"].transform("max")
    return e


def road_summary(e: pd.DataFrame, min_ratio: float) -> pd.DataFrame:
    """도로 · 방향마다 사건 수 · 상위 20% 횟수 · 비율 중앙값 → is_vulnerable."""
    s = (e.groupby(["type", "road_id", "road_name", "function_type", "direction"])
         .agg(n_events=("event_id", "nunique"), n_top20=("top20", "sum"), median_event_ratio=("event_ratio", "median"),
              max_event_ratio=("event_ratio", "max"))
         .reset_index())
    total = e.groupby("type")["event_id"].nunique().rename("type_events")
    s = s.merge(total, on="type")
    s["hit_share"] = s["n_top20"] / s["type_events"]   # 사건이 빠진 도로(빈값)는 상위에 못 든 것으로 센다
    s["is_vulnerable"] = (s["hit_share"] >= HIT_SHARE) & (s["median_event_ratio"] >= min_ratio)
    s["min_ratio"] = min_ratio
    return s.sort_values(["type", "is_vulnerable", "hit_share", "median_event_ratio"],
                         ascending=[True, False, False, False]).reset_index(drop=True)


def hit_rate(road_keys: set, e_target: pd.DataFrame) -> float:
    """목록 도로 · 방향이 대상 사건들에서 상위 20%에 든 비율 (사건 × 목록 도로 평균)."""
    if not road_keys:
        return np.nan
    keys = list(zip(e_target["road_id"], e_target["direction"]))
    in_list = np.array([k in road_keys for k in keys])
    per_event = pd.Series(e_target["top20"].to_numpy() & in_list).groupby(e_target["event_id"].to_numpy()).sum()
    return float(per_event.sum() / (len(road_keys) * e_target["event_id"].nunique()))


def permutation_p(k: int, e_target: pd.DataFrame, observed: float, rng: np.random.Generator,
                  n_perm: int = N_PERM) -> float:
    """같은 k개를 무작위로 골랐을 때 적중률이 observed 이상일 확률 (+1 보정)."""
    roads = e_target[["road_id", "direction"]].drop_duplicates().reset_index(drop=True)
    roads["i"] = np.arange(len(roads))
    t = e_target.merge(roads, on=["road_id", "direction"])
    ev_codes, ev_idx = np.unique(t["event_id"], return_inverse=True)
    top = np.zeros((len(roads), len(ev_codes)), dtype=bool)
    top[t["i"].to_numpy()[t["top20"].to_numpy()], ev_idx[t["top20"].to_numpy()]] = True
    per_road = top.sum(axis=1)
    sims = np.array([per_road[rng.choice(len(roads), k, replace=False)].sum() for _ in range(n_perm)])
    sims = sims / (k * len(ev_codes))
    return float((1 + (sims >= observed - 1e-12).sum()) / (n_perm + 1))


def choose_min_ratio(e: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Train 목록(최소 비율 후보마다) → Dev 적중률 · lift · 순열 p."""
    rows = []
    for kind in ["snow", "rain"]:
        tr = e[(e["model_set"] == "train") & (e["type"] == kind)]
        dv = e[(e["model_set"] == "dev") & (e["type"] == kind)]
        for mr in MIN_RATIO_CANDIDATES:
            s = road_summary(tr, mr)
            keys = set(zip(s.loc[s["is_vulnerable"], "road_id"], s.loc[s["is_vulnerable"], "direction"]))
            hr = hit_rate(keys, dv)
            p = permutation_p(len(keys), dv, hr, rng) if keys else np.nan
            rows.append({"type": kind, "min_ratio": mr, "n_list": len(keys), "train_events": tr["event_id"].nunique(),
                         "dev_events": dv["event_id"].nunique(), "dev_hit_rate": hr,
                         "lift": hr / TOP_SHARE if keys else np.nan, "perm_p": p})
    t = pd.DataFrame(rows)
    ok = t[t["n_list"] >= MIN_LIST]
    best = (ok.sort_values(["type", "lift", "n_list"], ascending=[True, False, False]).groupby("type").head(1))
    t["chosen"] = t.index.isin(best.index)
    return t


def check_test(final: pd.DataFrame, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """마지막 평가 때만: 학습 3년 목록이 Test 시즌 사건에서 상위 20%에 드는 비율 · lift · 순열 p (분석 설계 10.2 (3))."""
    rng = rng or np.random.default_rng(SEED)
    e_test = event_ratios(load_ratios("eval"))
    rows = []
    for kind in ["snow", "rain"]:
        f = final[(final["type"] == kind) & final["is_vulnerable"]]
        keys = set(zip(f["road_id"], f["direction"]))
        et = e_test[e_test["type"] == kind]
        hr = hit_rate(keys, et)
        rows.append({"type": kind, "n_list": len(keys), "test_events": et["event_id"].nunique(), "test_hit_rate": hr,
                     "lift": hr / TOP_SHARE, "perm_p": permutation_p(len(keys), et, hr, rng),
                     })
    t = pd.DataFrame(rows)
    t["repeats"] = (t["lift"] >= 2) & (t["perm_p"] < 0.05)
    return t


def leave_one_season_out(e: pd.DataFrame, chosen: dict, rng: np.random.Generator) -> pd.DataFrame:
    """종류마다 시즌 하나를 빼고 나머지 시즌 사건으로 목록을 만든 뒤, 뺀 시즌 사건에서 적중률 · lift · 순열 p."""
    rows = []
    for kind in ["snow", "rain"]:
        ek = e[e["type"] == kind]
        for season in sorted(ek["season"].unique()):
            src, tgt = ek[ek["season"] != season], ek[ek["season"] == season]
            s = road_summary(src, chosen.get(kind, 1.0))
            keys = set(zip(s.loc[s["is_vulnerable"], "road_id"], s.loc[s["is_vulnerable"], "direction"]))
            hr = hit_rate(keys, tgt)
            rows.append({"type": kind, "held_out": season, "build_events": src["event_id"].nunique(),
                         "held_out_events": tgt["event_id"].nunique(), "n_list": len(keys), "hit_rate": hr,
                         "lift": hr / TOP_SHARE if keys else np.nan,
                         "perm_p": permutation_p(len(keys), tgt, hr, rng) if keys else np.nan})
    return pd.DataFrame(rows)


def clear_control(final: pd.DataFrame, chosen: dict, rng: np.random.Generator) -> pd.DataFrame:
    """맑은 날 대조: (1) 최종 날씨 목록이 맑은 날 가짜 사건에서도 상위 20%에 드나 (2) 맑은 날로 같은 규칙을 돌리면 몇 개가 뽑히나."""
    ec = event_ratios(load_ratios("train", control=True))
    rows = []
    for kind in ["snow", "rain"]:
        f = final[(final["type"] == kind) & final["is_vulnerable"]]
        keys = set(zip(f["road_id"], f["direction"]))
        et = ec[ec["type"] == kind]
        hr = hit_rate(keys, et)
        s = road_summary(et, chosen.get(kind, 1.0))
        rows.append({"type": kind, "control_events": et["event_id"].nunique(), "n_weather_list": len(keys),
                     "clear_hit_rate": hr, "clear_lift": hr / TOP_SHARE,
                     "clear_perm_p": permutation_p(len(keys), et, hr, rng) if keys else np.nan,
                     "n_vulnerable_if_clear": int(s["is_vulnerable"].sum()),
                     "overlap_with_weather_list": int(sum((r, d) in keys for r, d in
                                                          zip(s.loc[s["is_vulnerable"], "road_id"],
                                                              s.loc[s["is_vulnerable"], "direction"])))})
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    v = load_ratios("train")
    e = event_ratios(v)
    cand = choose_min_ratio(e, rng)
    chosen = cand[cand["chosen"]].set_index("type")["min_ratio"].to_dict()
    finals = [road_summary(e[e["type"] == kind], chosen.get(kind, 1.0)) for kind in ["snow", "rain"]]
    final = pd.concat(finals, ignore_index=True)
    final.insert(1, "type_name", final["type"].map(KIND_NAME))

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    e.to_csv(TABLE_DIR / "vulnerable_event_ratios_train_dev.csv", index=False, encoding="utf-8-sig")
    cand.to_csv(TABLE_DIR / "vulnerable_min_ratio_dev.csv", index=False, encoding="utf-8-sig")
    final.to_csv(SHARED / "train_dev" / "vulnerable_roads_train_dev.csv", index=False, encoding="utf-8-sig")
    loso = leave_one_season_out(e, chosen, rng)
    loso.to_csv(TABLE_DIR / "vulnerable_leave_one_season_out.csv", index=False, encoding="utf-8-sig")
    ctrl = clear_control(final, chosen, rng)
    ctrl.to_csv(TABLE_DIR / "vulnerable_clear_control.csv", index=False, encoding="utf-8-sig")
    log = {"run_at": datetime.now(KST).isoformat(timespec="seconds"), "input": ROAD.name,
           "top_share": TOP_SHARE, "hit_share": HIT_SHARE, "chosen_min_ratio": chosen,
           "events": {f"{m}_{t}": int(n) for (m, t), n in e.groupby(["model_set", "type"])["event_id"].nunique().items()},
           "n_vulnerable": final.groupby("type")["is_vulnerable"].sum().astype(int).to_dict()}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "vulnerable_roads.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(cand.to_string())
    print(loso.to_string())
    print(ctrl.to_string())
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
