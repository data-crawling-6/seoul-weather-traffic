"""
예측 모델 0~5단계 (A 담당) - Train으로 학습, Dev로만 평가

노션 '분석 설계 8.3' 단계별 모델.
  0 평소값        y_hat = 1 (입력용 평소값 그대로)
  1 직전 상태     y_hat = ratio_t0 (t 시점 비율이 그대로 이어짐). ratio_t0가 비면 1
  2 선형회귀      (1)(2)(3) 입력. 빈 비율은 1.0 + 빈값 표시 열 (분석 설계 3.4 ⑦)
  3 CatBoost      같은 입력, 빈값 그대로. 날씨 없음
  4 CatBoost+날씨  3단계 입력 + (4) 관측 날씨 · 한강 수위 (5) 특보. 설정은 3단계와 같게 두고 입력만 바꾼다
  5 CatBoost+예보  4단계 입력 + 예보(fcst_seoul) + 구별 AWS(aws_seoul). B가 11/26 연구 기간 평일 전체로 다시 만든 표
                  예보: 목표 시각의 예보 중 발표 시각(base)이 t보다 앞선 것 가운데 가장 최근 것 (발표가 t와 같으면 못 씀)
                  AWS: t 시점(끝 시각 기준) 서울 지점 평균 · 최대
  가중치는 1배로 확정(11/27). 1 · 2 · 3배 비교는 --weights 1 2 3으로 다시 돌릴 수 있다

악천후 가중치 (노션 8.3 (2), 11/27 B 의견 반영)
  학습할 때 '내리는 중인 악천후' 행의 오차를 w배로 친다. 목표 시각 기준으로
  눈 악천후는 snow_phase == falling(B, #30), 비 악천후는 관측 강수 > 0인 시간이다.
  눈 그친 뒤 시간은 오히려 빨라서(B 분석 0.94배) 넣지 않는다.
  w = 1 · 2 · 3을 3 · 4단계 모두 Dev로 비교하고, 고른 w를 3 · 4 · 5단계에 같이 쓴다. w = 1은 기준선으로 남긴다.
  고르는 기준(돌리기 전에 정함): 악천후 MAE가 가장 낮은 w. 단 전체 MAE가 w = 1보다 5% 넘게 나빠지면 제외.
  weather_class · 목표 시각 강수는 가중치에만 쓰고 입력에는 넣지 않는다. 표본 뽑기는 하지 않는다.

평가 (분석 설계 9.2 (1))
  MAE 분 = |y_hat - y| x 입력용 평소 통행시간 / 60 (주 지표), MAE 비율, RMSE 분 · 비율
  큰 지연(실제 1.3배 이상) MAE, '20% 이상 느려짐'(실제 1.2배 이상) 적중률 · 정밀도
  나눠서: 전체 / 맑음 · 약한 비·눈 · 악천후 / 눈 · 비 / 겨울 · 여름 / 출근 · 퇴근 / 날씨 x 출퇴근 / 대체한 평소값 행
          / 반복 취약 도로 · 방향(B, vulnerable_roads_train_dev.csv의 눈 · 비 목록)
  각각 h = 1 · 3과 합친 값. 시간 수 = 겹치지 않는 (날짜, 목표 시각) 수
  정답이 빈 행(링크가 빠진 시각)은 학습 · 평가에서 뺀다

Test 시즌은 읽지 않는다 (load_processed 기본값, model_set == test 행이 있으면 멈춤).

실행
  python src/analysis/prediction.py
결과
  outputs/tables/prediction_dev_metrics.csv     단계 x 구분 x h 지표
  data/derived/prediction/dev_predictions.parquet Dev 행별 예측 (Git 밖)
  logs/kmj/prediction/prediction.json            입력 열 · 행 수 · 설정 · 걸린 시간
  위 세 파일은 늘 최신 결과다. 실행마다 같은 내용을 실행 시각 이름으로 한 벌 더 남긴다
  outputs/tables/prediction_runs/prediction_dev_metrics_YYYYMMDD_HHMM.csv
  logs/kmj/prediction/runs/prediction_YYYYMMDD_HHMM.json
  data/derived/prediction/runs/dev_predictions_YYYYMMDD_HHMM.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import load_processed  # noqa: E402

KST = timezone(timedelta(hours=9))
METRICS_PATH = ROOT / "outputs" / "tables" / "prediction_dev_metrics.csv"
PRED_PATH = ROOT / "data" / "derived" / "prediction" / "dev_predictions.parquet"
LOG_PATH = ROOT / "logs" / "kmj" / "prediction" / "prediction.json"

TARGET = "y_ratio"
KEY = ["date", "target_hour", "horizon", "road_id", "direction"]

# 입력 (1) 도로 (2) 시간 (3) 최근 상태
ROAD = ["road_id", "direction", "function_type", "n_links", "total_distance_m", "lanes", "urban_share", "is_riverside"]
TIME = ["target_hour", "horizon", "weekday", "month", "season_kind", "is_vacation_peak"]
RECENT = ["ratio_t0", "ratio_t1", "ratio_t2", "ratio_t0_missing", "ratio_t1_missing", "ratio_t2_missing",
          "city_ratio_t0", "type_ratio_t0"]
FEATURES = ROAD + TIME + RECENT

# 4단계에 더하는 입력 (4) 관측 날씨 · 한강 수위 (5) 특보. 모두 t 시점까지의 값
WEATHER = ["temp_c", "temp_drop_3h", "precip_1h", "precip_3h", "new_snow_3h_cm", "snow_depth_cm", "precip_type"]
WATER = ["wl_rise_3h", "wl_over_attention", "wl_over_warning", "wl2_rise_3h", "wl2_over_attention", "wl2_over_warning"]
WARN = [f"warn_{k}{s}" for k in ["snow", "rain", "cold"] for s in ["", "_upcoming", "_hours"]]
FEATURES_4 = FEATURES + WEATHER + WATER + WARN

# 5단계에 더하는 입력: 예보(목표 시각, t 이전 발표) · AWS(t 시점)
FCST_SRC = ["lead_h", "pcp_mean", "pcp_max", "sno_mean", "sno_max", "tmp_mean", "tmp_min",
            "pty_rain_regions", "pty_snow_regions"]
AWS_SRC = ["ta_mean", "ta_min", "rn_hr1_mean", "rn_hr1_max", "rn_day_max", "ws_mean", "ws_max", "hm_mean"]
FCST = [f"fcst_{c}" for c in FCST_SRC]
AWS = [f"aws_{c}" for c in AWS_SRC]
FEATURES_5 = FEATURES_4 + FCST + AWS

# 선형회귀에서 숫자 대신 범주로 넣을 열 (요일 · 월 · 시각은 크기 순서가 의미 없음)
LINEAR_CATEGORICAL = ["road_id", "direction", "function_type", "target_hour", "horizon", "weekday", "month", "season_kind"]
CATBOOST_CATEGORICAL = ["road_id", "direction", "function_type", "season_kind", "precip_type"]
FLAG_COLS = ["is_riverside", "is_vacation_peak", "ratio_t0_missing", "ratio_t1_missing", "ratio_t2_missing"]
WATER_FLAGS = ["wl_over_attention", "wl_over_warning", "wl2_over_attention", "wl2_over_warning"]
RATIO_COLS = ["ratio_t0", "ratio_t1", "ratio_t2"]

# 입력에 들어가면 안 되는 열 (정답 · 사후에 정해지는 값 · 평가 구분 · 4단계 이후 입력)
FORBIDDEN = {"y_ratio", "tt_target_s", "prior_target_s", "prior_replaced_target", "weather_class", "weather_kind",
             "model_set", "split", "season", "date", "precip_target", "snow_phase_target"}

CATBOOST_PARAMS = dict(iterations=800, learning_rate=0.08, depth=6, loss_function="RMSE",
                       random_seed=0, verbose=0, allow_writing_files=False)

STAGES = ["0_평소값", "1_직전상태", "2_선형회귀", "3_CatBoost", "4_CatBoost_날씨", "5_CatBoost_예보"]
BAD_WEIGHTS = [1]
STAGE_COLS = {"3_CatBoost": FEATURES, "4_CatBoost_날씨": FEATURES_4, "5_CatBoost_예보": FEATURES_5}
BAD_CLASS = "악천후"
MAX_TOTAL_WORSE = 0.05
BIG_DELAY = 1.3
SLOW = 1.2
VULNERABLE_PATH = ROOT / "data" / "shared" / "train_dev" / "vulnerable_roads_train_dev.csv"


def weighted_name(stage: str, w: int) -> str:
    return stage if w == 1 else f"{stage}_가중{w}"


def all_stages(weights: list[int] = BAD_WEIGHTS) -> list[str]:
    return STAGES[:3] + [weighted_name(st, w) for w in weights for st in STAGES[3:]]


def falling_bad(x: pd.DataFrame) -> np.ndarray:
    bad = x["weather_class"].astype(str) == BAD_CLASS
    snow = x["weather_kind"].astype(str) == "눈"
    falling = np.where(snow, x["snow_phase_target"].astype(str) == "falling", x["precip_target"].fillna(0) > 0)
    return (bad & falling).to_numpy()


def vulnerable_of(x: pd.DataFrame, path: Path = VULNERABLE_PATH) -> np.ndarray:
    """도로 · 방향이 B 목록에서 비 · 눈 어디에 취약한지. 둘 다면 '눈·비'"""
    if not path.exists():
        return np.full(len(x), "목록 없음")
    v = pd.read_csv(path, encoding="utf-8-sig")
    v = v.loc[v["is_vulnerable"].astype(bool)]
    key = x["road_id"].astype(str) + "|" + x["direction"].astype(str)
    lists = {t: set(g["road_id"].astype(str) + "|" + g["direction"].astype(str)) for t, g in v.groupby("type_name")}
    rain, snow = key.isin(lists.get("비", set())), key.isin(lists.get("눈", set()))
    return np.select([rain & snow, rain, snow], ["눈·비 취약", "비 취약", "눈 취약"], "나머지")


def bad_weights(train: pd.DataFrame, w: int) -> np.ndarray:
    return np.where(falling_bad(train), float(w), 1.0)


def add_target_weather(f: pd.DataFrame) -> pd.DataFrame:
    w = load_processed("weather_hourly", usecols=["datetime", "precip_mm", "snow_phase", "season"])
    w = w.assign(datetime=pd.to_datetime(w["datetime"])).drop_duplicates("datetime").set_index("datetime")
    target_dt = pd.to_datetime(f["date"]) + pd.to_timedelta(f["target_hour"], unit="h")
    return f.assign(precip_target=target_dt.map(w["precip_mm"]).to_numpy(),
                    snow_phase_target=target_dt.map(w["snow_phase"]).to_numpy())


def pick_weight(metrics: pd.DataFrame, stage: str, weights: list[int]) -> int:
    m = metrics.loc[metrics["horizon"] == "전체"]

    def mae(name, gtype, group):
        sel = (m["stage"] == name) & (m["group_type"] == gtype) & (m["group"] == group)
        return float(m.loc[sel, "mae_min"].iloc[0])

    base_total = mae(weighted_name(stage, 1), "전체", "전체")
    ok = [w for w in weights if mae(weighted_name(stage, w), "전체", "전체") <= base_total * (1 + MAX_TOTAL_WORSE)]
    return min(ok, key=lambda w: mae(weighted_name(stage, w), "날씨", BAD_CLASS))


def load() -> tuple[pd.DataFrame, int]:
    f = load_processed("features")
    if (f["model_set"] == "test").any():
        raise RuntimeError("Test 시즌 행이 들어왔습니다. 예측은 Train · Dev만 씁니다.")
    dropped = int(f[TARGET].isna().sum())
    f = f.loc[f[TARGET].notna()].reset_index(drop=True)
    for c in ["weather_class", "weather_kind", "road_id", "direction", "function_type", "season_kind"]:
        f[c] = f[c].astype(str)
    f = add_target_weather(f)
    t_dt = pd.to_datetime(f["date"]) + pd.to_timedelta(f["t_hour"], unit="h")
    target_dt = pd.to_datetime(f["date"]) + pd.to_timedelta(f["target_hour"], unit="h")
    return add_forecast_aws(f, t_dt, target_dt, load_processed("fcst_seoul"), load_processed("aws_seoul")), dropped


def add_forecast_aws(f: pd.DataFrame, t_dt: pd.Series, target_dt: pd.Series,
                     fcst: pd.DataFrame, aws: pd.DataFrame) -> pd.DataFrame:
    """예보는 목표 시각 예보 가운데 base < t인 가장 최근 발표, AWS는 t 시점 값. 없으면 빈값"""
    fc = fcst.assign(base=pd.to_datetime(fcst["base"]), fcst=pd.to_datetime(fcst["fcst"]))
    keys = pd.DataFrame({"t_dt": t_dt.to_numpy(), "target_dt": target_dt.to_numpy()}).drop_duplicates()
    cand = keys.merge(fc[["base", "fcst"] + FCST_SRC], left_on="target_dt", right_on="fcst", how="inner")
    cand = cand.loc[cand["base"] < cand["t_dt"]].sort_values("base")
    best = cand.groupby(["t_dt", "target_dt"], as_index=False).last()
    best = best.rename(columns={c: f"fcst_{c}" for c in FCST_SRC})[["t_dt", "target_dt"] + FCST]

    aw = aws.assign(datetime=pd.to_datetime(aws["datetime"])).drop_duplicates("datetime")
    aw = aw.rename(columns={c: f"aws_{c}" for c in AWS_SRC})[["datetime"] + AWS]

    out = pd.DataFrame({"t_dt": t_dt.to_numpy(), "target_dt": target_dt.to_numpy()})
    out = out.merge(best, on=["t_dt", "target_dt"], how="left").merge(aw, left_on="t_dt", right_on="datetime", how="left")
    return f.assign(**{c: out[c].to_numpy() for c in FCST + AWS})


def predict_stage0(x: pd.DataFrame) -> np.ndarray:
    return np.ones(len(x))


def predict_stage1(x: pd.DataFrame) -> np.ndarray:
    return x["ratio_t0"].fillna(1.0).to_numpy()


def linear_frame(x: pd.DataFrame) -> pd.DataFrame:
    out = x[FEATURES].copy()
    out[RATIO_COLS] = out[RATIO_COLS].fillna(1.0)
    for c in FLAG_COLS:
        out[c] = out[c].astype(int)
    for c in LINEAR_CATEGORICAL:
        out[c] = out[c].astype(str)
    return out


def fit_linear(train: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric = [c for c in FEATURES if c not in LINEAR_CATEGORICAL]
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), LINEAR_CATEGORICAL),
        ("num", "passthrough", numeric),
    ])
    model = make_pipeline(prep, LinearRegression())
    model.fit(linear_frame(train), train[TARGET])
    return model


def catboost_frame(x: pd.DataFrame, cols: list[str] = FEATURES) -> pd.DataFrame:
    out = x[cols].copy()
    for c in cat_cols(cols):
        out[c] = out[c].astype(str)
    for c in FLAG_COLS:
        out[c] = out[c].astype(int)
    # 수위 넘음 표시는 하천변 도로 여름에만 있다. 나머지는 빈값 그대로 둔다
    for c in set(WATER_FLAGS) & set(cols):
        out[c] = out[c].astype("float64")
    return out


def cat_cols(cols: list[str]) -> list[str]:
    return [c for c in CATBOOST_CATEGORICAL if c in cols]


def fit_catboost(train: pd.DataFrame, params: dict | None = None, cols: list[str] = FEATURES,
                 weight: np.ndarray | None = None):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(**(params or CATBOOST_PARAMS), cat_features=cat_cols(cols))
    model.fit(catboost_frame(train, cols), train[TARGET], sample_weight=weight)
    return model


def commute_of(target_hour: pd.Series) -> pd.Series:
    return np.where(target_hour.isin([8, 9]), "출근", "퇴근")


def metric_row(g: pd.DataFrame, pred_col: str) -> dict:
    err = g[pred_col] - g[TARGET]
    err_min = err * g["prior_target_s"] / 60
    big = g[TARGET] >= BIG_DELAY
    slow, pred_slow = g[TARGET] >= SLOW, g[pred_col] >= SLOW
    return {
        "n_rows": len(g),
        "n_hours": g[["date", "target_hour"]].drop_duplicates().shape[0],
        "mae_min": err_min.abs().mean(),
        "rmse_min": np.sqrt((err_min ** 2).mean()),
        "mae_ratio": err.abs().mean(),
        "rmse_ratio": np.sqrt((err ** 2).mean()),
        "mean_y": g[TARGET].mean(),
        "mean_pred": g[pred_col].mean(),
        "n_big": int(big.sum()),
        "mae_min_big": err_min[big].abs().mean() if big.any() else np.nan,
        "n_slow": int(slow.sum()),
        "slow_recall": (pred_slow & slow).sum() / slow.sum() if slow.any() else np.nan,
        "slow_precision": (pred_slow & slow).sum() / pred_slow.sum() if pred_slow.any() else np.nan,
    }


def evaluate(dev: pd.DataFrame, stages: list[str]) -> pd.DataFrame:
    commute = commute_of(dev["target_hour"])
    d = dev.assign(commute=commute,
                   weather_commute=dev["weather_class"].astype(str) + " x " + commute,
                   replaced=np.where(dev["prior_replaced_target"], "대체", "28일"),
                   kind=dev["weather_kind"].replace({"": "없음"}),
                   vulnerable=vulnerable_of(dev))
    d["vulnerable_weather"] = d["vulnerable"] + " x " + d["weather_class"].astype(str)
    groupings = {"전체": None, "날씨": "weather_class", "눈비": "kind", "계절": "season_kind",
                 "출퇴근": "commute", "날씨x출퇴근": "weather_commute", "평소값": "replaced",
                 "취약도로": "vulnerable", "취약도로x날씨": "vulnerable_weather"}
    rows = []
    for stage in stages:
        for h_name, part in [("전체", d)] + [(str(h), d.loc[d["horizon"] == h]) for h in sorted(d["horizon"].unique())]:
            for gtype, col in groupings.items():
                groups = [("전체", part)] if col is None else list(part.groupby(col, sort=True))
                for gname, g in groups:
                    if gtype == "눈비" and gname == "없음":
                        continue
                    rows.append({"stage": stage, "horizon": h_name, "group_type": gtype, "group": gname,
                                 **metric_row(g, f"pred_{stage}")})
    return pd.DataFrame(rows)


def run(params: dict | None = None, weights: list[int] = BAD_WEIGHTS) -> tuple[pd.DataFrame, dict]:
    t0 = time.perf_counter()
    f, dropped = load()
    train = f.loc[f["model_set"] == "train"].reset_index(drop=True)
    dev = f.loc[f["model_set"] == "dev"].reset_index(drop=True)
    assert not (set(FEATURES_5) & FORBIDDEN)

    seconds = {}
    dev["pred_0_평소값"] = predict_stage0(dev)
    dev["pred_1_직전상태"] = predict_stage1(dev)

    s = time.perf_counter()
    lin = fit_linear(train)
    dev["pred_2_선형회귀"] = lin.predict(linear_frame(dev))
    seconds["linear"] = round(time.perf_counter() - s, 1)

    def importance(model, cols):
        return dict(sorted(zip(cols, model.get_feature_importance().round(2).tolist()), key=lambda kv: -kv[1]))

    imp = {}
    for w in weights:
        wt = None if w == 1 else bad_weights(train, w)
        for stage, cols in STAGE_COLS.items():
            name = weighted_name(stage, w)
            s = time.perf_counter()
            model = fit_catboost(train, params, cols, wt)
            dev[f"pred_{name}"] = model.predict(catboost_frame(dev, cols))
            seconds[name] = round(time.perf_counter() - s, 1)
            imp[name] = importance(model, cols)

    stages = all_stages(weights)
    metrics = evaluate(dev, stages)
    picked = {st: pick_weight(metrics, st, weights) for st in STAGES[3:]}
    log = {
        "run_at": datetime.now(KST).isoformat(timespec="seconds"),
        "features": FEATURES,
        "features_added_stage4": WEATHER + WATER + WARN,
        "features_added_stage5": FCST + AWS,
        "stage5_missing_share": {c: round(float(f[c].isna().mean()), 4) for c in FCST + AWS},
        "rows": {"train": len(train), "dev": len(dev),
                 "train_seasons": sorted(train["season"].astype(str).unique()),
                 "dev_seasons": sorted(dev["season"].astype(str).unique())},
        "dropped_no_target": dropped,
        "linear": "빈 비율 1.0 + 빈값 표시, 범주 원핫(drop first): " + ", ".join(LINEAR_CATEGORICAL),
        "catboost_params": params or CATBOOST_PARAMS,
        "bad_weather_weights": weights,
        "bad_weather_train_share": round(float((train["weather_class"].astype(str) == BAD_CLASS).mean()), 4),
        "weighted_train_share": round(float(falling_bad(train).mean()), 4),
        "weight_rule": f"악천후 MAE 최소, 전체 MAE가 1배보다 {MAX_TOTAL_WORSE:.0%} 넘게 나빠지면 제외",
        "picked_weight": picked,
        "weight_decision": "1배(가중치 없음) 확정, 11/27 B 합의. h=3 기준이면 4단계 2배가 악천후 0.6% 낫지만 "
                           "차이가 작고 결과를 보고 기준을 바꾸지 않으려고 1배 유지",
        "vulnerable_roads": str(VULNERABLE_PATH.relative_to(ROOT)),
        "catboost_importance": imp,
        "seconds": {**seconds, "total": round(time.perf_counter() - t0, 1)},
    }
    keep = KEY + ["season", "weather_class", "weather_kind", "prior_target_s", "prior_replaced_target", TARGET] + \
        [f"pred_{s}" for s in stages]
    return metrics, {"log": log, "pred": dev[keep]}


def run_paths(stamp: str, metrics_path: Path = METRICS_PATH, pred_path: Path = PRED_PATH,
              log_path: Path = LOG_PATH) -> dict[str, Path]:
    return {
        "metrics": metrics_path.parent / "prediction_runs" / f"{metrics_path.stem}_{stamp}.csv",
        "pred": pred_path.parent / "runs" / f"{pred_path.stem}_{stamp}.parquet",
        "log": log_path.parent / "runs" / f"{log_path.stem}_{stamp}.json",
    }


def save_outputs(metrics: pd.DataFrame, extra: dict, metrics_path: Path = METRICS_PATH,
                 pred_path: Path = PRED_PATH, log_path: Path = LOG_PATH) -> dict[str, Path]:
    """최신 결과를 덮어쓰고, 같은 내용을 실행 시각 이름으로 한 벌 더 남긴다 (다시 돌려도 예전 결과가 남게)"""
    stamp = datetime.fromisoformat(extra["log"]["run_at"]).strftime("%Y%m%d_%H%M")
    archived = run_paths(stamp, metrics_path, pred_path, log_path)
    text = json.dumps(extra["log"], ensure_ascii=False, indent=1)
    for latest, copy in [(metrics_path, archived["metrics"]), (pred_path, archived["pred"]), (log_path, archived["log"])]:
        for path in (latest, copy):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".csv":
                metrics.round(4).to_csv(path, index=False, encoding="utf-8-sig")
            elif path.suffix == ".parquet":
                extra["pred"].to_parquet(path, index=False)
            else:
                path.write_text(text, encoding="utf-8")
    return archived


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, help="CatBoost 반복 수 (시험용으로 줄일 때)")
    ap.add_argument("--weights", type=int, nargs="+", default=BAD_WEIGHTS, help="악천후 가중치 목록 (기본 1배만)")
    args = ap.parse_args()
    params = dict(CATBOOST_PARAMS, iterations=args.iterations) if args.iterations else None

    metrics, extra = run(params, args.weights)
    save_outputs(metrics, extra)

    main_view = metrics.query("group_type == '전체' and horizon != '전체'")
    print(main_view[["stage", "horizon", "mae_min", "mae_ratio", "rmse_min"]].round(3).to_string(index=False))
    print(f"저장: {METRICS_PATH.relative_to(ROOT)}, {PRED_PATH.relative_to(ROOT)}, {LOG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
