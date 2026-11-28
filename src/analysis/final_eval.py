"""
예측 최종 평가 (A 담당) - Test는 딱 한 번

노션 '분석 설계 7.3, 10.3'. 설정은 Dev로 다 골랐고(가중치 1배, 5단계까지), 여기서는
Train + Dev로 다시 학습한 뒤 Test 시즌(2025~26 겨울, 2026 여름)을 한 번만 본다.

모드
  연습 (기본)      Train으로 학습, Dev로 평가. Test 폴더는 읽지 않는다. 코드 확인용
  --open-test      Train + Dev로 학습, Test로 평가. 시작하자마자 logs/kmj/final/test_opened.json을 남기고,
                   이 파일이 있으면 다시 돌지 않는다 (Test를 보고 고친 뒤 다시 보는 일을 막으려고)

판정 (10.3, 결과를 보기 전에 정한 기준)
  h = 3, 분 단위 MAE. 악천후가 3단계보다 10% 이상 줄고 맑음은 5% 넘게 나빠지지 않으면 "날씨 효과 있음"
  노션 기준은 4단계 대 3단계다. 5단계 대 3단계도 같은 방식으로 함께 적는다
  악천후 변화는 사건(event_id) 단위로 다시 뽑는 부트스트랩 95% 신뢰구간을 같이 낸다
  대조 검사: 목표 시각이 맑음인 행에서 "20% 이상 느려진다"(y_hat >= 1.2)고 예측한 비율

실행
  python src/analysis/final_eval.py                 # 연습 (Dev)
  python src/analysis/final_eval.py --open-test     # 마지막에 한 번
결과 (모드 이름이 파일 이름에 들어간다: practice / test)
  outputs/tables/final_{mode}_metrics.csv     단계 x 구분 x h 지표 (prediction.evaluate와 같은 형식)
  outputs/tables/final_{mode}_judge.csv       판정표: 악천후와 맑음 변화, 신뢰구간, 대조 검사
  data/derived/final/{mode}_predictions.parquet 행별 예측 (사례 검증용, Git 밖)
  logs/kmj/final/final_{mode}.json            설정, 행 수, 걸린 시간
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import prediction as p  # noqa: E402
from src.io_utils import load_processed  # noqa: E402

LOG_DIR = ROOT / "logs" / "kmj" / "final"
LOCK_PATH = LOG_DIR / "test_opened.json"
STAGES = ["0_평소값", "1_직전상태", "2_선형회귀", "3_CatBoost", "4_CatBoost_날씨", "5_CatBoost_예보"]
BAD_GOAL = -0.10      # 악천후 MAE 10% 이상 감소
CLEAR_LIMIT = 0.05    # 맑음 MAE 5% 넘게 나빠지면 안 됨
N_BOOT = 2000


def load_all(include_test: bool) -> pd.DataFrame:
    """prediction.load()와 같은 처리. include_test=True일 때만 Test 시즌까지 읽는다"""
    f = load_processed("features", include_test=include_test)
    f = f.loc[f[p.TARGET].notna()].reset_index(drop=True)
    for c in ["weather_class", "weather_kind", "road_id", "direction", "function_type", "season_kind"]:
        f[c] = f[c].astype(str)
    w = load_processed("weather_hourly", include_test=include_test,
                       usecols=["datetime", "precip_mm", "snow_phase", "event_id", "season"])
    w = w.assign(datetime=pd.to_datetime(w["datetime"])).drop_duplicates("datetime").set_index("datetime")
    t_dt = pd.to_datetime(f["date"]) + pd.to_timedelta(f["t_hour"], unit="h")
    target_dt = pd.to_datetime(f["date"]) + pd.to_timedelta(f["target_hour"], unit="h")
    f = f.assign(precip_target=target_dt.map(w["precip_mm"]).to_numpy(),
                 snow_phase_target=target_dt.map(w["snow_phase"]).to_numpy(),
                 event_target=target_dt.map(w["event_id"]).to_numpy())
    return p.add_forecast_aws(f, t_dt, target_dt,
                              load_processed("fcst_seoul", include_test=include_test),
                              load_processed("aws_seoul", include_test=include_test))


def split(f: pd.DataFrame, open_test: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if open_test:
        fit_sets, eval_set = {"train", "dev"}, "test"
    else:
        if (f["model_set"] == "test").any():
            raise RuntimeError("연습 모드인데 Test 행이 들어왔습니다.")
        fit_sets, eval_set = {"train"}, "dev"
    fit = f.loc[f["model_set"].isin(fit_sets)].reset_index(drop=True)
    ev = f.loc[f["model_set"] == eval_set].reset_index(drop=True)
    return fit, ev


def predict_all(fit: pd.DataFrame, ev: pd.DataFrame, params: dict | None = None) -> tuple[pd.DataFrame, dict]:
    ev = ev.copy()
    seconds = {}
    ev["pred_0_평소값"] = p.predict_stage0(ev)
    ev["pred_1_직전상태"] = p.predict_stage1(ev)
    ev["pred_2_선형회귀"] = p.fit_linear(fit).predict(p.linear_frame(ev))
    for stage, cols in p.STAGE_COLS.items():
        s = time.perf_counter()
        model = p.fit_catboost(fit, params, cols)   # 가중치 1배 (11/27 확정)
        ev[f"pred_{stage}"] = model.predict(p.catboost_frame(ev, cols))
        seconds[stage] = round(time.perf_counter() - s, 1)
        print(f"  {stage} {seconds[stage]}초", flush=True)
    return ev, seconds


def mae_min(g: pd.DataFrame, pred_col: str) -> float:
    return float(((g[pred_col] - g[p.TARGET]).abs() * g["prior_target_s"] / 60).mean())


def event_bootstrap(bad: pd.DataFrame, base: str, new: str, n: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    """사건 단위로 다시 뽑아 (new MAE / base MAE - 1)의 95% 구간. 사건 번호가 없는 영향 시간은 날짜로 묶는다"""
    cluster = bad["event_target"].astype("string").fillna("day:" + bad["date"].astype(str))
    err_b = (bad[base] - bad[p.TARGET]).abs() * bad["prior_target_s"] / 60
    err_n = (bad[new] - bad[p.TARGET]).abs() * bad["prior_target_s"] / 60
    sums = pd.DataFrame({"c": cluster.to_numpy(), "b": err_b.to_numpy(), "n": err_n.to_numpy()}).groupby("c").agg(
        b=("b", "sum"), n=("n", "sum"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n, len(sums)))
    b, nn = sums["b"].to_numpy()[idx].sum(axis=1), sums["n"].to_numpy()[idx].sum(axis=1)
    ratio = nn / b - 1
    return float(np.quantile(ratio, 0.025)), float(np.quantile(ratio, 0.975))


def judge_table(ev: pd.DataFrame) -> pd.DataFrame:
    h3 = ev.loc[ev["horizon"] == 3]
    bad = h3.loc[h3["weather_class"] == p.BAD_CLASS]
    clear = h3.loc[h3["weather_class"] == "맑음"]
    base = "pred_3_CatBoost"
    rows = []
    for stage in ["4_CatBoost_날씨", "5_CatBoost_예보"]:
        col = f"pred_{stage}"
        bad_change = mae_min(bad, col) / mae_min(bad, base) - 1
        clear_change = mae_min(clear, col) / mae_min(clear, base) - 1
        lo, hi = event_bootstrap(bad, base, col)
        rows.append({
            "stage": stage, "vs": "3_CatBoost",
            "bad_rows": len(bad), "bad_hours": bad[["date", "target_hour"]].drop_duplicates().shape[0],
            "bad_events": bad["event_target"].nunique(),
            "bad_mae_base": mae_min(bad, base), "bad_mae": mae_min(bad, col),
            "bad_change": bad_change, "bad_change_lo": lo, "bad_change_hi": hi,
            "clear_mae_base": mae_min(clear, base), "clear_mae": mae_min(clear, col), "clear_change": clear_change,
            "clear_slow_pred_base": float((clear[base] >= p.SLOW).mean()),
            "clear_slow_pred": float((clear[col] >= p.SLOW).mean()),
            "clear_slow_actual": float((clear[p.TARGET] >= p.SLOW).mean()),
            "weather_effect": bool(bad_change <= BAD_GOAL and clear_change <= CLEAR_LIMIT),
        })
    return pd.DataFrame(rows)


def write_lock(info: dict) -> None:
    if LOCK_PATH.exists():
        raise RuntimeError(f"Test를 이미 열었습니다 ({LOCK_PATH.relative_to(ROOT)}). 다시 보지 않습니다.")
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-test", action="store_true", help="Train + Dev로 학습하고 Test를 한 번 본다")
    ap.add_argument("--iterations", type=int, help="CatBoost 반복 수 (연습 모드 시험용)")
    args = ap.parse_args()
    if args.open_test and args.iterations:
        sys.exit("Test는 확정 설정으로만 봅니다. --iterations와 같이 쓸 수 없습니다.")
    mode = "test" if args.open_test else "practice"
    params = dict(p.CATBOOST_PARAMS, iterations=args.iterations) if args.iterations else None
    t0 = time.perf_counter()
    started = datetime.now(p.KST).isoformat(timespec="seconds")
    if args.open_test:
        write_lock({"opened_at": started, "fit": "train + dev", "eval": "test", "stages": STAGES,
                    "catboost_params": p.CATBOOST_PARAMS, "weight": 1})

    f = load_all(include_test=args.open_test)
    fit, ev = split(f, args.open_test)
    ev, seconds = predict_all(fit, ev, params)
    metrics = p.evaluate(ev, STAGES)
    judge = judge_table(ev)

    tables = ROOT / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.round(4).to_csv(tables / f"final_{mode}_metrics.csv", index=False, encoding="utf-8-sig")
    judge.round(4).to_csv(tables / f"final_{mode}_judge.csv", index=False, encoding="utf-8-sig")
    pred_dir = ROOT / "data" / "derived" / "final"
    pred_dir.mkdir(parents=True, exist_ok=True)
    keep = p.KEY + ["t_hour", "season", "weather_class", "weather_kind", "event_target", "prior_target_s",
                    "prior_replaced_target", p.TARGET] + [f"pred_{s}" for s in STAGES]
    ev[keep].to_parquet(pred_dir / f"{mode}_predictions.parquet", index=False)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"final_{mode}.json").write_text(json.dumps({
        "run_at": started, "mode": mode,
        "fit_seasons": sorted(fit["season"].astype(str).unique()),
        "eval_seasons": sorted(ev["season"].astype(str).unique()),
        "rows": {"fit": len(fit), "eval": len(ev)},
        "catboost_params": params or p.CATBOOST_PARAMS, "weight": 1,
        "rule": f"h=3 분 MAE, 악천후 {BAD_GOAL:+.0%} 이하, 맑음 {CLEAR_LIMIT:+.0%} 이하, 사건 부트스트랩 {N_BOOT}회",
        "seconds": {**seconds, "total": round(time.perf_counter() - t0, 1)},
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(metrics.query("group_type == '전체' and horizon == '3'")[["stage", "mae_min", "mae_ratio"]].round(3)
          .to_string(index=False))
    print(judge.round(3).T.to_string())


if __name__ == "__main__":
    main()
