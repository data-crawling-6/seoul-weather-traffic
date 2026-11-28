"""
최종 예측 모델 SHAP (A가 계산하고 B가 해석) - 노션 '분석 설계 11.1 (3)'

최종 모델(5단계 CatBoost, 가중치 1배)을 Train으로 학습하고, Dev 행에서만 변수별 기여도를 계산한다.
Test 시즌은 읽지 않는다 (prediction.load). Dev 결과(prediction.py)와 같은 모델이다.

- h = 3 행만 (주 지표)
- 나누기: 계절(겨울, 여름) x (악천후만, 전체). 전체는 계절마다 무작위 20,000행 (속도 때문)
- SHAP은 CatBoost 내장 TreeSHAP (get_feature_importance type=ShapValues). shap 패키지의 TreeExplainer와 같은 값이고 더 빠르다.
  단위는 y_ratio (평소 대비 배수). 예: 0.05면 그 변수 때문에 예측이 평소의 5%만큼 더 느려짐

실행
  python src/analysis/shap_values.py
결과
  outputs/tables/shap_summary.csv            구분 x 변수: 평균 |SHAP|, 평균 SHAP, 순위
  outputs/figures/kmj/shap_top15.png         구분별 평균 |SHAP| 상위 15개 막대 그림
  data/derived/final/shap_bad_rows.parquet   Dev 악천후 행별 SHAP (키, 실제, 예측 + 변수별 기여도, Git 밖)
  logs/kmj/final/shap.json                   설정, 행 수, 걸린 시간
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

FINAL_STAGE = "5_CatBoost_예보"
SAMPLE_ALL = 20_000
SUMMARY_PATH = ROOT / "outputs" / "tables" / "shap_summary.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "kmj" / "shap_top15.png"
ROWS_PATH = ROOT / "data" / "derived" / "final" / "shap_bad_rows.parquet"
LOG_PATH = ROOT / "logs" / "kmj" / "final" / "shap.json"


def shap_matrix(model, x: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    from catboost import Pool

    pool = Pool(p.catboost_frame(x, cols), cat_features=p.cat_cols(cols))
    sv = model.get_feature_importance(pool, type="ShapValues")   # 마지막 열은 기준값(평균 예측)
    return pd.DataFrame(sv[:, :-1], columns=cols, index=x.index)


def groups_of(f: pd.DataFrame, seed: int = 0) -> dict[str, pd.DataFrame]:
    h3 = f.loc[f["horizon"] == 3]
    out = {}
    for season in ["winter", "summer"]:
        s = h3.loc[h3["season_kind"] == season]
        out[f"{season} x 악천후"] = s.loc[s["weather_class"] == p.BAD_CLASS]
        out[f"{season} x 전체"] = s.sample(min(SAMPLE_ALL, len(s)), random_state=seed)
    return out


def summarize(shap_by_group: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, sv in shap_by_group.items():
        mean_abs = sv.abs().mean()
        rank = mean_abs.rank(ascending=False, method="min").astype(int)
        for col in sv.columns:
            rows.append({"group": name, "n_rows": len(sv), "feature": col, "mean_abs": mean_abs[col],
                         "mean": sv[col].mean(), "rank": rank[col]})
    return pd.DataFrame(rows).sort_values(["group", "rank"]).reset_index(drop=True)


def plot_top(summary: pd.DataFrame, path: Path, top: int = 15) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"  # 윈도우 한글 글꼴
    plt.rcParams["axes.unicode_minus"] = False
    names = list(dict.fromkeys(summary["group"]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, name in zip(axes.ravel(), names):
        s = summary.loc[(summary["group"] == name) & (summary["rank"] <= top)].sort_values("mean_abs")
        ax.barh(s["feature"], s["mean_abs"], color="#4c72b0")
        ax.set_title(f"{name} ({int(s['n_rows'].iloc[0]):,}행)")
        ax.set_xlabel("평균 |SHAP| (평소 대비 배수)")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("5단계 모델 변수 기여도 (Dev, h = 3)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, help="CatBoost 반복 수 (시험용)")
    args = ap.parse_args()
    params = dict(p.CATBOOST_PARAMS, iterations=args.iterations) if args.iterations else None
    t0 = time.perf_counter()

    f, _ = p.load()   # Train + Dev만, Test가 섞이면 멈춘다
    train = f.loc[f["model_set"] == "train"].reset_index(drop=True)
    dev = f.loc[f["model_set"] == "dev"].reset_index(drop=True)
    cols = p.STAGE_COLS[FINAL_STAGE]
    model = p.fit_catboost(train, params, cols)
    fit_sec = round(time.perf_counter() - t0, 1)

    groups = groups_of(dev)
    shap_by_group = {name: shap_matrix(model, g, cols) for name, g in groups.items()}
    summary = summarize(shap_by_group)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.round(5).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    plot_top(summary, FIG_PATH)

    bad_rows = []
    for season in ["winter", "summer"]:
        g = groups[f"{season} x 악천후"]
        info = g[p.KEY + ["season", "weather_kind", p.TARGET]].assign(pred=model.predict(p.catboost_frame(g, cols)))
        bad_rows.append(pd.concat([info, shap_by_group[f"{season} x 악천후"].add_prefix("shap_")], axis=1))
    ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(bad_rows, ignore_index=True).to_parquet(ROWS_PATH, index=False)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps({
        "run_at": datetime.now(p.KST).isoformat(timespec="seconds"),
        "model": FINAL_STAGE, "fit_seasons": sorted(train["season"].astype(str).unique()), "fit_rows": len(train),
        "shap_seasons": sorted(dev["season"].astype(str).unique()),
        "catboost_params": params or p.CATBOOST_PARAMS, "weight": 1,
        "groups": {k: len(v) for k, v in groups.items()},
        "seconds": {"fit": fit_sec, "total": round(time.perf_counter() - t0, 1)},
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    top = summary.loc[summary["rank"] <= 8, ["group", "rank", "feature", "mean_abs", "mean"]]
    print(top.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
