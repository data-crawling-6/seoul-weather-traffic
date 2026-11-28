"""
정체 경보 모델 · 경보 기준 (B 담당, 노션 분석 설계 9.2 ② · 일정 B '정체 경보 모델 → 경보 기준')

문제
  시각 t 에, h 시간 뒤(1 · 3시간) 평일 출퇴근 시각의 도로 · 방향이 '정체'(is_congested = 1)일지 맞힌다.
  정답 = road_congestion_length 의 is_congested (평소 2/3 속도 미만 링크가 도로 길이의 30% 이상).
  입력 = A 입력 변수 표(features)의 t 시점까지 값 (도로 · 시간 · 최근 상태 · 관측 날씨 · 한강 수위 · 특보, A 4단계와 같음).
  A 회귀는 큰 지연을 낮게 예측해서(실제 1.6배 → 1.13~1.21배), '크게 막히나'는 이 분류 모델이 맡는다.

모델 (Train 으로 학습, Dev 로 고르고 평가. Test 는 읽지 않는다)
  기준선 1  특보가 뜨면 경보      t 에 대설 · 호우특보가 발효 중이면 경보 (설계 9.2 ② 기준선)
  기준선 2  지금 느리면 경보      t 시점 비율(ratio_t0)이 클수록 위험. 임계값은 아래 규칙으로
  로지스틱 회귀                   숫자 열 표준화 + 범주 원핫, class_weight=balanced
  로지스틱 회귀 + 예보 · 겹침     위 입력 + 예보 + 겹침 열 (아래)
  결정트리 + 예보 · 겹침          깊이 3 · 5 · 7 · 9 중 Dev 겨울 PR-AUC 가 가장 큰 것, class_weight=balanced

  예보 (11/27 보완 1): fcst_seoul 에서 목표 시각을 예보한 발표 가운데 **발표 시각 + 1시간 <= t** 인 가장 최근 것.
       발표 뒤 자료가 올라오기까지 시간이 걸리므로 1시간 여유를 둔다 (t 시점에 실제로 볼 수 있던 예보만).
  겹침 열 (보완 3): 로지스틱 회귀는 '비 + 도시고속도로'처럼 겹칠 때 커지는 효과를 스스로 못 배운다.
       24시간 비교 · 생존분석에서 확인한 조합을 열로 만든다: (관측 · 예보 강수, 신적설) × 도시고속도로 · 퇴근 시각,
       특보 · 수위 상승 × 하천변 도로. 강수는 log(1 + mm).
  h = 1 · 3 마다 따로 학습한다.

경보 기준 (돌리기 전에 정함, 설계 9.2 ②: 재현율 우선 · 정확도는 쓰지 않음)
  Dev 재현율 80% 이상인 임계값 가운데 정밀도가 가장 높은 것. 80%에 못 미치면 재현율이 가장 높은 임계값.
  나눠 고르기 (보완 2): Dev 겨울로 깊이와 임계값을 고르고 그 뒤 Dev 여름을 평가한다.
       겨울은 선택용, 여름은 검증용이며 Test가 아니다.
  같은 경보 수 비교: 여름에서 특보 기준선과 경보 개수를 맞춘 임계값의 재현율 · 정밀도 (경보를 늘리지 않고 몇 개를 더 잡나)
  함께 적는 것: 정밀도 · 재현율 · PR-AUC · 오차행렬, 하루 경보 수 · 헛경보 수 (Dev 평일 하루, 180개 도로 · 방향 합),
  악천후 시간만 본 재현율 · 정밀도, 선행 시간(정체 시작 시각을 3시간 전 / 1시간 전에 잡았나)
  헛경보 단위: fp · false_alarms_per_day 는 도로 · 방향 × 목표 시각 칸, fp_road_days_per_day 는 같은 날 같은 도로를 한 번으로 묶은 수.
       둘 다 실제 이용자에게 보낸 알림 수는 아니다.
  선행 시간의 '시작' = 출퇴근 구간에서 처음 관측한 정체. 08 · 18시는 직전 상태를 몰라 따로 뺀 줄도 적는다.

실행
  python src/analysis/alert_model.py
결과
  outputs/tables/alert_dev_metrics.csv       모델 × h 지표 (고른 임계값에서)
  outputs/tables/alert_dev_pr_curve.csv      모델 × h 정밀도-재현율 곡선 (그림용, 점 200개)
  outputs/tables/alert_lead_time_dev.csv     정체 시작을 몇 시간 전에 잡았나
  outputs/tables/alert_tree_rules.txt        고른 결정트리 규칙 (위쪽 3단계)
  outputs/tables/alert_criteria_dev.csv      경보 기준 정리: 로지스틱 회귀 · 두 운영 기준(겨울에서 정함) · 여름 성능 · 선행 시간 · 헛경보
  data/derived/alert/dev_scores.parquet      Dev 행별 점수 (Git 밖)
  logs/ljh/alert_model/alert_model.json
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.io_utils import load_processed  # noqa: E402

KST = timezone(timedelta(hours=9))
TABLE_DIR = ROOT / "outputs" / "tables"
SCORE_PATH = ROOT / "data" / "derived" / "alert" / "dev_scores.parquet"
LOG_PATH = ROOT / "logs" / "ljh" / "alert_model" / "alert_model.json"

KEY = ["date", "target_hour", "horizon", "road_id", "direction"]
CATEGORICAL = ["road_id", "direction", "function_type", "target_hour", "weekday", "month", "season_kind", "precip_type"]
NUMERIC = ["n_links", "total_distance_m", "lanes", "urban_share", "is_riverside", "is_vacation_peak",
           "ratio_t0", "ratio_t1", "ratio_t2", "ratio_t0_missing", "ratio_t1_missing", "ratio_t2_missing",
           "city_ratio_t0", "type_ratio_t0",
           "temp_c", "temp_drop_3h", "precip_1h", "precip_3h", "new_snow_3h_cm", "snow_depth_cm",
           "wl_rise_3h", "wl_over_attention", "wl_over_warning", "wl2_rise_3h", "wl2_over_attention", "wl2_over_warning",
           "warn_snow", "warn_rain", "warn_cold", "warn_snow_hours", "warn_rain_hours", "warn_cold_hours"]
FCST = ["fc_pcp_mean", "fc_pcp_max", "fc_sno_max", "fc_tmp_min", "fc_rain_regions", "fc_snow_regions", "fc_lead_h",
        "fc_missing"]
INTER = ["x_rain_expr", "x_snow_expr", "x_fcrain_expr", "x_fcsnow_expr", "x_rain_evening", "x_fcrain_evening",
         "x_fcsnow_evening", "x_warnrain_river", "x_wlrise_river"]
TREE_DEPTHS = [3, 5, 7, 9]
MIN_RECALL = 0.80
CRITERIA_MODEL = "로지스틱 회귀"   # 경보에 쓰는 모델 (예보 · 겹침을 넣어도 나아지지 않아 단순한 쪽, 11/28)
SEED = 20261127


def load_data() -> pd.DataFrame:
    """A 입력 변수 표 + 정체 정답. Train · Dev 만 (Test 는 load_processed 기본값으로 읽지 않음)."""
    f = load_processed("features")
    if (f["model_set"] == "test").any():
        raise RuntimeError("Test 행이 들어왔습니다.")
    c = load_processed("road_congestion_length")[["date", "hour", "road_id", "direction", "is_congested"]]
    c = c.rename(columns={"hour": "target_hour"})
    f["date"], c["date"] = pd.to_datetime(f["date"]), pd.to_datetime(c["date"])
    for col in ["road_id", "direction"]:
        f[col], c[col] = f[col].astype(str), c[col].astype(str)
    d = f.merge(c, on=["date", "target_hour", "road_id", "direction"], how="left", validate="many_to_one")
    d = d[d["is_congested"].notna()].copy()       # 링크가 빠진 시각은 정답이 없어 뺀다
    d["y"] = d["is_congested"].astype(int)
    return add_forecast(d)


def add_forecast(d: pd.DataFrame) -> pd.DataFrame:
    """목표 시각 예보 가운데 t 시점에 볼 수 있던(발표 + 1시간 <= t) 가장 최근 발표를 붙인다."""
    fc = load_processed("fcst_seoul")
    fc["base"], fc["fcst"] = pd.to_datetime(fc["base"]), pd.to_datetime(fc["fcst"])
    keys = d[["date", "target_hour", "horizon"]].drop_duplicates().copy()
    keys["fcst"] = keys["date"] + pd.to_timedelta(keys["target_hour"].astype(int), unit="h")
    keys["cutoff"] = keys["fcst"] - pd.to_timedelta(keys["horizon"].astype(int) + 1, unit="h")
    m = pd.merge_asof(keys.sort_values("cutoff"), fc.sort_values("base")[
        ["fcst", "base", "lead_h", "pcp_mean", "pcp_max", "sno_max", "tmp_min", "pty_rain_regions", "pty_snow_regions"]],
        left_on="cutoff", right_on="base", by="fcst", direction="backward")
    if (m["base"] > m["cutoff"]).any():
        raise RuntimeError("t 시점 뒤에 나온 예보가 들어갔습니다.")
    m = m.rename(columns={"pcp_mean": "fc_pcp_mean", "pcp_max": "fc_pcp_max", "sno_max": "fc_sno_max",
                          "tmp_min": "fc_tmp_min", "pty_rain_regions": "fc_rain_regions",
                          "pty_snow_regions": "fc_snow_regions", "lead_h": "fc_lead_h"})
    vals = ["fc_pcp_mean", "fc_pcp_max", "fc_sno_max", "fc_tmp_min"]
    m[vals] = m[vals].mask(m[vals] <= -90)        # 기상청 결측 코드 -99
    m["fc_missing"] = (m["base"].isna() | m["fc_pcp_max"].isna()).astype(int)
    return d.merge(m.drop(columns=["fcst", "cutoff", "base"]), on=["date", "target_hour", "horizon"], how="left",
                   validate="many_to_one")


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    """빈값 채우기: 비율은 1(평소와 같음) + 빈값 표시 열은 이미 있음, 날씨 · 수위 · 특보 빈값은 0, 기온은 학습 중앙값."""
    x = d.copy()
    for c in ["ratio_t0", "ratio_t1", "ratio_t2", "city_ratio_t0", "type_ratio_t0"]:
        x[c] = x[c].fillna(1.0)
    for c in NUMERIC:
        x[c] = pd.to_numeric(x[c].astype(float) if x[c].dtype == bool else x[c], errors="coerce")
    x["temp_c"] = x["temp_c"].fillna(x.loc[x["model_set"] == "train", "temp_c"].median())
    x[NUMERIC] = x[NUMERIC].fillna(0.0)
    x["fc_tmp_min"] = x["fc_tmp_min"].fillna(x["temp_c"])
    x[FCST] = x[FCST].astype(float).fillna(0.0)
    expr = x["function_type"].astype(str).eq("도시고속도로").astype(float)
    evening = (x["target_hour"].astype(int) >= 12).astype(float)
    river = x["is_riverside"].astype(float)
    rain, fcrain = np.log1p(x["precip_3h"]), np.log1p(x["fc_pcp_max"])
    x["x_rain_expr"], x["x_snow_expr"] = rain * expr, x["new_snow_3h_cm"] * expr
    x["x_fcrain_expr"], x["x_fcsnow_expr"] = fcrain * expr, x["fc_sno_max"] * expr
    x["x_rain_evening"], x["x_fcrain_evening"], x["x_fcsnow_evening"] = rain * evening, fcrain * evening, x["fc_sno_max"] * evening
    x["x_warnrain_river"] = (x["warn_rain"] > 0).astype(float) * river
    x["x_wlrise_river"] = x["wl_rise_3h"].clip(lower=0) * river
    for c in CATEGORICAL:
        x[c] = x[c].astype(str).replace({"nan": "없음", "<NA>": "없음"})
    return x


def threshold_rule(y: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    """재현율 80% 이상 중 정밀도 최대인 임계값 (못 미치면 재현율 최대). (임계값, 정밀도, 재현율)."""
    p, r, t = precision_recall_curve(y, s)
    p, r = p[:-1], r[:-1]                         # 마지막 점(재현율 0)은 임계값이 없다
    ok = r >= MIN_RECALL
    i = int(np.argmax(np.where(ok, p, -1))) if ok.any() else int(np.argmax(r))
    return float(t[i]), float(p[i]), float(r[i])


def dev_season_masks(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """시간 순서대로 Dev 겨울을 선택, 다음 여름을 검증에 쓴다."""
    season = d["season"].astype(str)
    winter = season.eq("2024-25_winter").to_numpy()
    summer = season.eq("2025_summer").to_numpy()
    if not winter.any() or not summer.any() or not (winter | summer).all():
        raise ValueError("Dev에는 2024-25 겨울과 2025 여름이 모두 있어야 합니다.")
    return winter, summer


def evaluate(name: str, h: int, dv: pd.DataFrame, s: np.ndarray, thr: float, n_days: int,
             alert: np.ndarray | None = None) -> dict:
    y = dv["y"].to_numpy()
    a = s >= thr if alert is None else alert
    tp, fp = int((a & (y == 1)).sum()), int((a & (y == 0)).sum())
    fn, tn = int((~a & (y == 1)).sum()), int((~a & (y == 0)).sum())
    bad = dv["weather_class"].astype(str).eq("악천후").to_numpy()
    tp_b, fp_b, fn_b = int((a & bad & (y == 1)).sum()), int((a & bad & (y == 0)).sum()), int((~a & bad & (y == 1)).sum())
    binary = len(np.unique(s)) <= 2
    # 헛경보 단위 (코드 검토 4번): 위 fp 는 도로 · 방향 × 목표 시각 칸 수. 같은 날 같은 도로는 한 번으로 묶은 수도 함께 적는다
    fp_mask = a & (y == 0)
    fp_road_days = int(dv.loc[fp_mask, ["road_id", "direction", "date"]].drop_duplicates().shape[0])
    alert_road_days = int(dv.loc[a, ["road_id", "direction", "date"]].drop_duplicates().shape[0])
    return {"fp_road_days_per_day": fp_road_days / n_days, "alert_road_days_per_day": alert_road_days / n_days,
            "model": name, "horizon": h, "threshold": thr,
            "pr_auc": np.nan if binary else float(average_precision_score(y, s)),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "alerts_per_day": (tp + fp) / n_days, "false_alarms_per_day": fp / n_days,
            "bad_precision": tp_b / max(tp_b + fp_b, 1), "bad_recall": tp_b / max(tp_b + fn_b, 1),
            "positives": int(y.sum()), "rows": len(y), "dev_days": n_days}


def fit_models(tr: pd.DataFrame, dv: pd.DataFrame, h: int) -> tuple[dict, dict]:
    """h 하나에 대해 네 모델 점수(Dev)와 부가 정보."""
    scores, info = {}, {}
    scores["기준선: 특보면 경보"] = ((dv["warn_snow"] > 0) | (dv["warn_rain"] > 0)).astype(float).to_numpy()
    scores["기준선: 지금 느리면 경보"] = dv["ratio_t0"].to_numpy()

    def logit_scores(num):
        pre = ColumnTransformer([("num", StandardScaler(), num),
                                 ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL)])
        m = make_pipeline(pre, LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0))
        m.fit(tr[num + CATEGORICAL], tr["y"])
        coef = pd.Series(m[-1].coef_[0], index=m[0].get_feature_names_out())
        return m.predict_proba(dv[num + CATEGORICAL])[:, 1], coef

    scores["로지스틱 회귀"], coef = logit_scores(NUMERIC)
    full = NUMERIC + FCST + INTER
    scores["로지스틱 + 예보 · 겹침"], coef_f = logit_scores(full)
    info["logit_top"] = coef_f.reindex(coef_f.abs().sort_values(ascending=False).index).head(15).round(3).to_dict()
    info["logit_weather_coef"] = coef_f[[f"num__{c}" for c in FCST + INTER + ["precip_3h", "new_snow_3h_cm", "warn_rain", "warn_snow"]]].round(3).to_dict()

    pre_t = ColumnTransformer([("num", "passthrough", full),
                               ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL)])
    winter, _ = dev_season_masks(dv)
    best = None
    for depth in TREE_DEPTHS:
        tree = make_pipeline(pre_t, DecisionTreeClassifier(max_depth=depth, min_samples_leaf=200,
                                                           class_weight="balanced", random_state=SEED))
        tree.fit(tr[full + CATEGORICAL], tr["y"])
        s = tree.predict_proba(dv[full + CATEGORICAL])[:, 1]
        ap = average_precision_score(dv.loc[winter, "y"], s[winter])
        info.setdefault("tree_depth_pr_auc", {})[depth] = round(float(ap), 4)
        if best is None or ap > best[0]:
            best = (ap, depth, s, tree)
    scores["결정트리 + 예보 · 겹침"] = best[2]
    info["tree_depth"] = best[1]
    names = [n.replace("num__", "").replace("cat__", "") for n in best[3][0].get_feature_names_out()]
    info["tree_rules"] = export_text(best[3][-1], feature_names=names, max_depth=3, decimals=2)
    return scores, info


def lead_time(dv: pd.DataFrame, alerts: dict) -> pd.DataFrame:
    """정체 '시작' 시각(같은 날 같은 출퇴근 묶음에서 처음 막힌 시각)을 3시간 전 / 1시간 전에 잡았나.

    alerts[모델][h] = Dev 행 순서의 경보 여부(bool). 3시간 전 경보가 있으면 3시간, 없고 1시간 전이 있으면 1시간, 둘 다 없으면 놓침.
    """
    base = dv[dv["horizon"] == 1][["date", "target_hour", "road_id", "direction", "y", "weather_class"]].copy()
    base["block"] = np.where(base["target_hour"].astype(int) < 12, "출근", "퇴근")
    base = base.assign(_h=base["target_hour"].astype(int)).sort_values(["road_id", "direction", "date", "_h"])
    prev = base.groupby(["road_id", "direction", "date", "block"])["y"].shift(1).fillna(0)
    starts = base[(base["y"] == 1) & (prev == 0)][["date", "target_hour", "road_id", "direction", "weather_class"]]
    # 코드 검토 5번: 08 · 18시는 직전 시각(07 · 17시)의 정체 정답이 없어서, 이전부터 막혀 있었는지 모른다.
    # 그래서 이 표의 '시작'은 '출퇴근 구간에서 처음 관측한 정체'이고, 경계 시각을 뺀 줄도 따로 적는다
    starts["boundary"] = starts["target_hour"].astype(int).isin([8, 18])
    rows = []
    for model, by_h in alerts.items():
        hit = {}
        for h, a in by_h.items():
            k = dv.loc[dv["horizon"] == h, ["date", "target_hour", "road_id", "direction"]].assign(alert=a)
            hit[h] = starts.merge(k, on=["date", "target_hour", "road_id", "direction"], how="left")["alert"].fillna(False).to_numpy()
        lead = np.where(hit[3], "3시간 전", np.where(hit[1], "1시간 전", "놓침"))
        bad = starts["weather_class"].astype(str).eq("악천후").to_numpy()
        inner = ~starts["boundary"].to_numpy()
        for scope, m in [("전체", np.ones(len(starts), bool)), ("악천후", bad),
                         ("08 · 18시 시작 뺌", inner), ("악천후 · 08 · 18시 시작 뺌", bad & inner)]:
            vc = pd.Series(lead[m]).value_counts(normalize=True)
            rows.append({"model": model, "scope": scope, "starts": int(m.sum()),
                         "3시간 전": vc.get("3시간 전", 0.0), "1시간 전": vc.get("1시간 전", 0.0), "놓침": vc.get("놓침", 0.0)})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    d = prepare(load_data())
    metrics, curves, info_all, keep = [], [], {}, []
    alerts = {}
    crit_alerts, crit_rows = {}, []          # 경보 기준 정리 (로지스틱 회귀, 겨울에서 고른 두 운영 기준)
    for h in [1, 3]:
        tr = d[(d["model_set"] == "train") & (d["horizon"] == h)]
        dv = d[(d["model_set"] == "dev") & (d["horizon"] == h)]
        winter, summer = dev_season_masks(dv)
        summer_dv = dv.loc[summer]
        n_days = summer_dv["date"].nunique()
        scores, info = fit_models(tr, dv, h)
        info_all[h] = {k: v for k, v in info.items() if k != "tree_rules"}
        if h == 3:
            (TABLE_DIR / "alert_tree_rules.txt").write_text(info["tree_rules"], encoding="utf-8")
        sc = dv[KEY + ["y", "weather_class"]].copy()
        for name, s in scores.items():
            y = dv["y"].to_numpy()
            if name == "기준선: 특보면 경보":
                thr = 1.0
            else:
                thr, _, _ = threshold_rule(y[winter], s[winter])
                p, r, _ = precision_recall_curve(y[summer], s[summer])
                idx = np.unique(np.linspace(0, len(p) - 1, 200).astype(int))
                curves.append(pd.DataFrame({"model": name, "horizon": h, "precision": p[idx], "recall": r[idx]}))
            metrics.append(evaluate(name, h, summer_dv, s[summer], thr, n_days) | {"rule": "겨울 선택 → 여름 검증", "selection_season": "2024-25_winter", "evaluation_season": "2025_summer"})
            alerts.setdefault(name, {})[h] = s[summer] >= thr
            sc[name] = s
            # 같은 경보 수 비교 (여름): 특보 기준선과 경보 개수를 맞춘 임계값. 경보를 늘리지 않고 몇 개를 더 잡나
            if name == "기준선: 특보면 경보":
                n_base = int((s[summer] >= 1.0).sum())
            else:
                thr_b = float(np.sort(s[summer])[::-1][n_base - 1])
                metrics.append(evaluate(name, h, summer_dv, s[summer], thr_b, n_days)
                               | {"rule": "특보와 같은 경보 수 (여름)", "evaluation_season": "2025_summer"})
            if name == CRITERIA_MODEL:
                win_days = dv.loc[winter, "date"].nunique()
                base_w = scores["기준선: 특보면 경보"][winter]
                budget = int(round((base_w >= 1.0).sum() / win_days * win_days))   # 겨울 특보 경보 수와 같게
                thr_budget = float(np.sort(s[winter])[::-1][max(budget, 1) - 1])
                for rule, t_rule in [("재현율 80% (겨울에서 정함)", thr), ("특보와 같은 경보량 (겨울에서 정함)", thr_budget)]:
                    r = evaluate(name, h, summer_dv, s[summer], t_rule, n_days)
                    crit_rows.append({"rule": rule, "horizon": h, "threshold": t_rule, **{k: r[k] for k in [
                        "precision", "recall", "bad_precision", "bad_recall", "alerts_per_day", "false_alarms_per_day",
                        "alert_road_days_per_day", "fp_road_days_per_day"]}})
                    crit_alerts.setdefault(rule, {})[h] = s[summer] >= t_rule
        keep.append(sc)
    met = pd.DataFrame(metrics)
    order = ["기준선: 특보면 경보", "기준선: 지금 느리면 경보", "로지스틱 회귀", "로지스틱 + 예보 · 겹침", "결정트리 + 예보 · 겹침"]
    met["model"] = pd.Categorical(met["model"], order, ordered=True)
    met = met.sort_values(["horizon", "rule", "model"])
    dv_all = pd.concat([d[(d["model_set"] == "dev") & (d["season"] == "2025_summer") & (d["horizon"] == h)] for h in [1, 3]])
    lt = lead_time(dv_all, alerts)
    # 경보 기준 정리 표: 기준마다 여름 검증 성능 + 선행 시간 (정체 시작을 3시간 전 / 1시간 전에 잡은 비율)
    crit = pd.DataFrame(crit_rows)
    lt_c = lead_time(dv_all, crit_alerts).rename(columns={"model": "rule"})
    lt_c = lt_c[lt_c["scope"].isin(["전체", "악천후"])].pivot(index="rule", columns="scope", values=["3시간 전", "1시간 전", "놓침"])
    lt_c.columns = [f"lead_{a}_{b}" for a, b in lt_c.columns]
    crit = crit.merge(lt_c.reset_index(), on="rule", how="left")
    crit.insert(0, "model", CRITERIA_MODEL)
    crit.to_csv(TABLE_DIR / "alert_criteria_dev.csv", index=False, encoding="utf-8-sig")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    met.to_csv(TABLE_DIR / "alert_dev_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(curves).to_csv(TABLE_DIR / "alert_dev_pr_curve.csv", index=False, encoding="utf-8-sig")
    lt.to_csv(TABLE_DIR / "alert_lead_time_dev.csv", index=False, encoding="utf-8-sig")
    SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(keep).to_parquet(SCORE_PATH, index=False)
    log = {"run_at": datetime.now(KST).isoformat(timespec="seconds"), "rows": {
               f"{ms}_h{h}": int(((d["model_set"] == ms) & (d["horizon"] == h)).sum()) for ms in ["train", "dev"] for h in [1, 3]},
           "positive_rate": {ms: round(float(d.loc[d["model_set"] == ms, "y"].mean()), 4) for ms in ["train", "dev"]},
           "min_recall": MIN_RECALL, "features": NUMERIC + FCST + INTER + CATEGORICAL,
           "forecast_rule": "발표 시각 + 1시간 <= t 인 가장 최근 발표",
           "selection_season": "2024-25_winter", "evaluation_season": "2025_summer",
           "forecast_missing_rate": round(float(d["fc_missing"].mean()), 4), "models": info_all,
           "seconds": round(time.time() - t0, 1)}
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(met.round(3).to_string(index=False))
    print(lt.round(3).to_string(index=False))
    print(json.dumps(info_all, ensure_ascii=False, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
