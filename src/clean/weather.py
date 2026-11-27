"""
날씨 자료 정리 · 눈 · 비 사건 · 연구 기간 · 악천후일 목록 (B 담당)

흐름
1. load_asos()          : 원본 ASOS 시간자료(API · 포털 CSV)를 공통 열 이름으로 읽기
2. add_season()         : 겨울 · 여름 시즌과 학습 · 평가 구분 붙이기
3. extract_events()     : 비 · 눈이 온 시간을 이어 붙여 '사건'으로 묶고 종류 · 강도 판정
4. first_last()         : 시즌마다 첫눈 · 마지막 눈, 첫 큰비 · 마지막 큰비 날짜
5. period_options()     : 학습 3년 평균 등으로 연구 기간 후보를 만들고, 후보마다 사건이 얼마나 들어오는지 계산
6. select_days()        : 악천후일 · 비교일 목록 (C의 교통량 수집용, 개수 제한)
7. save_outputs()       : 학습 / 평가로 나눠 저장 + config/periods.yaml

기준 숫자는 config/periods.yaml 의 thresholds 가 우선이다 (팀의 약속).
아래 THRESHOLDS 는 yaml에 없는 키에만 쓰는 기본값이고, 둘이 다르면 경고를 출력한다.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))  # 실행 PC 시간대와 상관없이 한국 시간으로 기록
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = {"train": "train_dev", "eval": "holdout"}  # split 값 → data/shared 폴더 · 파일 이름 (io_utils.SPLIT_DIR 과 같음)

# ── 기준값 기본값 (실제 값은 config/periods.yaml 의 thresholds 가 우선) ──────
THRESHOLDS = {
    "snow_temp_c": 1.0,          # 강수가 있고 기온이 이 값 이하면 눈으로 본다
    "event_gap_h": 3,            # 비 · 눈이 이 시간보다 오래 끊기면 다른 사건
    "snow_light_cm": 1.0,        # 신적설 합 1cm 미만 = 약한 눈
    "snow_heavy_cm": 5.0,        # 5cm 이상 = 많은 눈 (사이 = 보통 눈)
    "rain_strong_mmh": 5.0,      # 시간당 5mm 이상 = 강한 비
    "rain_heavy_mmh": 30.0,      # 시간당 30mm 이상 또는
    "rain_heavy_total_mm": 80.0, # 사건 총 80mm 이상 = 호우
    "summer_day_mm": 30.0,       # 여름 기간을 정할 때 쓰는 '큰비 날' (하루 30mm 이상)
    "congestion_link_ratio": 1.5,        # 링크 정체: 통행시간 ÷ 평소값 > 1.5 (속도가 평소의 2/3 미만)
    "congestion_road_length_ratio": 0.3, # 도로 정체: 정체 링크 거리가 도로 길이의 30% 이상
    "freezing_temp_c": 0.5,      # 비 사건 중 기온이 이 값 이하인 시간이 있으면 어는비 표시
    "impact_after_snow_h": 12,   # 눈이 그친 뒤 영향을 보는 시간
    "impact_after_rain_h": 3,    # 비가 그친 뒤 영향을 보는 시간
}
# 출근 07~09시 · 퇴근 17~20시. 시각은 '그 시각까지 1시간'을 뜻한다 (끝 시각 기준).
#   TOPIS "~08시" 열 = 07:00~08:00 평균 (A가 보고서와 대조해 확인, 11/20)
#   ASOS 08시 강수량 = 07:00~08:00 누적 → 두 자료 모두 끝 시각 기준이라 같은 시각끼리 바로 붙인다.
#   (TOPIS "~24시"는 다음 날 00시와 같은 칸)
COMMUTE_HOURS = [8, 9, 18, 19, 20]  # 출근: ~08 · ~09시 / 퇴근: ~18 · ~19 · ~20시
BASELINE_DAYS = 28  # 평소값 구간: 연구 기간 시작 28일(4주) 전부터 (periods 의 baseline_start)


def apply_yaml_thresholds(th: dict, path: Path | None = None) -> dict:
    """config/periods.yaml 의 thresholds 로 th 를 덮는다 (제자리 수정).

    yaml 이 팀의 약속이라 yaml 값이 이긴다. 코드 값과 다르면 경고를 출력한다.
    yaml 이 없거나 thresholds 가 없으면 코드 기본값을 그대로 쓴다.
    """
    path = path or ROOT / "config" / "periods.yaml"
    try:
        import yaml
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (FileNotFoundError, ImportError):
        return th
    for key, value in (cfg.get("thresholds") or {}).items():
        if key in th and th[key] != value:
            print(f"⚠️ 기준값 {key}: 코드 기본값 {th[key]} 대신 periods.yaml 값 {value} 사용")
        th[key] = value
    return th

# 학습 3년 · 평가 시즌 (겨울은 11월이 속한 해로 부른다: W2022 = 2022~23 겨울)
TRAIN_SEASONS = ["W2022", "W2023", "W2024", "S2023", "S2024", "S2025"]
EVAL_SEASONS = ["W2025", "S2026"]
SEASON_NAMES = {
    "W2022": "2022~23 겨울", "W2023": "2023~24 겨울", "W2024": "2024~25 겨울", "W2025": "2025~26 겨울",
    "S2023": "2023 여름", "S2024": "2024 여름", "S2025": "2025 여름", "S2026": "2026 여름",
}
# 사건을 찾는 넓은 창 (연구 기간은 이 안에서 정한다)
WINTER_WINDOW = ((10, 1), (4, 30))  # 10/1 ~ 다음 해 4/30
SUMMER_WINDOW = ((5, 1), (9, 30))   # 5/1 ~ 9/30

# 현상번호(국내식)는 두 자리 코드가 이어 붙은 문자열이다 (예: '1905' = 19 + 05).
# 실제 자료에서 비 오는 시간엔 01, 2025-12-04 눈 오는 시간엔 05가 찍힌 것을 확인했다. 최종은 기상청 코드표로 확인.
RAIN_CODES = {"01"}
SNOW_CODES = {"05"}

apply_yaml_thresholds(THRESHOLDS)  # 함수 기본값(th=THRESHOLDS)도 같은 dict 를 보므로 yaml 값이 모든 계산에 쓰인다

INTENSITY_ORDER = {"많은 눈": 6, "호우": 5, "보통 눈": 4, "강한 비": 3, "약한 눈": 2, "약한 비": 1, "눈(안 쌓임)": 1}
BAD_INTENSITIES = {"많은 눈", "보통 눈", "호우", "강한 비"}

# ── 1. 읽기 ───────────────────────────────────────────────────────
API_COLS = {
    "tm": "datetime", "ta": "temp_c", "rn": "precip_mm", "ws": "wind_ms", "hm": "humidity",
    "dsnw": "snow_depth_cm", "hr3Fhsc": "new_snow_3h_cm", "vs": "visibility_10m",
    "gndSttCd": "ground_state", "dmstMtphNo": "phenomenon",
}
PORTAL_COLS = {
    "일시": "datetime", "기온(°C)": "temp_c", "강수량(mm)": "precip_mm", "풍속(m/s)": "wind_ms",
    "습도(%)": "humidity", "적설(cm)": "snow_depth_cm", "3시간신적설(cm)": "new_snow_3h_cm",
    "시정(10m)": "visibility_10m", "지면상태(지면상태코드)": "ground_state", "현상번호(국내식)": "phenomenon",
}
NUM_COLS = ["temp_c", "precip_mm", "wind_ms", "humidity", "snow_depth_cm", "new_snow_3h_cm", "visibility_10m"]


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc, index_col=False, dtype=str)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 읽을 수 없음: {path}")


def load_asos(raw_dirs=None) -> pd.DataFrame:
    """data/raw/asos (API) 와 data/raw/asos_manual (포털 CSV) 를 모두 읽어 한 표로 만든다."""
    raw_dirs = raw_dirs or [ROOT / "data/raw/asos", ROOT / "data/raw/asos_manual"]
    frames = []
    for d in map(Path, raw_dirs):
        for f in sorted(d.glob("*.csv")):
            df = _read_csv(f)
            mapping = API_COLS if "tm" in df.columns else PORTAL_COLS
            df = df.rename(columns=mapping)
            keep = [c for c in set(mapping.values()) if c in df.columns]
            frames.append(df[keep])
    if not frames:
        raise FileNotFoundError("ASOS 파일이 없습니다. crawling/kma_asos.py 를 먼저 실행하세요.")
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    if "phenomenon" not in df.columns:
        df["phenomenon"] = np.nan
    df = df.drop_duplicates("datetime").sort_values("datetime")
    # 빠진 시간도 한 줄씩 만들어 두고, 빠졌다는 표시를 남긴다
    full = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    df = df.set_index("datetime").reindex(full)
    df.index.name = "datetime"
    df["missing_obs"] = df["temp_c"].isna()
    # 강수 · 신적설 빈칸 = 비 · 눈 없음
    df["precip_mm"] = df["precip_mm"].fillna(0.0)
    df["new_snow_3h_cm"] = df["new_snow_3h_cm"].fillna(0.0)
    df["snow_depth_cm"] = df["snow_depth_cm"].fillna(0.0)
    return df.reset_index()


# ── 2. 시즌 ───────────────────────────────────────────────────────
def season_of(ts: pd.Timestamp) -> str | None:
    m = ts.month
    if m >= 10:
        return f"W{ts.year}"
    if m <= 4:
        return f"W{ts.year - 1}"
    return f"S{ts.year}"


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["season"] = df["datetime"].map(season_of)
    df["split"] = np.where(df["season"].isin(TRAIN_SEASONS), "train",
                           np.where(df["season"].isin(EVAL_SEASONS), "eval", "other"))
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    return df


# ── 3. 사건 ───────────────────────────────────────────────────────
def extract_events(df: pd.DataFrame, th=THRESHOLDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """비 · 눈이 온 시간을 이어 붙여 사건으로 묶는다. (시간표, 사건표) 를 돌려준다."""
    df = df.copy()
    tokens = df["phenomenon"].map(
        lambda x: set() if pd.isna(x) else {str(x)[i:i + 2] for i in range(0, len(str(x)), 2)})
    code_rain = tokens.map(lambda t: bool(t & RAIN_CODES))
    code_snow = tokens.map(lambda t: bool(t & SNOW_CODES))
    depth_up = df["snow_depth_cm"].diff().fillna(0) > 0
    # 눈이 오는 동안에는 시간 강수량이 비어 있는 경우가 많아, 현상번호 · 적설 증가도 함께 본다
    wet = (df["precip_mm"] > 0) | (df["new_snow_3h_cm"] > 0) | code_rain | code_snow | depth_up
    is_snow = code_snow | depth_up | (df["new_snow_3h_cm"] > 0) | ((df["precip_mm"] > 0) & (df["temp_c"] <= th["snow_temp_c"]))
    df["precip_type"] = np.where(~wet, "", np.where(is_snow, "snow", "rain"))

    wet_times = df.loc[wet, "datetime"]
    gap = wet_times.diff() > pd.Timedelta(hours=th["event_gap_h"])
    event_no = gap.cumsum()
    df["event_id"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[wet, "event_id"] = event_no.values

    rows = []
    for eid, g in df[wet].groupby("event_id"):
        start, end = g["datetime"].min(), g["datetime"].max()
        # 3시간 신적설은 3시간 뒤에 기록되므로 끝 + 3시간까지 합친다
        win = df[(df["datetime"] >= start) & (df["datetime"] <= end + pd.Timedelta(hours=3))]
        new_snow = win["new_snow_3h_cm"].sum()
        snow_share = (g["precip_type"] == "snow").mean()
        total = g["precip_mm"].sum()
        max_h = g["precip_mm"].max()
        if new_snow > 0:
            kind = "snow"
            intensity = ("약한 눈" if new_snow < th["snow_light_cm"]
                         else "많은 눈" if new_snow >= th["snow_heavy_cm"] else "보통 눈")
        elif snow_share >= 0.5:
            kind, intensity = "snow", "눈(안 쌓임)"
        else:
            kind = "rain"
            if max_h >= th["rain_heavy_mmh"] or total >= th["rain_heavy_total_mm"]:
                intensity = "호우"
            elif max_h >= th["rain_strong_mmh"]:
                intensity = "강한 비"
            else:
                intensity = "약한 비"
        freezing = kind == "rain" and (g["temp_c"] <= th["freezing_temp_c"]).any()
        rows.append({
            "event_id": int(eid), "start": start, "end": end,
            "hours": int((end - start) / pd.Timedelta(hours=1)) + 1,
            "type": kind, "intensity": intensity, "freezing_rain": bool(freezing),
            "precip_total_mm": round(total, 1), "precip_max_mmh": round(max_h, 1),
            "new_snow_cm": round(new_snow, 1), "max_snow_depth_cm": round(win["snow_depth_cm"].max(), 1),
            "min_temp_c": round(g["temp_c"].min(), 1),
            "season": season_of(start),
        })
    ev = pd.DataFrame(rows)
    ev["split"] = np.where(ev["season"].isin(TRAIN_SEASONS), "train",
                           np.where(ev["season"].isin(EVAL_SEASONS), "eval", "other"))
    df = df.merge(ev[["event_id", "intensity"]].rename(columns={"intensity": "event_intensity"}),
                  on="event_id", how="left")
    return df, ev


# ── 4. 첫 · 마지막 날 ────────────────────────────────────────────
def _season_origin(season: str) -> date:
    y = int(season[1:])
    return date(y, 10, 1) if season.startswith("W") else date(y, 5, 1)


def _in_window(ts: pd.Timestamp, season: str) -> bool:
    (sm, sd), (em, ed) = WINTER_WINDOW if season.startswith("W") else SUMMER_WINDOW
    y = int(season[1:])
    lo = date(y, sm, sd)
    hi = date(y + 1, em, ed) if season.startswith("W") else date(y, em, ed)
    return lo <= ts.date() <= hi


def first_last(df: pd.DataFrame, ev: pd.DataFrame, th=THRESHOLDS) -> pd.DataFrame:
    """시즌마다 첫 · 마지막 날.
    겨울: 쌓인 눈(신적설 0.1cm 이상) 사건 기준. 참고로 안 쌓인 눈까지 포함한 날짜도 함께 적는다.
    여름: 하루 강수량이 summer_day_mm 이상인 '큰비 날' 기준.
    """
    rows = []
    for season in TRAIN_SEASONS + EVAL_SEASONS:
        if season.startswith("W"):
            e = ev[(ev["season"] == season) & (ev["type"] == "snow")]
            e = e[e["start"].map(lambda t: _in_window(t, season))]
            acc = e[e["new_snow_cm"] > 0]
            first = acc["start"].min() if len(acc) else pd.NaT
            last = acc["end"].max() if len(acc) else pd.NaT
            rows.append({"season": season, "name": SEASON_NAMES[season], "kind": "겨울 (쌓인 눈)",
                         "first": first, "last": last, "n_events": len(acc),
                         "first_any_snow": e["start"].min() if len(e) else pd.NaT,
                         "last_any_snow": e["end"].max() if len(e) else pd.NaT})
        else:
            d = df[df["season"] == season].groupby("date")["precip_mm"].sum()
            d = d[d >= th["summer_day_mm"]]
            d = d[[(_in_window(pd.Timestamp(x), season)) for x in d.index]]
            rows.append({"season": season, "name": SEASON_NAMES[season],
                         "kind": f"여름 (하루 {th['summer_day_mm']:.0f}mm 이상)",
                         "first": pd.Timestamp(min(d.index)) if len(d) else pd.NaT,
                         "last": pd.Timestamp(max(d.index)) if len(d) else pd.NaT,
                         "n_events": len(d), "first_any_snow": pd.NaT, "last_any_snow": pd.NaT})
    out = pd.DataFrame(rows)
    out["split"] = np.where(out["season"].isin(TRAIN_SEASONS), "train", "eval")
    out["first_offset"] = [(f.date() - _season_origin(s)).days if pd.notna(f) else np.nan
                           for f, s in zip(out["first"], out["season"])]
    out["last_offset"] = [(l.date() - _season_origin(s)).days if pd.notna(l) else np.nan
                          for l, s in zip(out["last"], out["season"])]
    return out


# ── 5. 연구 기간 후보 ─────────────────────────────────────────────
def _offset_to_md(offset: float, kind: str) -> str:
    base = date(2021, 10, 1) if kind == "W" else date(2021, 5, 1)  # 윤년이 아닌 기준 해
    d = base + timedelta(days=int(round(offset)))
    return f"{d.month:02d}-{d.day:02d}"


def _coverage(df_days: pd.DataFrame, kind: str, lo: float, hi: float, split: str) -> float:
    sub = df_days[(df_days["kind"] == kind) & (df_days["split"] == split)]
    if sub.empty:
        return np.nan
    return float(((sub["offset"] >= lo) & (sub["offset"] <= hi)).mean())


def period_options(df: pd.DataFrame, ev: pd.DataFrame, fl: pd.DataFrame, th=THRESHOLDS) -> pd.DataFrame:
    """학습 3년의 첫 · 마지막 날로 연구 기간 후보를 만들고, 후보마다 악천후 날이 얼마나 들어오는지 본다."""
    # 커버리지 계산 대상: 겨울은 쌓인 눈 사건 날, 여름은 큰비 날 (학습 · 평가 모두)
    rec = []
    for _, e in ev[(ev["type"] == "snow") & (ev["new_snow_cm"] > 0)].iterrows():
        if e["season"] in TRAIN_SEASONS + EVAL_SEASONS and e["season"].startswith("W"):
            rec.append({"kind": "W", "split": "train" if e["season"] in TRAIN_SEASONS else "eval",
                        "offset": (e["start"].date() - _season_origin(e["season"])).days})
    daily = df[df["season"].isin(TRAIN_SEASONS + EVAL_SEASONS)].groupby(["season", "date"])["precip_mm"].sum()
    for (s, d), v in daily.items():
        if s.startswith("S") and v >= th["summer_day_mm"]:
            rec.append({"kind": "S", "split": "train" if s in TRAIN_SEASONS else "eval",
                        "offset": (d - _season_origin(s)).days})
    days = pd.DataFrame(rec)

    rows = []
    for kind, label in (("W", "겨울"), ("S", "여름")):
        t = fl[(fl["split"] == "train") & fl["season"].str.startswith(kind)]
        if t["first_offset"].isna().all():
            continue
        mf, ml = t["first_offset"].mean(), t["last_offset"].mean()
        lo_min, hi_max = t["first_offset"].min(), t["last_offset"].max()
        cands = [
            ("평균", mf, ml),
            ("평균 ± 1주", mf - 7, ml + 7),
            ("평균 ± 2주", mf - 14, ml + 14),
            ("가장 이른 ~ 가장 늦은", lo_min, hi_max),
        ]
        for name, lo, hi in cands:
            rows.append({
                "season_kind": label, "option": name,
                "start": _offset_to_md(lo, kind), "end": _offset_to_md(hi, kind),
                "days": int(round(hi - lo)) + 1,
                "train_coverage": _coverage(days, kind, lo, hi, "train"),
                "eval_coverage": _coverage(days, kind, lo, hi, "eval"),
            })
    return pd.DataFrame(rows)


def recommend_period(opts: pd.DataFrame, min_coverage: float = 0.9) -> dict:
    """학습 기간 악천후 날의 min_coverage 이상이 들어오는 후보 중 가장 짧은 것을 고른다."""
    out = {}
    for label in opts["season_kind"].unique():
        o = opts[opts["season_kind"] == label]
        ok = o[o["train_coverage"] >= min_coverage]
        pick = (ok if len(ok) else o).sort_values("days").iloc[0]
        out[label] = {"option": pick["option"], "start": pick["start"], "end": pick["end"],
                      "train_coverage": round(float(pick["train_coverage"]), 3),
                      "eval_coverage": None if pd.isna(pick["eval_coverage"]) else round(float(pick["eval_coverage"]), 3)}
    return out


def in_period(ts: pd.Timestamp, season: str, period: dict) -> bool:
    """MM-DD 형식의 연구 기간 안인지 (겨울은 해를 넘긴다)"""
    winter = season.startswith("W")
    p = period["겨울" if winter else "여름"]
    y = int(season[1:])
    sm, sd = map(int, p["start"].split("-"))
    em, ed = map(int, p["end"].split("-"))
    sy = y if (not winter or sm >= 7) else y + 1
    ey = y + 1 if (winter and em < 7) else y
    if em == 2 and ed == 29 and ey % 4 != 0:
        ed = 28
    return date(sy, sm, sd) <= ts.date() <= date(ey, em, ed)


def build_periods(period: dict) -> list[dict]:
    """research_period(MM-DD)로 시즌별 날짜 목록(periods)을 만든다. C 병합 코드가 이 목록을 읽는다.

    in_period() 와 같은 규칙으로 해를 붙이고, baseline_start 는 시작일 BASELINE_DAYS 일 전.
    """
    rows = []
    for s in TRAIN_SEASONS + EVAL_SEASONS:
        winter = s.startswith("W")
        y = int(s[1:])
        p = period["겨울" if winter else "여름"]
        sm, sd = map(int, p["start"].split("-"))
        em, ed = map(int, p["end"].split("-"))
        sy = y if (not winter or sm >= 7) else y + 1
        ey = y + 1 if (winter and em < 7) else y
        if em == 2 and ed == 29 and ey % 4 != 0:
            ed = 28
        start = date(sy, sm, sd)
        rows.append({"season": f"{y}-{str(y + 1)[2:]}_winter" if winter else f"{y}_summer",
                     "split": "train" if s in TRAIN_SEASONS else "eval",
                     "start": start, "end": date(ey, em, ed),
                     "baseline_start": start - timedelta(days=BASELINE_DAYS)})
    return sorted(rows, key=lambda r: r["start"])


# ── 6. 악천후일 · 비교일 ─────────────────────────────────────────
def _holiday_set(years):
    try:
        import holidays  # pip install holidays
        kr = holidays.KR(years=list(years))
        return set(kr.keys())
    except ImportError:
        print("⚠️ holidays 패키지가 없어 공휴일을 빼지 못했습니다. pip install holidays")
        return set()


MUST_INCLUDE = [date(2025, 12, 4)]  # 검증 사례는 항상 넣는다


def mark_impact(df: pd.DataFrame, ev: pd.DataFrame, th=THRESHOLDS) -> pd.DataFrame:
    """impact(영향 시간의 사건 강도) · impact_rank · snow_phase(falling / after / 빈값) 열을 붙인다.

    겹치는 사건은 가장 센 사건이 impact 를 정하고, snow_phase 도 그 사건을 따른다.
    """
    df = df.copy()
    impact = pd.Series("", index=df.index)
    snow_phase = pd.Series("", index=df.index)
    rank = pd.Series(0, index=df.index)
    bad_ev = ev[ev["intensity"].isin(BAD_INTENSITIES)]
    for _, e in bad_ev.iterrows():
        after = th["impact_after_snow_h"] if e["type"] == "snow" else th["impact_after_rain_h"]
        m = (df["datetime"] >= e["start"]) & (df["datetime"] <= e["end"] + pd.Timedelta(hours=after))
        r = INTENSITY_ORDER[e["intensity"]]
        upd = m & (rank < r)
        impact[upd], rank[upd] = e["intensity"], r
        # impact 를 정한 사건(가장 센 사건)을 따라 눈 단계를 정한다. 비 사건이 덮으면 빈값으로 되돌린다
        snow_phase[upd] = (np.where(df.loc[upd, "datetime"] <= e["end"], "falling", "after")
                           if e["type"] == "snow" else "")
    df["impact"], df["impact_rank"], df["snow_phase"] = impact, rank, snow_phase
    return df


def select_days(df: pd.DataFrame, ev: pd.DataFrame, period: dict, max_train: int = 30,
                max_eval: int = 15, th=THRESHOLDS) -> pd.DataFrame:
    """악천후일과 비교일 목록.
    악천후일: 연구 기간 안 평일 중, 보통 이상 눈 · 강한 비 · 호우의 영향 시간(그친 뒤 눈 12시간 · 비 3시간 포함)이
             출퇴근 시간과 겹친 날. 강한 순으로 학습 max_train 일 · 평가 max_eval 일까지
             (C가 교통량 API로 받을 수 있는 양에 맞춘 제한). 2025-12-04는 항상 넣는다.
    비교일: 같은 시즌 · 같은 요일 ±1주 · ±2주 중, 그날과 전날 밤에 비 · 눈이 없고 쌓인 눈도 없는 평일.
    snow_phase: impact 가 눈 사건에서 왔을 때 falling(사건 시작 ~ 끝 시각) / after(그친 뒤 영향 시간), 눈이 아니면 빈값.
                impact 는 그대로 두고 옆에 붙인다. 모델 입력이 아니라 채점 · 가중치 대상을 나누는 구분용.
    """
    hol = _holiday_set(range(2022, 2028))
    df = df.copy()
    df["in_period"] = [in_period(t, s, period) if s in TRAIN_SEASONS + EVAL_SEASONS else False
                       for t, s in zip(df["datetime"], df["season"])]

    # 영향 시간 표시
    df = mark_impact(df, ev, th)

    df["weekday"] = df["datetime"].dt.weekday
    workday = (df["weekday"] < 5) & (~df["date"].isin(hol))
    commute = df["hour"].isin(COMMUTE_HOURS)
    cand = df[df["in_period"] & workday & commute & (df["impact_rank"] > 0)]
    bad = (cand.groupby("date").agg(season=("season", "first"), split=("split", "first"),
                                     intensity=("impact", lambda s: max(s, key=lambda x: INTENSITY_ORDER[x])),
                                     rank=("impact_rank", "max"))
           .reset_index())
    daily = df.groupby("date").agg(precip=("precip_mm", "sum"), new_snow=("new_snow_3h_cm", "sum"),
                                   depth=("snow_depth_cm", "max"))
    bad = bad.merge(daily, left_on="date", right_index=True)
    bad = bad.sort_values(["rank", "precip", "new_snow"], ascending=False)
    must = bad[bad["date"].isin(MUST_INCLUDE)]
    rest = bad[~bad["date"].isin(MUST_INCLUDE)]
    ev_rest = rest[rest["split"] == "eval"].head(max(0, max_eval - len(must)))
    bad = pd.concat([rest[rest["split"] == "train"].head(max_train), must, ev_rest])

    # 비교일 후보: 연구 기간 안 평일 · 그날 강수 0 · 쌓인 눈 0 · 전날 18시 이후 강수 0
    dry = daily[(daily["precip"] == 0) & (daily["new_snow"] == 0) & (daily["depth"] == 0)].index
    prev_night = df[(df["hour"] >= 18)].groupby("date")["precip_mm"].sum()
    period_days = set(df.loc[df["in_period"], "date"])
    used = set(bad["date"])
    matches = []
    for _, b in bad.sort_values("date").iterrows():
        found = None
        for k in (-7, 7, -14, 14, -21, 21):
            c = b["date"] + timedelta(days=k)
            pn = prev_night.get(c - timedelta(days=1), 0)
            if (c in dry and pn == 0 and c in period_days and c not in used and c not in hol
                    and season_of(pd.Timestamp(c)) == b["season"]):
                found = c
                break
        if found:
            used.add(found)
        matches.append(found)
    bad = bad.sort_values("date").assign(matched_date=matches)

    out = [pd.DataFrame({"date": bad["date"], "role": "악천후일", "season": bad["season"], "split": bad["split"],
                         "intensity": bad["intensity"], "precip_mm": bad["precip"].round(1),
                         "new_snow_cm": bad["new_snow"].round(1), "matched_date": bad["matched_date"]})]
    comp = bad.dropna(subset=["matched_date"])
    out.append(pd.DataFrame({"date": comp["matched_date"], "role": "비교일", "season": comp["season"],
                             "split": comp["split"], "intensity": "", "precip_mm": 0.0, "new_snow_cm": 0.0,
                             "matched_date": comp["date"]}))
    res = pd.concat(out).sort_values(["split", "date"]).reset_index(drop=True)
    res["weekday"] = pd.to_datetime(res["date"]).dt.day_name(locale=None)
    res["season_name"] = res["season"].map(SEASON_NAMES)
    return res, df


# ── 7. 저장 ───────────────────────────────────────────────────────
HOURLY_COLS = ["datetime", "date", "hour", "season", "split", "in_period", "temp_c", "precip_mm",
               "new_snow_3h_cm", "snow_depth_cm", "wind_ms", "humidity", "visibility_10m", "phenomenon",
               "precip_type", "event_id", "event_intensity", "impact", "snow_phase", "missing_obs"]


def add_warnings(hourly: pd.DataFrame):
    """기상특보 원본(data/raw/kma_warning)이 있으면 서울 특보 상태 열을 붙인다."""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.clean import warnings_seoul as WR
    res = WR.build(hourly["datetime"])
    if res is None:
        print("⚠️ 기상특보 원본이 없어 특보 열 없이 저장합니다 (crawling/kma_warning.py 실행 필요)")
        return hourly, None, None
    state, items, eps = res
    hourly = hourly.drop(columns=[c for c in state.columns if c != "datetime" and c in hourly.columns])
    hourly = hourly.merge(state, on="datetime", how="left")
    eps["season"] = eps["effective_start"].map(season_of)
    eps["split"] = np.where(eps["season"].isin(TRAIN_SEASONS), "train",
                            np.where(eps["season"].isin(EVAL_SEASONS), "eval", "other"))
    return hourly, items, eps


def save_outputs(hourly, ev, fl, opts, rec, days, th=THRESHOLDS, warn_items=None, warn_eps=None):
    import yaml
    proc = ROOT / "data/shared"
    cols = HOURLY_COLS + [c for c in hourly.columns if c.startswith("warn_")]
    for split in ("train", "eval"):
        (proc / SPLIT_DIR[split]).mkdir(parents=True, exist_ok=True)
        if warn_eps is not None:
            warn_eps[warn_eps["split"] == split].to_csv(proc / SPLIT_DIR[split] / f"weather_warnings_{SPLIT_DIR[split]}.csv",
                                                        index=False, encoding="utf-8-sig")
        hourly[hourly["split"] == split][cols].to_csv(
            proc / SPLIT_DIR[split] / f"weather_hourly_{SPLIT_DIR[split]}.csv", index=False, encoding="utf-8-sig")
        ev[ev["split"] == split].to_csv(proc / SPLIT_DIR[split] / f"weather_events_{SPLIT_DIR[split]}.csv",
                                        index=False, encoding="utf-8-sig")
        days[days["split"] == split].to_csv(proc / SPLIT_DIR[split] / f"weather_days_{SPLIT_DIR[split]}.csv",
                                            index=False, encoding="utf-8-sig")
    tables = ROOT / "outputs/tables"
    tables.mkdir(parents=True, exist_ok=True)
    if warn_items is not None:
        warn_items.drop(columns=["seoul_areas"]).assign(seoul_areas=warn_items["seoul_areas"].map(",".join)).to_csv(
            tables / "seoul_warning_items.csv", index=False, encoding="utf-8-sig")
    fl.to_csv(tables / "season_first_last.csv", index=False, encoding="utf-8-sig")
    opts.to_csv(tables / "research_period_options.csv", index=False, encoding="utf-8-sig")

    update_periods_yaml({
        "station": "ASOS 서울(108)",
        "seasons": {"train": [SEASON_NAMES[s] for s in TRAIN_SEASONS],
                    "eval": [SEASON_NAMES[s] for s in EVAL_SEASONS]},
        "research_period": rec,
        "periods": build_periods(rec),
        "commute_hours": COMMUTE_HOURS,
        "thresholds": th,
    })


def update_periods_yaml(updates: dict, path: Path | None = None) -> bool:
    """config/periods.yaml 에서 이 코드가 계산하는 값만 바꾼다.

    - 파일을 통째로 다시 쓰지 않는다: periods · filters · time_definition · files 같은
      다른 키(C · A가 읽는 값)는 그대로 둔다.
    - thresholds 는 키 단위로 합친다 (파일에만 있는 기준값도 보존).
    - 값이 하나도 바뀌지 않으면 파일을 건드리지 않는다 → 주석이 유지된다.
    - 값이 바뀌면 바뀐 키를 출력한다. 이때 yaml 주석은 사라지므로 커밋 전에 git diff로 확인한다.
    반환: 파일을 썼으면 True
    """
    import yaml

    path = path or ROOT / "config" / "periods.yaml"
    old = {}
    if path.exists():
        old = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    new = dict(old)
    for key, value in updates.items():
        if key == "thresholds":
            new[key] = {**(old.get(key) or {}), **value}
        else:
            new[key] = value
    changed = [k for k in new if k != "updated" and new.get(k) != old.get(k)]
    if not changed:
        print(f"periods.yaml: 바뀐 값 없음 → 파일 그대로 둠 ({path.name})")
        return False
    new["updated"] = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(new, f, allow_unicode=True, sort_keys=False)
    print(f"periods.yaml: {', '.join(changed)} 갱신 (주석은 사라짐 · 커밋 전에 git diff config/periods.yaml 확인)")
    return True


def run_all(max_train_days: int = 30, max_eval_days: int = 15, min_coverage: float = 0.9):
    """처음부터 끝까지 한 번에 실행"""
    raw = add_season(load_asos())
    hourly, ev = extract_events(raw)
    fl = first_last(hourly, ev)
    opts = period_options(hourly, ev, fl)
    rec = recommend_period(opts, min_coverage)
    days, hourly = select_days(hourly, ev, rec, max_train=max_train_days, max_eval=max_eval_days)
    hourly, w_items, w_eps = add_warnings(hourly)
    save_outputs(hourly, ev, fl, opts, rec, days, warn_items=w_items, warn_eps=w_eps)
    return {"hourly": hourly, "events": ev, "first_last": fl, "options": opts, "recommended": rec,
            "days": days, "warning_items": w_items, "warning_episodes": w_eps}


if __name__ == "__main__":
    r = run_all()
    print(r["first_last"][["name", "kind", "first", "last", "n_events"]].to_string(index=False))
    print(r["options"].to_string(index=False))
    print(r["recommended"])
    print(r["days"]["role"].value_counts().to_string())
