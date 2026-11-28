"""
예보 · AWS를 서울 전체 값으로 모은 표 (B 담당)

도로마다 권역 · AWS 지점 정보가 아직 없어서(C 링크 좌표 이후), 11/25에는 서울 전체 값을 기본으로 넘긴다.
예보는 권역별 값도 열로 같이 넣어 두어, 나중에 도로별 권역 연결표가 생기면 그 열을 골라 쓸 수 있게 한다.

입력
  data/shared/{train_dev,holdout}/fcst_{split}.csv        발표 × 예보 시각 × 권역 (kma_apihub.py 결과)
  data/shared/{train_dev,holdout}/aws_hourly_{split}.csv  지점 × 시각 (kma_apihub.py 결과)

결과
  data/shared/{train_dev,holdout}/fcst_seoul_{split}.csv  발표(base) × 예보 시각(fcst) × 몇 시간 전(lead_h)
      pcp_mean · pcp_max  1시간 강수량 (mm, 4개 권역 평균 · 최댓값)
      sno_mean · sno_max  1시간 신적설 (cm)
      tmp_mean · tmp_min  기온 (℃)
      pty_rain_regions    비가 오는 권역 수 (강수형태 1 비 · 2 비/눈 · 4 소나기)
      pty_snow_regions    눈이 오는 권역 수 (강수형태 2 비/눈 · 3 눈)
      pcp_se · pcp_ne · pcp_sw · pcp_nw 등  권역별 값 (se 동남 · ne 동북 · sw 서남 · nw 서북)
  data/shared/{train_dev,holdout}/aws_seoul_{split}.csv   시각(datetime, 끝 시각 기준)
      n_stations          그 시각에 자료가 있는 지점 수
      ta_mean · ta_min    기온 (℃)
      rn_hr1_mean · rn_hr1_max  1시간 강수량 (mm)
      rn_day_max          그날 누적 강수량 최댓값 (mm)
      ws_mean · ws_max    풍속 (m/s)
      hm_mean             습도 (%)

규칙
- 예보의 -99 (결측)는 권역 행에서 먼저 빈값으로 바꾼 뒤 평균 · 최댓값을 낸다 (코드 검토 7번: 평균 뒤에 바꾸면 [-99, 5] → -47로 통과).
  값이 있는 권역 수는 pcp_regions 열로 남기고, 강수형태(PTY) 결측은 '비 · 눈 없음'과 따로 pty_missing_regions 로 센다.
- AWS의 -99 (결측)와 강수 · 풍속 · 습도의 음수는 빈값으로 바꾼 뒤 평균 · 최댓값을 낸다. 모든 지점이 빈값이면 결과도 빈값.
- 시각은 모두 끝 시각 기준 (hour=8 은 07:00~08:00).
- 예보는 발표 시각(base)이 t 이전인 행만 써야 한다 (미래 정보 누수 방지). 이 표는 거르지 않고 모두 남긴다.

실행
  python src/clean/weather_seoul.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "shared"
SPLIT_DIR = {"train": "train_dev", "eval": "holdout"}  # split 값 → data/shared 폴더 · 파일 이름 (io_utils.SPLIT_DIR 과 같음)
REGION_CODE = {"동남권": "se", "동북권": "ne", "서남권": "sw", "서북권": "nw"}
AWS_MISSING = -50   # 이 값 이하(-99 등)는 결측


def fcst_seoul(f: pd.DataFrame) -> pd.DataFrame:
    keys = ["base", "fcst", "lead_h"]
    f = f.copy()
    f["region"] = f["region"].map(REGION_CODE)
    for c in ["PCP", "SNO", "TMP", "PTY"]:
        f[c] = f[c].where(f[c] > AWS_MISSING)          # -99 결측 → 빈값 (평균 · 최댓값 전에)
    f["is_rain"] = f["PTY"].isin([1, 2, 4]).astype(int)
    f["is_snow"] = f["PTY"].isin([2, 3]).astype(int)
    f["pty_missing"] = f["PTY"].isna().astype(int)
    g = (f.groupby(keys)
         .agg(n_regions=("region", "nunique"),
              pcp_mean=("PCP", "mean"), pcp_max=("PCP", "max"),
              sno_mean=("SNO", "mean"), sno_max=("SNO", "max"),
              tmp_mean=("TMP", "mean"), tmp_min=("TMP", "min"),
              pty_rain_regions=("is_rain", "sum"), pty_snow_regions=("is_snow", "sum"),
              pcp_regions=("PCP", "count"), pty_missing_regions=("pty_missing", "sum"))
         .reset_index())
    wide = f.pivot_table(index=keys, columns="region", values=["PCP", "SNO", "TMP", "PTY"], aggfunc="first")
    wide.columns = [f"{v.lower()}_{r}" for v, r in wide.columns]
    wide = wide[sorted(wide.columns)].reset_index()
    out = g.merge(wide, on=keys, how="left")
    fc = pd.to_datetime(out["fcst"])
    out.insert(2, "date", fc.dt.strftime("%Y-%m-%d"))
    out.insert(3, "hour", fc.dt.hour)
    return out.sort_values(["fcst", "lead_h"]).reset_index(drop=True)


def aws_seoul(a: pd.DataFrame) -> pd.DataFrame:
    a = a.copy()
    a["TA"] = a["TA"].where(a["TA"] > AWS_MISSING)
    for c in ["WS", "RN_HR1", "RN_DAY", "HM"]:   # 0 이상만 가능한 값: 음수는 모두 결측 (RN_HR1 -1.0 · -10.6 등 3건)
        a[c] = a[c].where(a[c] >= 0)
    g = (a.groupby("datetime")
         .agg(n_stations=("STN", "nunique"),
              ta_mean=("TA", "mean"), ta_min=("TA", "min"),
              rn_hr1_mean=("RN_HR1", "mean"), rn_hr1_max=("RN_HR1", "max"),
              rn_day_max=("RN_DAY", "max"),
              ws_mean=("WS", "mean"), ws_max=("WS", "max"),
              hm_mean=("HM", "mean"))
         .reset_index())
    dt = pd.to_datetime(g["datetime"])
    g.insert(1, "date", dt.dt.strftime("%Y-%m-%d"))
    g.insert(2, "hour", dt.dt.hour)
    return g.sort_values("datetime").reset_index(drop=True)


def main():
    for split in ("train", "eval"):
        d = PROC_DIR / SPLIT_DIR[split]
        f = fcst_seoul(pd.read_csv(d / f"fcst_{SPLIT_DIR[split]}.csv"))
        a = aws_seoul(pd.read_csv(d / f"aws_hourly_{SPLIT_DIR[split]}.csv"))
        f.round(3).to_csv(d / f"fcst_seoul_{SPLIT_DIR[split]}.csv", index=False, encoding="utf-8-sig")
        a.round(3).to_csv(d / f"aws_seoul_{SPLIT_DIR[split]}.csv", index=False, encoding="utf-8-sig")
        print(f"저장: {split} 예보 {len(f):,}행 (권역 4개 아닌 행 {int((f['n_regions'] != 4).sum())}) · "
              f"AWS {len(a):,}시각 (지점 수 {a['n_stations'].min()}~{a['n_stations'].max()})")


if __name__ == "__main__":
    main()
