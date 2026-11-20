"""공통 파일 읽기

    from src.io_utils import read_topis, read_topis_gaps

    df = read_topis("2024-12")      # data/raw/topis/csv/topis_speed_2024-12.csv
    gaps = read_topis_gaps()        # config/topis_gaps.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOPIS_CSV_DIR = ROOT / "data" / "raw" / "topis" / "csv"
TOPIS_GAPS = ROOT / "config" / "topis_gaps.csv"

# 시간별 속도 열. "~08시" = 07:00~08:00 평균 (끝 시각 기준)
HOUR_COLS = [f"~{h:02d}시" for h in range(1, 25)]


def _unify_hour_name(col):
    # 2024-02 이전 파일은 "08시", 2024-03부터는 "~08시". 뜻은 같아서 "~08시"로 통일
    if col[:2].isdigit() and col.endswith("시"):
        return "~" + col
    return col


def read_topis(month, **kwargs):
    """TOPIS 월별 CSV 읽기. month는 "2024-12" 같은 연월이나 파일 경로.

    - 결측은 원본에 "NULL" 글자로 들어 있는데 pandas 기본 설정으로 NaN이 된다
    - 시간 열 이름은 "~01시" ~ "~24시"로 통일된다
    - usecols에 시간 열 이름을 넣으면 옛 형식 파일에서 오류가 나므로, 다 읽은 뒤 고른다
    """
    path = Path(month)
    if path.suffix != ".csv":
        path = TOPIS_CSV_DIR / f"topis_speed_{month}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    return df.rename(columns=_unify_hour_name)


def read_topis_gaps():
    """서울 전체 링크가 동시에 비는 시간 목록 (수집 공백 추정).

    datetime은 끝 시각이다. 예: hour=22 -> 21:00~22:00 -> datetime 22:00, hour=24 -> 다음 날 00:00
    """
    return pd.read_csv(TOPIS_GAPS, encoding="utf-8-sig", parse_dates=["datetime"])
