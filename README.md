# 악천후 모드: 눈 · 비 오는 날 서울 도로 지연 예측

> 2026 데이터크롤링 프로젝트 · 사회문제 해결 유형
> 제출: 2026년 12월 16일(수) 오전 11시 · 발표: 같은 날 12:00~15:30 (북608호, 팀당 15분)

## 한 줄 요약

눈 · 비 오는 날 서울 도로는 해마다 반복해서 마비되지만, 기존 교통혼잡예보는 "어느 도로가 몇 시에 얼마나 느려지는지"를 미리 알려 주지 않는다.
과거 악천후 날의 TOPIS 통행속도 · 기상 · 교통량 데이터로 도로별 지연을 예측하고, TOPIS 교통혼잡예보에 **"악천후 모드"**(도로별 예상 지연 안내)를 더하자고 제안한다.

**검증 질문**: 2025년 12월 4일 퇴근대란을 오후 3시에 미리 알릴 수 있었나? (+ 2026년 7월 14~15일 집중호우)

## 팀 · 역할

| 담당 | 맡는 질문 | 동적 크롤링 | 핵심 분석 |
| --- | --- | --- | --- |
| **A · 예측** | 몇 시간 뒤 각 도로가 평소보다 얼마나 늦어지나? | TOPIS 통행속도 (과제 필수) | 선형회귀 → CatBoost 단계별 실험, 시계열 교차검증, SHAP |
| **B · 날씨와 경보** | 어떤 날씨 조건에서 느려지고, 언제 막히기 시작하나? | 기상특보(대설 · 호우) · 관측 | 정체 경보(로지스틱 · 결정트리), 연관성 분석, 생존분석 |
| **C · 교통량과 공간** | 어느 도로가 취약하고, 보도된 정체와 어떻게 겹치나? | TOPIS 지점별 월간 교통량 · 네이버 뉴스(폭설 · 폭우) | 통행시간 표 · 교통량 매핑, K-means → DTW, Folium 지도 |

자세한 기획 · 역할 · 일정은 팀 노션 페이지를 본다.

## 폴더 구조

```
seoul-weather-traffic/
├── README.md
├── requirements.txt        # 필요한 패키지
├── .gitignore
├── api_keys.example.txt    # 키 이름만 적은 견본 (진짜 api_keys.txt는 올리지 않음)
│
├── config/                 # 팀 공통 규칙 · 기준표 (사람이 정하고 코드가 읽는다)
│   ├── periods.yaml        # 연구 기간 · Train/Dev/Test · 출퇴근 시간 · 휴가철 · 날씨 · 정체 기준값
│   ├── roads.csv           # 분석 도로 90개
│   ├── road_aliases.csv    # 뉴스에서 쓰는 도로 별칭
│   ├── topis_gaps.csv      # TOPIS 수집 시스템 공백 41시간
│   └── columns.md          # 공통 열 이름 · 단위 설명서 (사람이 읽는 문서)
│
├── crawling/               # 수집 코드 (담당은 아래 「코드 담당표」)
├── src/                    # 전처리 · 계산 · 분석 코드 (담당은 아래 「코드 담당표」)
│
├── notebooks/              # 분석 (번호 = 순서, 영문자 = 담당) · 작성 방법은 아래 「노트북 작성 방법」
│   ├── 00_kmj_collect.ipynb   # 수집 기록 (담당자별 1개)
│   ├── 00_ljh_collect.ipynb
│   ├── 00_ysy_collect.ipynb
│   ├── 01_kmj_topis_check.ipynb
│   ├── 02_ljh_weather_events.ipynb
│   ├── 03_ysy_travel_time.ipynb
│   ├── 04_ysy_traffic_mapping.ipynb
│   ├── 05_kmj_vacation_peak.ipynb
│   ├── 10_ljh_comparison.ipynb
│   ├── 11_kmj_prediction.ipynb
│   ├── 12_ljh_alert_survival.ipynb
│   ├── 13_ysy_clustering.ipynb
│   └── 20_all_case_1204.ipynb
│
├── data/
│   ├── raw/                # 원본 (깃에 올리지 않음). 팀 공유 드라이브에서 받아 여기에 푼다. 모든 코드는 이 아래만 읽는다
│   │   ├── topis_speed/    # A: TOPIS 통행속도 월별 YYYY-MM.csv 50개 · topis_manifest.csv
│   │   ├── topis_traffic_volume/ # C: TOPIS 교통량 원본 xlsx 26개 (YYYY-MM.xlsx)
│   │   ├── link_coords/    # C: 링크 좌표 원본 (20261123T213044 최종본)
│   │   ├── naver_news/     # C: 뉴스 기사 원문 7개
│   │   ├── asos/           # B: ASOS 서울(108) 월별 csv
│   │   ├── kma_warning/    # B: 특보 목록 · 통보문 본문
│   │   ├── kma_fcst/       # B: 과거 단기예보 격자
│   │   ├── kma_aws/        # B: 구별 AWS 시간자료
│   │   ├── hanriver_level/ # B: 한강 수위 10분 자료 (홍수통제소 HRFCO)
│   │   └── seoul_road_heating/ # B: 자치구별 도로 열선 설치 현황
│   ├── derived/            # 원본으로 만든 결과 (깃에 올리지 않음, 커서 Release · 공유 드라이브로 나눔)
│   │   ├── topis_weather/  # C: TOPIS + 날씨 결합표 (통행시간 v04 · v05 · 링크 기본표 v02)
│   │   ├── topis_traffic_volume/ # C: 교통량 정리 parquet (train · eval) · 교통량 + 악천후일 결합 CSV
│   │   ├── road_lines/     # C: 도로 선 geojson (링크 좌표로 만듦)
│   │   ├── seoul_road_heating/ # B: 도로 열선 원본을 UTF-8로 바꾼 복사본 (원본은 CP949)
│   │   └── naver_news/     # C: 뉴스 분류 결과 (기사 · 검수 · 사건 연결 · 통제 도로 후보)
│   ├── shared/             # 팀 공통 작은 표 (깃에 올림): 날씨 · 평소값 · 정체 정답 · 입력 변수
│   │   ├── train_dev/      # Train + Dev (겨울 2022~23 · 2023~24 · 2024~25 / 여름 2023 · 2024 · 2025). 둘은 model_set 열로 나눈다
│   │   └── holdout/        # Test 시즌 (겨울 2025~26 / 여름 2026, model_set = test) · 발표 직전까지 열지 않기
│   └── sample/             # 작은 샘플 (코드 테스트용)
│
├── outputs/
│   ├── figures/            # 그래프
│   └── tables/             # MAE 비교표, 경보 성능표 등
│
└── logs/                   # 크롤링 실행 로그 (발표 증거용)
    ├── kmj/                # A 실행 기록 (topis_speed · topis_network · topis_manifest) · screenshots/ 화면 캡처
    ├── ljh/                # B 실행 기록 (kma_* · hrfco_waterlevel · baseline · link_baseline · volume_direction) · screenshots/ 화면 캡처
    └── ysy/                # C 개인 실행·오류·검증 기록 (traffic_volume · vertices 수집 기록 포함)
        ├── travel_time/
        ├── traffic_mapping/
        ├── traffic_volume/
        └── naver_news/
```


### 코드 담당표

폴더는 사람이 아니라 하는 일로 나누고, 담당은 이 표에 적는다. 이름순으로 보면 같은 작업끼리 모인다.

**crawling/ (수집)**

| 파일 | 담당 | 수집하는 것 | 저장 위치 |
| --- | --- | --- | --- |
| `topis_speed.py` | A 민주 | TOPIS 통행속도 (Selenium) | `data/raw/topis_speed/` |
| `topis_forecast.py` | A 민주 | TOPIS 혼잡예보, 매일 06시 자동 (GitHub Actions) | `data-collect` 브랜치 |
| `kma_asos.py` | B 정희 | 기상청 ASOS 서울 관측 (공공데이터포털 API) | `data/raw/asos/` |
| `kma_apihub.py` | B 정희 | 과거 단기예보 · 구별 AWS (기상청 API허브) | `data/raw/kma_fcst/`, `kma_aws/` |
| `kma_warning.py` | B 정희 | 기상특보 목록 · 통보문 (Selenium) | `data/raw/kma_warning/` |
| `hrfco_waterlevel.py` | B 정희 | 한강 수위 (한강홍수통제소 API) | `data/raw/hanriver_level/` |
| `traffic_volume.py` | C 서연 | TOPIS 월별 교통량 xlsx (Selenium) | `data/raw/topis_traffic_volume/` |
| `link_coords.py` | C 서연 | 링크 좌표 (서울시 API) · 도로 선 만들기 | `data/raw/link_coords/`, `data/derived/road_lines/` |
| `naver_news.py` | C 서연 | 네이버 뉴스 (Selenium) · 분류 | `data/raw/naver_news/`, `data/derived/naver_news/` |

**src/ (전처리 · 계산 · 분석)**

| 파일 | 담당 | 하는 일 |
| --- | --- | --- |
| `io_utils.py` | 공통 | 데이터 위치, TOPIS 읽기, Train · Dev · Test 붙이기(`add_model_set`), 공통 표 읽기(`load_processed`), 날씨 판정 붙이기 |
| `weather.py` | B 정희 | 시간별 날씨 표, 눈 · 비 사건, 연구 기간, 악천후일 · 비교일 90일 |
| `weather_seoul.py` | B 정희 | 예보 · AWS를 서울 전체 값(평균 · 최대 · 권역)으로 |
| `warnings_seoul.py` | B 정희 | 특보 통보문 → 서울 특보 상태 |
| `road_baseline.py` | B 정희 | 도로 · 방향 평소 통행시간, 정체 기준 후보 비교 |
| `link_baseline.py` | B 정희 | 링크 평소값, 정체 길이 비율 → 정체 정답(`is_congested`) |
| `volume_direction.py` | B 정희 | 교통량 지점에 TOPIS 상행 · 하행 붙이기 |
| `prior_baseline.py` | A 민주 | 예측 입력용 평소값 (그날 이전 28일) |
| `features.py` | A 민주 | 예측 모델 입력 변수 표 |
| `topis_weather_merge.py` | C 서연 | TOPIS + 날씨 결합표 (도로 통행시간 v04 · v05, 링크 v02) |
| `traffic_mapping.py` | C 서연 | 교통량 지점 ↔ TOPIS 도로 · 링크 연결 |
| `traffic_mapping_candidates.py` | C 서연 | 매핑 후보 점검 |
| `traffic_analysis.py` | C 서연 | 교통량 변화 · 군집(K-means · DTW) · 지도 · 뉴스 연결 |

## 실행 순서

0. **수집 기록** (`notebooks/00_X_collect`): 수집이 끝나면 담당자별로 핵심 코드 · 로그 · 화면 · 결과 · 실패 기록을 정리
1. **수집** (`crawling/`): TOPIS 통행속도(A), 기상특보 · ASOS · 예보 · AWS · 한강 수위(B), 뉴스 · 교통량(TOPIS 월별 파일) · 링크 좌표(C)
2. **공통 표 만들기** (`notebooks/01~04`): 악천후 사건 · 연구 기간(B) → 도로 통행시간 표(C) → 교통량 매핑(C)
3. **분석 · 모델** (`notebooks/10~13`): 비교 분석 · 경보 · 생존분석(B), 예측(A), 군집(C)
4. **사례 검증** (`notebooks/20`): 2025-12-04 폭설 · 2026-07-14~15 집중호우 (세 명 함께)

### C · TOPIS 월별 교통량 수집과 전처리

교통량은 `VolInfo` API 대량 호출 대신 TOPIS의 **지점별 일자별 교통량**
화면을 Selenium으로 조작해 수집한다. 연도 선택 후 AJAX로 월 목록이
갱신되는 것을 기다리고 필요한 26개월의 XLSX 다운로드 버튼을 누른다.
단순 다운로드 URL 직접 호출은 사용하지 않는다.

실행 환경은 Python 3.11 `youth-isolation`이며 다음 패키지가 필요하다.

- `selenium>=4.20`: TOPIS 화면 조작과 다운로드
- `pandas>=2.2`, `openpyxl>=3.1`: 월별 XLSX 전처리
- `pyarrow>=26.0.1`: train/eval Parquet 저장

PowerShell에서 프로젝트 루트로 이동한 다음 아래 순서로 **직접 실행**한다.
코드 import만으로는 브라우저·네트워크·파일 저장이 발생하지 않는다.

```powershell
# 0. 필수 패키지 확인: 모두 True여야 한다.
C:\Users\USER\.conda\envs\youth-isolation\python.exe -c "import importlib.util as u; print({n: bool(u.find_spec(n)) for n in ['selenium','pandas','openpyxl','pyarrow']})"

# 현재 환경에서 openpyxl만 없다면 사용자가 직접 아래 명령을 실행한다.
C:\Users\USER\.conda\envs\youth-isolation\python.exe -m pip install "openpyxl>=3.1"

# 1. 정희의 train/eval 90일에서 필요한 26개 연월만 출력한다.
C:\Users\USER\.conda\envs\youth-isolation\python.exe crawling\traffic_volume.py months

# 2. 전체 실행 전 2024-11 한 달만 Selenium·기록·구조 검증을 시험한다.
C:\Users\USER\.conda\envs\youth-isolation\python.exe crawling\traffic_volume.py test --month 2024-11

# 3. 화면에 전용 Chrome을 띄워 26개월 수집·90일 전처리를 한 실행으로 기록한다.
#    실행 중 해당 Chrome 창을 직접 조작하지 않는다.
C:\Users\USER\.conda\envs\youth-isolation\python.exe crawling\traffic_volume.py run

# 4. 브라우저를 실행하지 않고 과거 성공·실패·중단 이력을 확인한다.
C:\Users\USER\.conda\envs\youth-isolation\python.exe crawling\traffic_volume.py history
```

화면 표시가 필요 없는 재실행에만 `run --headless`를 선택할 수 있다.
데이터 위치는 저장소 `data/raw/`이다 (`src/io_utils.py`의 `data_root()`, 11/26부터 D 드라이브 · 환경변수 사용 안 함).
다른 위치에서 시험할 때만 명령마다 `--data-root 경로`를 붙인다.

생성 위치:

```text
data\raw\topis_traffic_volume\                  원본
└── YYYY-MM.xlsx
data\derived\topis_traffic_volume\              정리 결과
├── traffic_volume_train_실행ID_v01_(성공).parquet
└── traffic_volume_eval_실행ID_v01_(성공).parquet
logs\ysy\traffic_volume\                  실행 기록 (깃에 올림)
├── yang_seoyeon_traffic_volume_test_YYYY-MM_실행ID_v01_(성공)\
├── yang_seoyeon_traffic_volume_실행ID_v01_(성공)\
├── yang_seoyeon_traffic_volume_실행ID_v01_(실패)\
└── yang_seoyeon_traffic_volume_실행ID_v01_(중단)\
```

새 교통량 실행 기록은 `logs/ysy/traffic_volume/`에 저장한다. 예전 위치
(`data/raw/traffic_volume/runs/`)에 남은 기록이 있으면 `history` 명령과
05 노트북에서 계속 함께 조회한다.

이미 크기가 0보다 큰 월별 파일이 있으면 덮어쓰지 않고 건너뛴다. 실패한
월은 기본 3회까지만 재시도하고 JSONL 로그에 시각·월·시도 횟수·실패 원인을
남긴다. 실행 폴더에는 Git 커밋 ID·브랜치·미커밋 여부·변경 파일명과 코드
SHA-256도 기록한다. 실행 시작 시 폴더를 `(중단)`으로 만들고 정상 완료 시
`(성공)`, 오류 종료 시 `(실패)`로 바꾸므로 강제 종료 기록도 보존된다.

`notebooks/00_ysy_collect.ipynb`는 핵심 함수 코드, 모든 실행
이력, 오류·재시도 로그, 검증표와 결과 표를 읽어 보여준다.
조회 셀만 실행하면 Selenium은 시작되지 않으며, 실제 수집 셀의
`EXECUTE_ONE_MONTH_TEST = True` 또는 `EXECUTE_COLLECTION = True`를 직접
설정한 경우에만 수집한다. 1개월 시험은 전체 Parquet를 만들지 않고 해당
월의 대상 날짜·5시간대 검증표와 100행 샘플만 실행 폴더에 남긴다.

전처리 결과의 `hour`는 TOPIS 속도와 같은 **끝 시각 기준**이다.
즉 교통량 `7시`(07:00~08:00)는 `hour=8`로 저장한다. 원본 XLSX와 결과
Parquet는 D 드라이브에만 두고 Git에 추가하지 않는다.


## 데이터 기준

| 항목 | 기준 |
| --- | --- |
| 학습 (3년) | 겨울 2022~23 · 2023~24 · 2024~25 / 여름 2023 · 2024 · 2025 |
| 평가 (한 번만) | 겨울 2025~26 (12/4 포함) / 여름 2026 |
| 연구 기간 | 겨울 11/27 ~ 3/10 · 여름 5/9 ~ 9/23 (학습 3년 첫 · 마지막 사건 평균, `config/periods.yaml`) |
| 분석 단위 | 모델 · 군집은 도로 × 방향 (도시고속 11 + 주간선 79 = 90개 × 2방향), 지도는 구간(링크ID) |
| 공통 형식 | 링크ID × 날짜 × 시간 (열 이름은 `config/columns.md`) |
| 평가 시간 | 평일 출근 07~09시 · 퇴근 17~20시 → TOPIS 열 "~08 · ~09시" / "~18 · ~19 · ~20시" |
| 시간 기준 | 끝 시각 기준: TOPIS "~08시" = 07:00~08:00 평균, ASOS 08시 강수량 = 07:00~08:00 누적 → 같은 시각끼리 붙인다. TOPIS "~24시"는 다음 날 00시 |

## 노트북 작성 방법

### 노트북 이름 (번호 = 단계, 이니셜 = 담당)

`번호_이니셜_내용.ipynb` (예: `02_ljh_weather_events.ipynb`). 이니셜은 `ljh` 이정희(B 날씨와 경보) · `ysy` 양서연(C 교통량과 공간) · `kmj` 강민주(A 예측) · `all` 함께. `notebooks/` 바로 아래에 두고 하위 폴더는 만들지 않는다. 이름순으로 정렬하면 작업 순서가 된다.

> 11/24 영문자(A · B · C)에서 이니셜로 바꾸는 중. 아직 영문자인 노트북은 담당자가 바꾼다.

| 번호 | 단계 | 언제 | 노트북 |
| --- | --- | --- | --- |
| 00 | 수집 기록: 어떻게 받았나 | ~ 11/25 | `00_kmj` · `00_ljh_collect` · `00_ysy_collect` |
| 01–09 | 공통 표 만들기: 원본을 다듬어 셋이 같이 쓰는 표로 | ~ 11/25 (동결) | `01_kmj_topis_check` · `02_ljh_weather_events` · `03_ysy_travel_time` · `04_ysy_traffic_mapping` · `05_kmj_vacation_peak` |
| 10–19 | 분석 · 모델 | 11/25 ~ 12/9 | `10_ljh_comparison` · `11_kmj_prediction` · `12_ljh_alert_survival` · `13_ysy_clustering` |
| 20–29 | 사례 검증 (함께) | 12/10 ~ | `20_all_case_1204` |
| 30 | 제출용 안내 (함께) | 12/14 제출 정리 | `30_all_summary`: 문제 → 데이터 → 핵심 결과 → 해결안을 짧게, 자세한 건 다른 노트북 안내. `outputs/`의 그림 · 표만 읽는다 |

- 뒤 번호는 앞 번호의 결과를 읽는다 (10번대는 01–09의 공통 표, 20번대는 10번대 결과).
- 같은 단계 안의 번호는 겹치지 않게 나눠 가진 것이라 순서 차이는 없다.

### .py와 노트북 나누기

| 무엇 | 어디에 | 왜 |
| --- | --- | --- |
| 수집 본체 (오래 도는 것) | `crawling/*.py`, 터미널에서 실행 | 1시간씩 도는 수집이 노트북 커널과 함께 끊기지 않게, 이어받기 · 재시도 옵션을 쓰려고 |
| 여러 명이 쓰는 전처리 · 계산 | `src/*.py` | 한 곳만 고치면 모두에게 반영되게 |
| 수집 기록 · 확인 · 분석 · 모델 · 그래프 | `notebooks/*.ipynb` | 과정과 결과를 한눈에 보고 PPT 근거로 쓰려고 |

- 처음 해 보는 전처리는 노트북에서 먼저 짜도 된다. 다른 사람도 쓰게 되면 `src/`로 옮긴다.
- **Train · Dev · Test는 폴더가 아니라 `model_set` 열로 고른다.** 규칙은 `config/periods.yaml`의 `model_sets` 한 곳. 읽을 때 `src.io_utils.load_processed("weather_hourly")`(Test 빼고) 또는 `add_model_set(df)`로 붙이고, 마지막 평가 때만 `include_test=True`로 연다.
- `data/shared/` 공통 표는 **.py로만 만든다**. 노트북은 읽기만 하고, 노트북에서 만든 그래프 · 표는 `outputs/`에 저장한다.

### 수집 기록 노트북 `00_이니셜_collect.ipynb` (담당자별 1개)

노트북은 .py를 **실행하지 않고 읽기만** 한다 (다시 실행해도 데이터를 다시 받거나 덮어쓰지 않는다). 예시는 `notebooks/00_ljh_collect.ipynb`. 복사해서 이름과 자료만 바꿔 쓴다.

```
맨 위   자료 목록 표 (자료 · 방법 · 코드 파일)
        준비 셀: show_func · log_summary · show_table 함수
자료마다 (1, 2, 3 …)
  ① 출처 · 흐름 · 실행한 명령어 · 어려웠던 점 한두 줄
  ② 핵심 코드: 함수마다 '하는 일 · 동적 페이지라서 한 처리' 설명 표 + show_func로 원문 보여 주기
  ③ 실행 화면: logs/이니셜/screenshots/ 전후 캡처 (Selenium 자료)
  ④ 실행 로그 요약: 기간 · 건수 · WARNING/ERROR 수
  ⑤ 결과 표: 행 수 · 기간 · 결측 · 앞 몇 줄
마지막  실패 · 재시도 모아 보기: 로그의 WARNING/ERROR를 종류별로 세기 + 문제 → 원인 → 해결 → 결과 표
        정리: 결과 파일 · 건수
```

- **핵심 코드 설명은 PPT 필수 항목**(공지 3번)이다. ②의 설명 표를 그대로 PPT에 옮긴다.
- 코드는 노트북에 복사하지 않고 `show_func("crawling/xxx.py", "함수명")`으로 원문을 띄운다 (.py를 고쳐도 노트북과 안 어긋남).
- 실제 실행 장면을 남기고 싶으면 짧은 시험 실행만 셀로 넣는다: `!python ../crawling/xxx.py --test`

### 로그 규칙 (실패 · 재시도 기록)

1. 로그는 **이어 쓰기** (`logging` `mode="a"`). 덮어쓰지 않아야 처음 실패부터 해결까지 순서대로 남는다.
2. 재시도는 `WARNING`, 끝내 실패하거나 멈추면 `ERROR`. 예: `2024-11 시도 2/3 실패: 403 ...`
3. 끝까지 실패한 건은 마지막에 "실패 N건"으로 남기고, 다시 실행해서 몇 건이 채워졌는지도 남긴다.
4. 인증키는 `***`로 가린다 (오류 메시지에 요청 주소가 통째로 찍힌다).
5. 고친 뒤에도 실패 로그는 지우지 않는다 (PPT 5번 · 어려웠던 점의 증거).

### 커밋 전

- **Kernel → Restart & Run All** 후 결과까지 저장해서 커밋한다. 출력이 비어 있는 노트북은 기록이 아니다.
- 출력은 짧게: 전체 DataFrame 대신 `head()` · 요약, 긴 로그는 끝부분만.
- 노트북은 자기 것만 고친다 (.ipynb는 merge 충돌을 풀기 어렵다).

## 규칙

- **인증키는 올리지 않는다.** `api_keys.txt`는 `.gitignore`에 들어 있다. 새로 받으면 `api_keys.example.txt`에 키 이름만 추가한다.
- **원본 데이터는 올리지 않는다.** 깃허브는 100MB 넘는 파일을 거부한다. 원본은 팀 드라이브, 공통 표는 `data/shared/`에 CSV로 저장한다 (50MB가 넘으면 parquet).
- **노트북은 자기 것만 고친다.** 같이 쓰는 함수는 `src/`에 `.py`로 만든다.
- **`config/`는 팀의 약속이다.** 바꾸면 모두에게 알린다.
- **크롤링할 때마다 `logs/`에 기록한다.** 수집 시각, 받은 파일 수, 실패 · 재시도를 남긴다 (PPT 근거).

## 브랜치

- `main`: 합친 결과
- `a-prediction` · `b-weather` · `c-traffic`: 각자 작업 → PR로 `main`에 합친다 (머지는 **Create a merge commit**)
- 새 작업을 시작하기 전에 `main`을 자기 브랜치로 먼저 가져온다
- 합치는 날: 11/20 · 11/21 · 11/24 · 11/25 (데이터 동결) · 12/9

## 핵심 날짜

| 날짜 | 내용 |
| --- | --- |
| 11/20 | 분석 도로 목록 · 공통 열 이름 (C) |
| 11/21 | 연구 기간 · 강도 기준 · 악천후일 목록 (B), TOPIS 시험 수집 (A) |
| 11/24 | 도로 통행시간 표 (C) |
| 11/25 | 평소값 · 정체 기준 (B), 교통량 표 (C), **데이터 동결** |
| 12/9 | 예측 결과 · SHAP (A), 발동 기준 (B) |
| 12/16 | 오전 11시 제출 (B), 12시 발표 |
