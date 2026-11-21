# 공통 열 이름과 단위

양서연이 2026-11-22 TOPIS·날씨 결합 결과와 팀 계획을 기준으로 정한 초안이다. 팀 데이터의 공통 분석 단위는 **도로 × 방향 × 날짜 × 시간**이며, 지도용 상세 자료는 `link_id` 단위를 유지한다.

## 공통 규칙

- 열 이름은 영문 소문자 `snake_case`를 사용한다.
- 날짜는 `YYYY-MM-DD`, 시각은 24시간제 정수로 기록한다.
- 시간 열은 **구간 종료시각**이다. 예: `hour=8`은 07:00~08:00 구간이다.
- 시간대는 `Asia/Seoul`이다.
- 거리는 m, 속도는 km/h, 통행시간은 초, 강수는 mm, 적설은 cm를 사용한다.
- 알 수 없거나 관측되지 않은 값은 빈 문자열이나 0이 아니라 결측값으로 둔다.
- TOPIS의 `NULL`은 pandas에서 `NaN`으로 읽는다.
- 결합 기본키는 `date + hour + road_name + direction`이다.
- 지도용 링크 자료의 기본키는 `date + hour + link_id`이다.

## 도로·시간 식별 열

| 열 이름 | 자료형 | 단위/허용값 | 의미 |
|---|---|---|---|
| `road_id` | string | `R001`~`R090` | `config/roads.csv`에서 부여한 분석 도로 ID |
| `road_name` | string | TOPIS 도로명 | 분석 도로명 |
| `function_type` | string | `도시고속도로`, `주간선도로` | TOPIS 기능유형 |
| `direction` | string | TOPIS 원자료 값 | 도로 진행 방향. 임의로 상·하행을 바꾸지 않음 |
| `link_id` | string | 숫자형 문자열 | TOPIS 표준 링크 식별자. 앞자리 보존을 위해 문자열 사용 |
| `date` | date | `YYYY-MM-DD` | 관측 날짜 |
| `hour` | integer | `0`~`23` | 구간 종료시각 |
| `commute_period` | category | `morning`, `evening` | 출근(`8,9`) 또는 퇴근(`18,19,20`) |
| `season` | string | `YYYY_summer`, `YYYY-YY_winter` | 연구 계절 |
| `split` | category | `train`, `eval` | 학습·평가 구분 |

## TOPIS 통행 열

| 열 이름 | 자료형 | 단위 | 의미 |
|---|---|---|---|
| `link_count` | integer | 개 | 도로·방향을 구성하는 고유 링크 수 |
| `observed_link_count` | integer | 개 | 유효 속도가 있는 링크 수 |
| `missing_speed_links` | integer | 개 | 속도가 결측이거나 0 이하인 링크 수 |
| `total_distance_m` | float | m | 도로·방향 전체 링크 거리 합계 |
| `observed_distance_m` | float | m | 유효 속도가 있는 링크 거리 합계 |
| `speed_kmh` | float | km/h | 링크 단위 TOPIS 평균속도 |
| `road_speed_kmh` | float | km/h | `observed_distance_m / travel_time_seconds × 3.6`으로 계산한 도로 평균속도 |
| `link_travel_time_seconds` | float | 초 | 링크 거리÷링크 속도로 계산한 통행시간 |
| `travel_time_seconds` | float | 초 | 도로·방향 내 링크 통행시간 합계 |
| `source_month` | string | `YYYY-MM` | 원본 TOPIS 월 파일 |

통행시간은 `distance_m / (speed_kmh × 1000 / 3600)`으로 계산한다. 링크 속도가 결측이거나 0 이하이면 해당 링크 통행시간을 임의 계산하지 않는다. 모든 링크 속도가 결측이면 도로 통행시간도 `NaN`으로 둔다.

## 날씨·경보 열

| 열 이름 | 자료형 | 단위/허용값 | 의미 |
|---|---|---|---|
| `temp_c` | float | °C | 기온 |
| `precip_mm` | float | mm | 시간 강수량 |
| `new_snow_3h_cm` | float | cm | 3시간 신적설 |
| `snow_depth_cm` | float | cm | 적설 깊이 |
| `wind_ms` | float | m/s | 풍속 |
| `humidity` | float | % | 상대습도 |
| `visibility_10m` | float | 10m | 시정 |
| `precip_type` | category | 날씨자료 값 | 강수 유형 |
| `event_id` | string | 날씨자료 값 | 악천후 사건 ID |
| `event_intensity` | category | 날씨자료 값 | 사건 강도 |
| `impact` | category | 날씨자료 값 | 맑음·약한 비눈·악천후 구분 |
| `missing_obs` | boolean | `True`, `False` | 기상 관측 결측 여부 |
| `warn_snow` | boolean | `True`, `False` | 대설특보 여부 |
| `warn_rain` | boolean | `True`, `False` | 호우특보 여부 |
| `warn_cold` | boolean | `True`, `False` | 한파특보 여부 |

## 향후 교통량·공간 열

| 열 이름 | 자료형 | 단위 | 의미 |
|---|---|---|---|
| `traffic_volume` | float | 대/시간 | 도로·방향·시간별 교통량 |
| `traffic_source_id` | string | 제공기관 ID | 교통량 지점 또는 링크 식별자 |
| `longitude` | float | EPSG:4326 | 경도 |
| `latitude` | float | EPSG:4326 | 위도 |
| `geometry` | geometry | EPSG:4326 | 지도용 링크 또는 도로선 |

교통량 API의 실제 응답 열과 단위가 확인되기 전까지 `traffic_volume` 이외의 세부 열을 확정하지 않는다. 원본 열은 보존하고, 공통 열로 변환한 근거를 코드와 로그에 기록한다.

## 분석 도로 기준

- `config/roads.csv`는 2025년 12월 TOPIS의 도시고속도로 11개와 주간선도로 79개, 총 90개를 기준으로 한다.
- `direction_scope=both`는 원자료의 모든 방향을 포함한다는 뜻이며 방향을 합산한다는 뜻이 아니다.
- 기간 전체에는 도로망 변경으로 `경인고속도로`, `자유로`, `여의대로`가 추가로 나타난다. 이 세 도로는 본 90개에서 제외하고 별도 검토표에 보존한다.
- 기준월 이전에 현재 도로명이 없으면 다른 도로명으로 임의 대체하지 않고 결측으로 둔다.
