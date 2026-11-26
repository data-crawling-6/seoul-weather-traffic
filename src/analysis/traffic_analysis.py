"""[서연] 악천후일-비교일 교통량 변화, 도로 집계, K-means 군집."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

import sys  # noqa: E402

if str(ROOT) not in sys.path:  # python src/analysis/traffic_analysis.py 로 바로 실행해도 src 를 찾게
    sys.path.insert(0, str(ROOT))
from src.io_utils import data_root as _data_root, derived_of, derived_root as _derived_root  # noqa: E402

# 원본 데이터 위치: 저장소 data/raw/ (io_utils.data_root). C 결합표는 data/derived/topis_weather/ (io_utils.derived_root)
DATA_ROOT = _data_root()
TRAFFIC_DIR = _derived_root() / "topis_weather"
TRAVEL_FILE = "commute_20261122_v05.parquet"


def _default_travel_path() -> Path:
    """통행시간 v05: data/derived/topis_weather/ (Release yang-seoyeon-traffic-weather-v3)."""
    return TRAFFIC_DIR / TRAVEL_FILE
REQUIRED = {"date", "matched_date", "role", "split", "season", "road_id", "road_name",
            "spot_id", "io_type", "hour", "commute_period", "traffic_volume",
            "intensity", "precip_mm", "new_snow_cm"}


def latest_input(root: Path = ROOT) -> Path:
    """교통량·악천후일 결합 CSV. outputs/tables 에 없으면 Release(yang-seoyeon-traffic-volume-v1)에서 받은
    data/derived/topis_traffic_volume/ 파일을 쓴다 (PR #23 에서 큰 CSV 를 저장소에서 빼고 Release 로 옮김)."""
    files = sorted((root / "outputs/tables").glob("yang_seoyeon_traffic_weather_days_joined_*_(성공)_v*.csv"))
    if not files:
        files = sorted((root / "data" / "derived" / "topis_traffic_volume").glob("traffic_weather_days_joined_*_v*.csv"))
    if not files:
        raise FileNotFoundError("교통량·악천후일 결합 CSV가 없습니다. Release yang-seoyeon-traffic-volume-v1 의 CSV를 "
                                "data/derived/topis_traffic_volume/ 에 받아 두세요.")
    return files[-1]


def make_pairs(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError(f"필수 열 누락: {sorted(missing)}")
    d = data.copy()
    d["date"] = pd.to_datetime(d["date"], errors="raise").dt.strftime("%Y-%m-%d")
    d["matched_date"] = pd.to_datetime(d["matched_date"], errors="raise").dt.strftime("%Y-%m-%d")
    d["traffic_volume"] = pd.to_numeric(d["traffic_volume"], errors="coerce")
    keys = ["road_id", "spot_id", "io_type", "hour"]
    bad = d.loc[d["role"].eq("악천후일")].copy()
    control = d.loc[d["role"].eq("비교일"), ["date", *keys, "traffic_volume"]].copy()
    control = control.rename(columns={"date": "matched_date", "traffic_volume": "comparison_volume"})
    if bad.duplicated(["date", *keys]).any() or control.duplicated(["matched_date", *keys]).any():
        raise ValueError("날짜·지점·방향·시간 관측 키가 중복됩니다.")
    pairs = bad.merge(
        control,
        on=["matched_date", *keys],
        how="left",
        validate="one_to_one",
        indicator="_comparison_merge",
    )
    structurally_unmatched = pairs["_comparison_merge"].eq("left_only")
    pairs = pairs.rename(columns={"date": "severe_date", "traffic_volume": "severe_volume"})
    valid = pairs["severe_volume"].notna() & pairs["comparison_volume"].notna() & pairs["comparison_volume"].gt(0)
    pairs["volume_delta"] = np.where(valid, pairs["severe_volume"] - pairs["comparison_volume"], np.nan)
    pairs["volume_change_pct"] = np.where(valid, pairs["volume_delta"] / pairs["comparison_volume"] * 100, np.nan)
    pairs["pair_status"] = np.select(
        [pairs["severe_volume"].isna(), pairs["comparison_volume"].isna(), pairs["comparison_volume"].eq(0)],
        ["악천후값_결측", "비교일값_결측", "비교일값_0"], default="유효")
    report = {
        "severe_rows": len(bad),
        "pair_rows": len(pairs),
        "valid_pairs": int(valid.sum()),
        "missing_or_zero_pairs": int((~valid).sum()),
        "structurally_unmatched_comparison_rows": int(structurally_unmatched.sum()),
        "comparison_volume_missing_rows": int(pairs["comparison_volume"].isna().sum()),
    }
    pairs = pairs.drop(columns="_comparison_merge")
    return pairs, report


def aggregate_roads(pairs: pd.DataFrame) -> pd.DataFrame:
    group = ["split", "season", "severe_date", "matched_date", "road_id", "road_name",
             "hour", "commute_period", "intensity", "precip_mm", "new_snow_cm"]
    return (pairs.groupby(group, dropna=False, observed=True)
            .agg(spot_direction_pairs=("pair_status", "size"),
                 valid_pairs=("volume_change_pct", "count"),
                 severe_volume_median=("severe_volume", "median"),
                 comparison_volume_median=("comparison_volume", "median"),
                 volume_delta_median=("volume_delta", "median"),
                 volume_change_pct_median=("volume_change_pct", "median"))
            .reset_index())


def make_features(roads: pd.DataFrame, split: str) -> pd.DataFrame:
    selected = roads.loc[roads["split"].eq(split)].copy()
    selected["is_snow"] = pd.to_numeric(selected["new_snow_cm"], errors="coerce").fillna(0).gt(0)
    selected["is_rain"] = (
        pd.to_numeric(selected["precip_mm"], errors="coerce").fillna(0).gt(0)
        & ~selected["is_snow"]
    )
    rows = []
    for (road_id, road_name), g in selected.groupby(["road_id", "road_name"], observed=True):
        values = g["volume_change_pct_median"]
        rows.append({"split": split, "road_id": road_id, "road_name": road_name,
                     "median_change_pct": values.median(),
                     "change_iqr": values.quantile(.75) - values.quantile(.25),
                     "decrease_share": values.lt(0).mean(),
                     "morning_median": g.loc[g["commute_period"].eq("morning"), "volume_change_pct_median"].median(),
                     "evening_median": g.loc[g["commute_period"].eq("evening"), "volume_change_pct_median"].median(),
                     "rain_median": g.loc[g["is_rain"], "volume_change_pct_median"].median(),
                     "snow_median": g.loc[g["is_snow"], "volume_change_pct_median"].median(),
                     "valid_road_hours": int(values.notna().sum())})
    return pd.DataFrame(rows)


def cluster_roads(
    train_features: pd.DataFrame,
    eval_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    cols = ["median_change_pct", "change_iqr", "decrease_share", "morning_median", "evening_median"]
    complete = train_features.dropna(subset=cols).copy()
    if len(complete) < 4:
        raise ValueError("K-means에 사용할 완전한 도로 특징이 4개 미만입니다.")
    scaler = StandardScaler()
    x = scaler.fit_transform(complete[cols])
    scores = {}
    upper = min(6, len(complete) - 1)
    for k in range(2, upper + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(x)
        scores[k] = float(silhouette_score(x, labels))
    best_k = max(scores, key=scores.get)
    model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    complete["cluster"] = model.fit_predict(x)
    train_result = train_features.merge(
        complete[["road_id", "cluster"]], on="road_id", how="left", validate="one_to_one"
    )

    eval_complete = eval_features.dropna(subset=cols).copy()
    if not eval_complete.empty:
        eval_complete["cluster"] = model.predict(scaler.transform(eval_complete[cols]))
    eval_result = eval_features.merge(
        eval_complete[["road_id", "cluster"]], on="road_id", how="left", validate="one_to_one"
    )

    agreement = train_result[["road_id", "cluster"]].merge(
        eval_result[["road_id", "cluster"]], on="road_id", suffixes=("_train", "_eval"), validate="one_to_one"
    )
    comparable = agreement[["cluster_train", "cluster_eval"]].notna().all(axis=1)
    agreement_rate = float(
        agreement.loc[comparable, "cluster_train"].eq(agreement.loc[comparable, "cluster_eval"]).mean()
    ) if comparable.any() else None

    combined = pd.concat([train_result, eval_result], ignore_index=True)
    cluster_summary = (combined.dropna(subset=["cluster"])
        .groupby(["split", "cluster"], observed=True)
        .agg(road_count=("road_id", "nunique"),
             median_change_pct=("median_change_pct", "median"),
             median_change_iqr=("change_iqr", "median"),
             median_decrease_share=("decrease_share", "median"),
             morning_median=("morning_median", "median"),
             evening_median=("evening_median", "median"),
             rain_median=("rain_median", "median"),
             snow_median=("snow_median", "median"),
             median_valid_road_hours=("valid_road_hours", "median"))
        .reset_index())

    distances = model.transform(x)
    complete["distance_to_centroid"] = distances[np.arange(len(complete)), complete["cluster"].astype(int)]
    representatives = (complete.sort_values(["cluster", "distance_to_centroid"])
        .groupby("cluster", observed=True).head(5)
        [["cluster", "road_id", "road_name", "distance_to_centroid", *cols]])

    report = {
        "feature_columns": cols,
        "best_k": int(best_k),
        "silhouette_scores": {str(k): v for k, v in scores.items()},
        "clustered_roads": len(complete),
        "unclustered_roads": int(train_result["cluster"].isna().sum()),
        "eval_clustered_roads": len(eval_complete),
        "eval_unclustered_roads": int(eval_result["cluster"].isna().sum()),
        "train_eval_comparable_roads": int(comparable.sum()),
        "train_eval_same_cluster_roads": int(
            agreement.loc[comparable, "cluster_train"].eq(agreement.loc[comparable, "cluster_eval"]).sum()
        ),
        "train_eval_cluster_agreement_rate": agreement_rate,
        "fit_scope": "train only",
        "eval_scope": "prediction with train-fitted scaler and centroids",
        "random_state": 42,
    }
    return train_result, eval_result, cluster_summary, representatives, report


def _latest_analysis_report(root: Path) -> tuple[Path, dict]:
    log_dir = root / "logs/ysy/traffic_analysis"
    files = sorted({
        *log_dir.glob("traffic_analysis_*_(성공)_v*.json"),
        *log_dir.glob("yang_seoyeon_traffic_analysis_*_(성공)_v*.json"),
    })
    if not files:
        raise FileNotFoundError("성공한 교통량 분석 JSON 로그가 없습니다.")
    path = files[-1]
    report = json.loads(path.read_text(encoding="utf-8"))
    required_outputs = {"clusters", "eval_clusters", "cluster_summary", "representatives"}
    missing = required_outputs - set(report.get("outputs", {}))
    if missing:
        raise ValueError(f"최신 분석 로그에 시각화 입력 경로가 없습니다: {sorted(missing)}")
    return path, report


def visualize_latest(root: Path = ROOT, data_root: Path = DATA_ROOT) -> dict:
    """[서연] 최신 분석 결과만 읽어 표·그래프·지도를 만든다. 분석은 재실행하지 않는다."""
    try:
        import matplotlib.pyplot as plt
        import geopandas as gpd
        import folium
    except ImportError as exc:
        raise RuntimeError(
            "시각화 패키지가 없습니다. youth-isolation 환경에 matplotlib, geopandas, folium이 필요합니다."
        ) from exc

    # [서연] Windows 제출 환경에서 한글 제목·범례가 네모로 깨지지 않게 한다.
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    log_path, analysis = _latest_analysis_report(root)
    output_paths = {name: Path(path) for name, path in analysis["outputs"].items()}
    train = pd.read_csv(output_paths["clusters"], encoding="utf-8-sig")
    evaluation = pd.read_csv(output_paths["eval_clusters"], encoding="utf-8-sig")
    summary = pd.read_csv(output_paths["cluster_summary"], encoding="utf-8-sig")
    representatives = pd.read_csv(output_paths["representatives"], encoding="utf-8-sig")

    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    suffix = f"{stamp}_(성공)_v01"
    figure_dir = root / "outputs/figures/ysy"
    table_dir = root / "outputs/tables"
    log_dir = root / "logs/ysy/traffic_analysis"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_eval_plot": figure_dir / f"cluster_train_eval_{suffix}.png",
        "commute_plot": figure_dir / f"cluster_commute_{suffix}.png",
        "weather_plot": figure_dir / f"cluster_weather_{suffix}.png",
        "map_png": figure_dir / f"cluster_map_{suffix}.png",
        "map_html": figure_dir / f"cluster_map_{suffix}.html",
        "eval_unclustered": table_dir / f"yang_seoyeon_cluster_eval_unclassified_{suffix}.csv",
        "interpretation": table_dir / f"yang_seoyeon_cluster_interpretation_{suffix}.json",
        "visualization_log": log_dir / f"traffic_visualization_{suffix}.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("시각화 출력 경로가 이미 존재합니다.")

    split_order = [value for value in ["train", "eval"] if value in set(summary["split"])]
    clusters = sorted(summary["cluster"].dropna().astype(int).unique())
    colors = {0: "#d95f02", 1: "#1b9e77", 2: "#7570b3", 3: "#e7298a"}

    def grouped_plot(columns: list[str], labels: list[str], title: str, ylabel: str, path: Path) -> None:
        fig, axes = plt.subplots(1, len(split_order), figsize=(6 * len(split_order), 4), sharey=True)
        if len(split_order) == 1:
            axes = [axes]
        x = np.arange(len(clusters))
        width = 0.8 / len(columns)
        for ax, split in zip(axes, split_order):
            part = summary.loc[summary["split"].eq(split)].set_index("cluster")
            for idx, (column, label) in enumerate(zip(columns, labels)):
                values = [part.at[cluster, column] if cluster in part.index else np.nan for cluster in clusters]
                ax.bar(x + (idx - (len(columns) - 1) / 2) * width, values, width, label=label)
            ax.axhline(0, color="black", linewidth=.8)
            ax.set_xticks(x, [f"군집 {cluster}" for cluster in clusters])
            ax.set_title(split)
            ax.set_ylabel(ylabel)
            ax.legend()
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    grouped_plot(["median_change_pct"], ["전체"], "군집별 train/eval 교통량 변화", "변화율 중앙값(%)", paths["train_eval_plot"])
    grouped_plot(["morning_median", "evening_median"], ["출근", "퇴근"], "군집별 출퇴근 교통량 변화", "변화율 중앙값(%)", paths["commute_plot"])
    grouped_plot(["rain_median", "snow_median"], ["비", "눈"], "군집별 비·눈 교통량 변화", "변화율 중앙값(%)", paths["weather_plot"])

    unclassified = evaluation.loc[evaluation["cluster"].isna()].copy()
    unclassified.to_csv(paths["eval_unclustered"], index=False, encoding="utf-8-sig")

    geo_files = sorted((derived_of(data_root) / "road_lines").glob("road_lines_*_(성공).geojson"))
    mapping_files = sorted(TRAFFIC_DIR.glob("link_hourly_*_v02.parquet"))
    if not geo_files or not mapping_files:
        raise FileNotFoundError("도로 GeoJSON 또는 링크-도로 매핑 Parquet를 찾지 못했습니다.")
    links = gpd.read_file(geo_files[-1])
    mapping = pd.read_parquet(mapping_files[-1], columns=["link_id", "road_id", "road_name"]).drop_duplicates()
    links["link_id"] = links["link_id"].astype(str)
    mapping["link_id"] = mapping["link_id"].astype(str)
    mapped = links.merge(mapping, on="link_id", how="inner", validate="one_to_one")
    mapped = mapped.merge(train[["road_id", "cluster"]], on="road_id", how="inner", validate="many_to_one")
    mapped = gpd.GeoDataFrame(mapped, geometry="geometry", crs=links.crs).to_crs(4326)

    fig, ax = plt.subplots(figsize=(9, 9))
    for cluster, part in mapped.groupby("cluster", observed=True):
        part.plot(ax=ax, color=colors.get(int(cluster), "#666666"), linewidth=1.2, label=f"군집 {int(cluster)}")
    ax.set_axis_off()
    ax.legend()
    ax.set_title("교통량 변화 군집별 분석 도로")
    fig.tight_layout()
    fig.savefig(paths["map_png"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    bounds = mapped.total_bounds
    fmap = folium.Map(location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2], zoom_start=11)
    for cluster, part in mapped.groupby("cluster", observed=True):
        folium.GeoJson(
            part[["road_id", "road_name", "cluster", "geometry"]],
            name=f"군집 {int(cluster)}",
            style_function=lambda _feature, color=colors.get(int(cluster), "#666666"): {
                "color": color, "weight": 3, "opacity": .8
            },
            tooltip=folium.GeoJsonTooltip(fields=["road_id", "road_name", "cluster"]),
        ).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    fmap.save(paths["map_html"])

    cluster_report = analysis["clustering"]
    summary_records = summary.replace({np.nan: None}).to_dict(orient="records")
    interpretation = {
        "status": "성공",
        "source_analysis_log": str(log_path),
        "best_k": cluster_report["best_k"],
        "silhouette_score": cluster_report["silhouette_scores"][str(cluster_report["best_k"])],
        "train_eval_cluster_agreement_rate": cluster_report["train_eval_cluster_agreement_rate"],
        "eval_unclustered_roads": cluster_report["eval_unclustered_roads"],
        "cluster_statistics": summary_records,
        "representative_roads": representatives.replace({np.nan: None}).to_dict(orient="records"),
        "interpretation_notes": [
            "군집 번호는 순위가 아니라 교통량 변화 특성이 유사한 도로 묶음이다.",
            "군집은 train 자료로만 학습했고 eval은 train 기준으로 예측했다.",
            "교통량 지점이 연결된 52개 도로 결과이며 전체 90개 도로를 대표하지 않는다.",
            "결측으로 군집화되지 않은 eval 도로는 별도 표에 보존했다.",
        ],
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["interpretation"].write_text(json.dumps(interpretation, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["visualization_log"].write_text(json.dumps(interpretation, ensure_ascii=False, indent=2), encoding="utf-8")
    return interpretation


def combine_travel_time(
    root: Path = ROOT,
    travel_path: Path = _default_travel_path(),
) -> dict:
    """[서연] 최신 교통량 변화와 v05 통행시간 변화를 도로·시간 단위로 결합한다."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("그래프 생성을 위해 matplotlib이 필요합니다.") from exc
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    analysis_log, analysis = _latest_analysis_report(root)
    road_hours_path = Path(analysis["outputs"]["road_hours"])
    road_hours = pd.read_csv(road_hours_path, encoding="utf-8-sig", dtype={"road_id": "string"})
    required = {"split", "severe_date", "matched_date", "road_id", "road_name", "hour",
                "commute_period", "intensity", "volume_change_pct_median"}
    missing = required - set(road_hours.columns)
    if missing:
        raise ValueError(f"교통량 도로·시간 결과 필수 열 누락: {sorted(missing)}")
    if not travel_path.exists():
        raise FileNotFoundError(f"통행시간 v05가 없습니다: {travel_path}")

    travel_columns = ["date", "hour", "road_id", "direction", "travel_time_seconds",
                      "road_speed_kmh", "link_count", "observed_link_count"]
    travel = pd.read_parquet(travel_path, columns=travel_columns)
    travel["date"] = pd.to_datetime(travel["date"], errors="raise").dt.strftime("%Y-%m-%d")
    road_hours["severe_date"] = pd.to_datetime(road_hours["severe_date"], errors="raise").dt.strftime("%Y-%m-%d")
    road_hours["matched_date"] = pd.to_datetime(road_hours["matched_date"], errors="raise").dt.strftime("%Y-%m-%d")
    travel["road_id"] = travel["road_id"].astype("string")
    if travel.duplicated(["date", "hour", "road_id", "direction"]).any():
        raise ValueError("통행시간 v05의 날짜·시간·도로·방향 키가 중복됩니다.")
    if road_hours.duplicated(["split", "severe_date", "road_id", "hour"]).any():
        raise ValueError("교통량 도로·시간 결과 키가 중복됩니다.")

    severe_travel = travel.rename(columns={
        "date": "severe_date",
        "travel_time_seconds": "severe_travel_time_seconds",
        "road_speed_kmh": "severe_road_speed_kmh",
        "link_count": "severe_link_count",
        "observed_link_count": "severe_observed_link_count",
    })
    comparison_travel = travel.rename(columns={
        "date": "matched_date",
        "travel_time_seconds": "comparison_travel_time_seconds",
        "road_speed_kmh": "comparison_road_speed_kmh",
        "link_count": "comparison_link_count",
        "observed_link_count": "comparison_observed_link_count",
    })
    base = road_hours[["split", "severe_date", "matched_date", "road_id", "road_name", "hour",
                       "commute_period", "intensity", "volume_change_pct_median"]].copy()
    direction_pairs = base.merge(
        severe_travel,
        on=["severe_date", "road_id", "hour"],
        how="left",
        validate="one_to_many",
        indicator="_severe_merge",
    ).merge(
        comparison_travel,
        on=["matched_date", "road_id", "hour", "direction"],
        how="left",
        validate="many_to_one",
        indicator="_comparison_merge",
    )
    valid = (
        direction_pairs["severe_travel_time_seconds"].notna()
        & direction_pairs["comparison_travel_time_seconds"].notna()
        & direction_pairs["comparison_travel_time_seconds"].gt(0)
    )
    direction_pairs["travel_time_change_pct"] = np.where(
        valid,
        (direction_pairs["severe_travel_time_seconds"] / direction_pairs["comparison_travel_time_seconds"] - 1) * 100,
        np.nan,
    )
    direction_pairs["travel_pair_status"] = np.select(
        [direction_pairs["_severe_merge"].eq("left_only"),
         direction_pairs["_comparison_merge"].eq("left_only"),
         direction_pairs["severe_travel_time_seconds"].isna(),
         direction_pairs["comparison_travel_time_seconds"].isna(),
         direction_pairs["comparison_travel_time_seconds"].eq(0)],
        ["악천후일_키미매칭", "비교일_키미매칭", "악천후일_통행시간결측",
         "비교일_통행시간결측", "비교일_통행시간0"],
        default="유효",
    )
    direction_pairs = direction_pairs.drop(columns=["_severe_merge", "_comparison_merge"])

    group_keys = ["split", "severe_date", "matched_date", "road_id", "road_name", "hour",
                  "commute_period", "intensity", "volume_change_pct_median"]
    travel_road_hour = (direction_pairs.groupby(group_keys, dropna=False, observed=True)
        .agg(direction_count=("direction", "nunique"),
             valid_direction_count=("travel_time_change_pct", "count"),
             travel_time_change_pct_median=("travel_time_change_pct", "median"),
             severe_speed_kmh_median=("severe_road_speed_kmh", "median"),
             comparison_speed_kmh_median=("comparison_road_speed_kmh", "median"))
        .reset_index())

    cluster_frames = []
    for split, key in [("train", "clusters"), ("eval", "eval_clusters")]:
        frame = pd.read_csv(Path(analysis["outputs"][key]), encoding="utf-8-sig", usecols=["road_id", "cluster"])
        frame["split"] = split
        cluster_frames.append(frame)
    cluster_map = pd.concat(cluster_frames, ignore_index=True)
    combined = travel_road_hour.merge(cluster_map, on=["split", "road_id"], how="left", validate="many_to_one")

    def correlation_row(frame: pd.DataFrame, scope: str, value: str) -> dict:
        usable = frame[["volume_change_pct_median", "travel_time_change_pct_median"]].dropna()
        if len(usable) < 3:
            pearson = spearman = np.nan
        else:
            pearson = usable.iloc[:, 0].corr(usable.iloc[:, 1], method="pearson")
            spearman = usable.iloc[:, 0].rank().corr(usable.iloc[:, 1].rank(), method="pearson")
        return {"scope": scope, "value": value, "valid_rows": len(usable),
                "pearson_r": pearson, "spearman_rho": spearman,
                "traffic_change_median": usable.iloc[:, 0].median() if len(usable) else np.nan,
                "travel_time_change_median": usable.iloc[:, 1].median() if len(usable) else np.nan}

    correlation_rows = [correlation_row(combined, "overall", "all")]
    for column in ["split", "cluster", "commute_period", "intensity"]:
        for value, group in combined.groupby(column, dropna=False, observed=True):
            correlation_rows.append(correlation_row(group, column, str(value)))
    correlations = pd.DataFrame(correlation_rows)

    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    suffix = f"{stamp}_(성공)_v01"
    table_dir = root / "outputs/tables"
    figure_dir = root / "outputs/figures/ysy"
    log_dir = root / "logs/ysy/traffic_analysis"
    figure_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "direction_pairs": table_dir / f"yang_seoyeon_traffic_travel_direction_pairs_{suffix}.csv",
        "combined": table_dir / f"yang_seoyeon_traffic_travel_road_hour_{suffix}.csv",
        "correlations": table_dir / f"yang_seoyeon_traffic_travel_correlations_{suffix}.csv",
        "invalid_rows": table_dir / f"yang_seoyeon_traffic_travel_invalid_{suffix}.csv",
        "scatter": figure_dir / f"traffic_travel_scatter_{suffix}.png",
        "log": log_dir / f"traffic_travel_join_{suffix}.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("통행시간 결합 출력 경로가 이미 존재합니다.")

    direction_pairs.to_csv(paths["direction_pairs"], index=False, encoding="utf-8-sig")
    combined.to_csv(paths["combined"], index=False, encoding="utf-8-sig")
    correlations.to_csv(paths["correlations"], index=False, encoding="utf-8-sig")
    invalid = direction_pairs.loc[direction_pairs["travel_pair_status"].ne("유효")].copy()
    invalid.to_csv(paths["invalid_rows"], index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, split in zip(axes, ["train", "eval"]):
        part = combined.loc[combined["split"].eq(split)]
        for cluster, group in part.groupby("cluster", dropna=False, observed=True):
            label = "미분류" if pd.isna(cluster) else f"군집 {int(cluster)}"
            ax.scatter(group["volume_change_pct_median"], group["travel_time_change_pct_median"],
                       s=12, alpha=.35, label=label)
        ax.axhline(0, color="black", linewidth=.7)
        ax.axvline(0, color="black", linewidth=.7)
        ax.set_title(split)
        ax.set_xlabel("교통량 변화율 중앙값(%)")
        ax.set_ylabel("통행시간 변화율 중앙값(%)")
        ax.legend()
    fig.suptitle("악천후 교통량 변화와 통행시간 변화")
    fig.tight_layout()
    fig.savefig(paths["scatter"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "성공",
        "source_analysis_log": str(analysis_log),
        "traffic_road_hours": len(road_hours),
        "direction_pairs": len(direction_pairs),
        "combined_road_hours": len(combined),
        "valid_travel_direction_pairs": int(valid.sum()),
        "invalid_travel_direction_pairs": int((~valid).sum()),
        "severe_structural_unmatched": int(direction_pairs["travel_pair_status"].eq("악천후일_키미매칭").sum()),
        "comparison_structural_unmatched": int(direction_pairs["travel_pair_status"].eq("비교일_키미매칭").sum()),
        "combined_valid_rows": int(combined[["volume_change_pct_median", "travel_time_change_pct_median"]].notna().all(axis=1).sum()),
        "direction_rule": "TOPIS 방향별 변화를 계산한 뒤 도로·시간 중앙값으로 집계; 교통량 유입·유출과 임의 대응하지 않음",
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["log"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def validate_relationships(root: Path = ROOT) -> dict:
    """[서연] 기존 결합 결과의 극단값 영향과 하위집단별 상관 안정성을 검증한다."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("그래프 생성을 위해 matplotlib이 필요합니다.") from exc
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    combined_files = sorted((root / "outputs/tables").glob(
        "yang_seoyeon_traffic_travel_road_hour_*_(성공)_v*.csv"
    ))
    if not combined_files:
        raise FileNotFoundError("성공한 교통량·통행시간 결합 CSV가 없습니다.")
    combined_path = combined_files[-1]
    data = pd.read_csv(combined_path, encoding="utf-8-sig", dtype={"road_id": "string"})
    x_col = "volume_change_pct_median"
    y_col = "travel_time_change_pct_median"
    required = {"split", "severe_date", "road_id", "hour", "commute_period", "intensity",
                "cluster", x_col, y_col}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"민감도 분석 필수 열 누락: {sorted(missing)}")

    # [서연] 기존 결합 CSV에 없던 season은 동일 분석 실행의 도로·시간 표에서 검증해 붙인다.
    _, analysis = _latest_analysis_report(root)
    road_hours = pd.read_csv(Path(analysis["outputs"]["road_hours"]), encoding="utf-8-sig",
                             usecols=["split", "severe_date", "road_id", "hour", "season"],
                             dtype={"road_id": "string"}).drop_duplicates()
    data = data.merge(
        road_hours,
        on=["split", "severe_date", "road_id", "hour"],
        how="left",
        validate="one_to_one",
    )
    if data["season"].isna().any():
        raise ValueError("결합 결과 일부에 season을 연결하지 못했습니다.")

    complete = data[[x_col, y_col]].notna().all(axis=1)
    quantiles = {
        x_col: data.loc[complete, x_col].quantile([.01, .99]).to_dict(),
        y_col: data.loc[complete, y_col].quantile([.01, .99]).to_dict(),
    }
    outlier = complete & (
        data[x_col].lt(quantiles[x_col][.01]) | data[x_col].gt(quantiles[x_col][.99])
        | data[y_col].lt(quantiles[y_col][.01]) | data[y_col].gt(quantiles[y_col][.99])
    )
    data["sensitivity_outlier"] = outlier

    def correlation_row(frame: pd.DataFrame, dataset: str, scope: str, value: str) -> dict:
        usable = frame[[x_col, y_col]].dropna()
        if len(usable) < 3:
            pearson = spearman = np.nan
        else:
            pearson = usable[x_col].corr(usable[y_col], method="pearson")
            spearman = usable[x_col].rank().corr(usable[y_col].rank(), method="pearson")
        return {"dataset": dataset, "scope": scope, "value": value, "valid_rows": len(usable),
                "pearson_r": pearson, "spearman_rho": spearman,
                "traffic_change_median": usable[x_col].median() if len(usable) else np.nan,
                "travel_time_change_median": usable[y_col].median() if len(usable) else np.nan}

    rows = []
    datasets = {"full": data, "trimmed_1_99pct": data.loc[~data["sensitivity_outlier"]]}
    for dataset_name, frame in datasets.items():
        rows.append(correlation_row(frame, dataset_name, "overall", "all"))
        for column in ["split", "season", "cluster", "commute_period", "intensity"]:
            for value, group in frame.groupby(column, dropna=False, observed=True):
                rows.append(correlation_row(group, dataset_name, column, str(value)))
    correlations = pd.DataFrame(rows)

    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    suffix = f"{stamp}_(성공)_v01"
    table_dir = root / "outputs/tables"
    figure_dir = root / "outputs/figures/ysy"
    log_dir = root / "logs/ysy/traffic_analysis"
    figure_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "outliers": table_dir / f"yang_seoyeon_traffic_travel_outliers_{suffix}.csv",
        "sensitivity": table_dir / f"yang_seoyeon_traffic_travel_sensitivity_{suffix}.csv",
        "plot": figure_dir / f"traffic_travel_sensitivity_{suffix}.png",
        "log": log_dir / f"traffic_travel_sensitivity_{suffix}.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("민감도 분석 출력 경로가 이미 존재합니다.")
    data.loc[data["sensitivity_outlier"]].to_csv(paths["outliers"], index=False, encoding="utf-8-sig")
    correlations.to_csv(paths["sensitivity"], index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex="row", sharey="row")
    for row_idx, (dataset_name, frame) in enumerate(datasets.items()):
        for col_idx, split in enumerate(["train", "eval"]):
            ax = axes[row_idx, col_idx]
            part = frame.loc[frame["split"].eq(split)]
            for cluster, group in part.groupby("cluster", dropna=False, observed=True):
                label = "미분류" if pd.isna(cluster) else f"군집 {int(cluster)}"
                ax.scatter(group[x_col], group[y_col], s=12, alpha=.35, label=label)
            ax.axhline(0, color="black", linewidth=.7)
            ax.axvline(0, color="black", linewidth=.7)
            ax.set_title(f"{dataset_name} · {split}")
            ax.set_xlabel("교통량 변화율 중앙값(%)")
            ax.set_ylabel("통행시간 변화율 중앙값(%)")
            ax.legend()
    fig.suptitle("교통량–통행시간 관계의 극단값 민감도")
    fig.tight_layout()
    fig.savefig(paths["plot"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    overall = correlations.loc[correlations["scope"].eq("overall")].set_index("dataset")
    def overall_values(dataset: str) -> dict:
        item = overall.loc[dataset]
        return {"valid_rows": int(item["valid_rows"]),
                "pearson_r": float(item["pearson_r"]),
                "spearman_rho": float(item["spearman_rho"])}

    report = {
        "status": "성공",
        "source_combined": str(combined_path),
        "complete_rows": int(complete.sum()),
        "outlier_rows": int(outlier.sum()),
        "outlier_rule": "두 변화율 중 하나라도 전체 유효 자료의 1백분위 미만 또는 99백분위 초과",
        "thresholds": {
            column: {"p01": float(values[.01]), "p99": float(values[.99])}
            for column, values in quantiles.items()
        },
        "overall_full": overall_values("full"),
        "overall_trimmed": overall_values("trimmed_1_99pct"),
        "interpretation_rule": "전체와 민감도 결과가 크게 다르면 극단값 영향을 명시하고 인과관계로 해석하지 않음",
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["log"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run(root: Path = ROOT, input_path: Path | None = None) -> dict:
    source = input_path or latest_input(root)
    data = pd.read_csv(source, encoding="utf-8-sig", dtype={"road_id": "string", "spot_id": "string"})
    pairs, pair_report = make_pairs(data)
    roads = aggregate_roads(pairs)
    train_features = make_features(roads, "train")
    eval_features = make_features(roads, "eval")
    clusters, eval_clusters, cluster_summary, representatives, cluster_report = cluster_roads(
        train_features, eval_features
    )
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    suffix = f"{stamp}_(성공)_v01"
    table_dir = root / "outputs/tables"
    log_dir = root / "logs/ysy/traffic_analysis"
    log_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pairs": table_dir / f"yang_seoyeon_traffic_pairs_{suffix}.csv",
        "road_hours": table_dir / f"yang_seoyeon_road_traffic_change_{suffix}.csv",
        "features": table_dir / f"yang_seoyeon_road_cluster_features_{suffix}.csv",
        "clusters": table_dir / f"yang_seoyeon_road_kmeans_clusters_{suffix}.csv",
        "eval_clusters": table_dir / f"yang_seoyeon_road_kmeans_eval_{suffix}.csv",
        "cluster_summary": table_dir / f"yang_seoyeon_road_cluster_summary_{suffix}.csv",
        "representatives": table_dir / f"yang_seoyeon_road_cluster_representatives_{suffix}.csv",
        "log": log_dir / f"traffic_analysis_{suffix}.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("분석 출력 경로가 이미 존재합니다.")
    pairs.to_csv(paths["pairs"], index=False, encoding="utf-8-sig")
    roads.to_csv(paths["road_hours"], index=False, encoding="utf-8-sig")
    train_features.to_csv(paths["features"], index=False, encoding="utf-8-sig")
    clusters.to_csv(paths["clusters"], index=False, encoding="utf-8-sig")
    eval_clusters.to_csv(paths["eval_clusters"], index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(paths["cluster_summary"], index=False, encoding="utf-8-sig")
    representatives.to_csv(paths["representatives"], index=False, encoding="utf-8-sig")
    report = {"status": "성공", "input": str(source), "rows": {"source": len(data), "pairs": len(pairs),
              "road_hours": len(roads), "train_roads": len(train_features), "eval_roads": len(eval_features)},
              "pairing": pair_report,
              "clustering": cluster_report, "outputs": {k: str(v) for k, v in paths.items()}}
    paths["log"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def combine_news(root: Path = ROOT, data_root: Path = DATA_ROOT,
                 news_dir: Path | None = None, combined_path: Path | None = None,
                 summary_path: Path | None = None) -> dict:
    """[서연] 기존 분석 결과에 뉴스 후보를 연결한다. 수집·학습은 재실행하지 않는다."""
    import matplotlib.pyplot as plt
    import re
    news_dir = news_dir or derived_of(data_root) / "naver_news"
    def latest(folder, pattern):
        files = sorted(folder.glob(pattern))
        if not files:
            raise FileNotFoundError(f"입력 없음: {folder / pattern}")
        return files[-1]
    table_dir = root / "outputs/tables"
    combined_path = combined_path or latest(table_dir, "yang_seoyeon_traffic_travel_road_hour_*_(성공)_v*.csv")
    summary_path = summary_path or latest(table_dir, "yang_seoyeon_road_cluster_summary_*_(성공)_v*.csv")
    candidate_path = latest(news_dir, "news_road_candidates_*_(성공)_v*.csv")
    event_path = candidate_path.with_name(candidate_path.name.replace("news_road_candidates", "news_event_links"))
    def read(path):
        return pd.read_csv(path, encoding="utf-8-sig", dtype={"road_id": "string"})
    traffic, summary, news, events = map(read, [combined_path, summary_path, candidate_path, event_path])
    keys = ["split", "event_date", "road_id"]
    x, y = "volume_change_pct_median", "travel_time_change_pct_median"
    traffic = traffic.rename(columns={"severe_date": "event_date"})
    for frame in [traffic, news, events]:
        frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise").dt.strftime("%Y-%m-%d")
    if traffic.duplicated(keys + ["hour"]).any() or news.duplicated(keys + ["article_id"]).any():
        raise ValueError("교통 또는 뉴스 연결 키 중복")
    if summary.duplicated(["split", "cluster"]).any():
        raise ValueError("군집 요약 키 중복")
    counts = traffic.dropna(subset=["cluster"]).groupby(["split", "cluster"])["road_id"].nunique()
    expected = summary.set_index(["split", "cluster"])["road_count"]
    if not counts.sort_index().equals(expected.sort_index()):
        raise ValueError("교통 결합 파일과 군집 요약의 도로 수 불일치")
    for col in ["road_name", "cluster", "intensity"]:
        if traffic.groupby(keys)[col].nunique(dropna=False).gt(1).any():
            raise ValueError(f"도로·날짜 내 {col} 불일치")
    traffic[[x,y]] = traffic[[x,y]].replace([np.inf,-np.inf], np.nan)
    traffic["both_valid"] = traffic[[x,y]].notna().all(axis=1)
    daily = traffic.groupby(keys, dropna=False).agg(
        traffic_road_name=("road_name", "first"), cluster=("cluster", "first"),
        intensity=("intensity", "first"), observed_hours=("hour", "nunique"),
        valid_volume_hours=(x,"count"), valid_travel_hours=(y,"count"),
        paired_valid_hours=("both_valid","sum"), volume_daily_median=(x,"median"),
        travel_daily_median=(y,"median")).reset_index()
    # 동일 기사·날짜의 발행일 후보를 보존한다. 실제 사건일은 미확인이다.
    meta = events[["split","event_date","article_id","published_date_candidate","date_offset"]]
    if meta.duplicated(["split","event_date","article_id"]).any():
        raise ValueError("뉴스 날짜 연결 키 중복")
    cases = news.merge(meta, on=["split","event_date","article_id"], how="left", validate="many_to_one", indicator="metadata_match")
    cases = cases.merge(daily, on=keys, how="left", validate="many_to_one", indicator="traffic_match")
    norm = lambda v: re.sub(r"\s+", "", str(v))
    name_ok = cases["road_name"].map(norm).eq(cases["traffic_road_name"].map(norm))
    cases["analysis_candidate"] = (cases["case_status"].eq("candidate_review") & cases["road_evidence"].eq("title") & cases["date_status"].eq("in_window") & cases["relevance"].eq("candidate") & cases["traffic_match"].eq("both") & cases["metadata_match"].eq("both") & name_ok & cases["paired_valid_hours"].gt(0))
    cases["review_note"] = "기사 본문·실제 사건일·시간·인과관계 미확인"
    eligible = cases.loc[cases["analysis_candidate"]].copy()
    # 기사 수가 많은 도로·날짜에 가중치를 주지 않는다.
    selected = eligible[keys].drop_duplicates().merge(daily, on=keys, validate="one_to_one")
    cluster_cases = selected.groupby(["split","cluster"], dropna=False).agg(
        candidate_road_days=("road_id","size"), candidate_roads=("road_id","nunique"),
        volume_median=("volume_daily_median","median"), travel_median=("travel_daily_median","median")).reset_index()
    cluster_cases = cluster_cases.merge(summary, on=["split","cluster"], how="outer", validate="one_to_one")
    cluster_cases[["candidate_road_days","candidate_roads"]] = cluster_cases[["candidate_road_days","candidate_roads"]].fillna(0).astype(int)
    representatives = eligible.sort_values(["paired_valid_hours","travel_daily_median","article_id"], ascending=[False,False,True]).drop_duplicates(keys).groupby(["split","cluster"],dropna=False).head(5)
    hourly = traffic.merge(selected[keys], on=keys, how="inner", validate="many_to_one")
    date_support = events.loc[events["relevance"].eq("candidate") & events["date_status"].eq("in_window")].copy()
    date_support["support_scope"] = "날짜별 검색 후보; 서울·사건일·도로 미확인"
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")
    suffix = f"{stamp}_(성공)_v01"
    fig_dir = root / "outputs/figures/ysy"
    log_dir = root / "logs/ysy/traffic_analysis"
    for folder in [table_dir, fig_dir, log_dir]: folder.mkdir(parents=True, exist_ok=True)
    frames = {"news_case_audit":cases,"news_case_road_days":selected,"news_case_hourly":hourly,"news_cluster_cases":cluster_cases,"news_representative_cases":representatives,"news_date_support":date_support}
    paths = {k:table_dir / f"yang_seoyeon_{k}_{suffix}.csv" for k in frames}
    paths.update(plot=fig_dir / f"news_cases_{suffix}.png", log=log_dir / f"traffic_news_join_{suffix}.json")
    if any(p.exists() for p in paths.values()): raise FileExistsError("출력 경로 중복")
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8,5))
    for (split, cluster), group in selected.groupby(["split","cluster"],dropna=False):
        ax.scatter(group["volume_daily_median"],group["travel_daily_median"],label=f"{split} / {cluster}")
    ax.axhline(0,color="gray"); ax.axvline(0,color="gray")
    ax.set(xlabel="교통량 변화율 중앙값(%)",ylabel="통행시간 변화율 중앙값(%)",title="뉴스 연결 후보 도로·날짜별 교통 변화")
    if not selected.empty: ax.legend()
    fig.tight_layout(); fig.savefig(paths["plot"],dpi=180); plt.close(fig)
    for k, frame in frames.items(): frame.to_csv(paths[k],index=False,encoding="utf-8-sig")
    report = {"status":"성공","inputs":{ "combined":str(combined_path),"summary":str(summary_path),"news":str(candidate_path),"events":str(event_path)},"traffic_rows":len(traffic),"news_candidates":len(news),"traffic_matched":int(cases["traffic_match"].eq("both").sum()),"eligible_candidates":len(eligible),"eligible_unique_articles":int(eligible["article_id"].nunique()),"output_rows":{k:len(v) for k,v in frames.items()},"limitations":"제목 기반 검토 후보만 연결; 사건일·시간·원인 미확인. 뉴스는 학습에 사용하지 않음. 도로·날짜별 중앙값은 관측 출퇴근 시간에 한정.","outputs":{k:str(v) for k,v in paths.items()}}
    paths["log"].write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


def analyze_dtw(root: Path = ROOT, combined_path: Path | None = None) -> dict:
    """[서연] 출퇴근 구간별 대표 패턴의 DTW 및 고정 시간 거리 비교."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt
    import platform
    table = root / "outputs/tables"
    if combined_path is None:
        files = sorted(table.glob("yang_seoyeon_traffic_travel_road_hour_*_(성공)_v*.csv"))
        if not files:
            raise FileNotFoundError("교통량·통행시간 결합 CSV 없음")
        combined_path = files[-1]
    d = pd.read_csv(combined_path, encoding="utf-8-sig", dtype={"road_id":"string"})
    cols = ["volume_change_pct_median", "travel_time_change_pct_median"]
    keys = ["split", "road_id", "severe_date", "hour"]
    required = set(keys + cols + ["road_name", "cluster", "commute_period"])
    if required - set(d):
        raise ValueError(f"필수 열 누락: {required - set(d)}")
    if d[keys].isna().any().any() or d.duplicated(keys).any():
        raise ValueError("관측 키 결측 또는 중복")
    d["severe_date"] = pd.to_datetime(d["severe_date"], errors="raise").dt.strftime("%Y-%m-%d")
    if d.duplicated(keys).any():
        raise ValueError("날짜 정규화 후 관측 키 중복")
    hours = [8,9,18,19,20]
    if not d.hour.isin(hours).all() or set(d.split) != {"train","eval"}:
        raise ValueError("예상하지 않은 시간대 또는 split")
    if not d.commute_period.eq(d.hour.map({8:"morning",9:"morning",18:"evening",19:"evening",20:"evening"})).all():
        raise ValueError("출퇴근 시간대 불일치")
    if set(d.loc[d.split.eq("train"),"severe_date"]) & set(d.loc[d.split.eq("eval"),"severe_date"]):
        raise ValueError("train/eval 날짜 중복")
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf,-np.inf],np.nan)
    # 동일 날짜·시간에 두 변수 모두 유효한 관측만 사용한다.
    d.loc[~d[cols].notna().all(axis=1),cols] = np.nan
    meta = d.groupby(["split","road_id"], observed=True).agg(road_name=("road_name","first"), kmeans_cluster=("cluster","first"))
    for c in ["road_name","cluster"]:
        if d.groupby(["split","road_id"])[c].nunique(dropna=False).gt(1).any():
            raise ValueError(f"도로별 {c} 불일치")
    profiles = d.groupby(["split","road_id","hour"])[cols].agg(["median","count"])
    records=[]; arrays={"train":[],"eval":[]}; ids={"train":[],"eval":[]}
    for (split,rid), row in meta.iterrows():
        p=profiles.loc[split,rid].reindex(hours)
        values=p.loc[:,[(c,"median") for c in cols]].to_numpy(float)
        counts=p[(cols[0],"count")].fillna(0).to_numpy(int)
        valid=bool(np.isfinite(values).all() and (counts>=3).all())
        for i,h in enumerate(hours):
            records.append(dict(split=split,road_id=rid,road_name=row.road_name,hour=h,valid_days=int(counts[i]),volume_median=values[i,0],travel_median=values[i,1],eligible=valid))
        if valid:
            ids[split].append(rid); arrays[split].append(values)
    if len(ids["train"])<4:
        raise ValueError("완전한 train 패턴 4개 미만")
    train=np.asarray(arrays["train"]); evaluation=np.asarray(arrays["eval"]).reshape(-1,5,2)
    scaler=StandardScaler().fit(train.reshape(-1,2))
    train=scaler.transform(train.reshape(-1,2)).reshape(-1,5,2)
    evaluation=scaler.transform(evaluation.reshape(-1,2)).reshape(-1,5,2) if len(evaluation) else evaluation
    def distance(a,b,warp):
        total=0.
        for lo,hi in [(0,2),(2,5)]:
            x,y=a[lo:hi],b[lo:hi]; n=len(x)
            cost=np.full((n+1,n+1),np.inf); cost[0,0]=0
            for i in range(1,n+1):
                for j in range(max(1,i-warp),min(n,i+warp)+1):
                    cost[i,j]=np.sum((x[i-1]-y[j-1])**2)+min(cost[i-1,j],cost[i,j-1],cost[i-1,j-1])
            total+=cost[n,n]
        return float(np.sqrt(total))
    def matrix(a,b,w):
        return np.array([[distance(x,y,w) for y in b] for x in a]).reshape(len(a),len(b))
    scores=[]; assignments=meta.reset_index(); matrices={}; labels_by={}; medoids_by={}
    for method,w in [("euclidean",0),("dtw",1)]:
        D=matrix(train,train,w); matrices[method]=D
        candidates=[]
        for k in range(2,min(6,len(train)-1)+1):
            try: model=AgglomerativeClustering(n_clusters=k,metric="precomputed",linkage="average")
            except TypeError: model=AgglomerativeClustering(n_clusters=k,affinity="precomputed",linkage="average")
            labels=model.fit_predict(D); score=float(silhouette_score(D,labels,metric="precomputed"))
            scores.append(dict(method=method,k=k,silhouette=score)); candidates.append((score,k,labels))
        _,k,labels=max(candidates,key=lambda v:(v[0],-v[1])); labels_by[method]=labels
        medoids=[int(ix[np.argmin(D[np.ix_(ix,ix)].sum(axis=1))]) for c in range(k) for ix in [np.flatnonzero(labels==c)]]
        medoids_by[method]=[str(ids["train"][i]) for i in medoids]
        E=matrix(evaluation,train[medoids],w)
        pred=E.argmin(axis=1) if len(evaluation) else []
        mapping={(s,r):int(l) for s,ls in [("train",labels),("eval",pred)] for r,l in zip(ids[s],ls)}
        assignments[method+"_cluster"]=[mapping.get((s,r),np.nan) for s,r in zip(assignments.split,assignments.road_id)]
    upper=np.triu_indices(len(train),1); eu=matrices["euclidean"][upper]; dt=matrices["dtw"][upper]
    if not np.all(dt<=eu+1e-9): raise AssertionError("DTW 거리 검증 실패")
    stamp=datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")+"_(성공)_v01"
    figdir=root/"outputs/figures/ysy"; logdir=root/"logs/ysy/traffic_analysis"
    for folder in [table,figdir,logdir]: folder.mkdir(parents=True,exist_ok=True)
    frames={"dtw_profiles":pd.DataFrame(records),"dtw_clusters":assignments,"dtw_scores":pd.DataFrame(scores)}
    for method,D in matrices.items(): frames["dtw_distance_"+method]=pd.DataFrame(D,index=ids["train"],columns=ids["train"]).rename_axis("road_id").reset_index()
    paths={k:table/f"yang_seoyeon_{k}_{stamp}.csv" for k in frames}
    paths.update(plot=figdir/f"dtw_patterns_{stamp}.png",log=logdir/f"traffic_dtw_{stamp}.json")
    if any(p.exists() for p in paths.values()): raise FileExistsError("출력 경로 중복")
    plt.rcParams["font.family"]="Malgun Gothic" if platform.system()=="Windows" else "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    raw=np.asarray(arrays["train"])
    for c,rid in enumerate(medoids_by["dtw"]):
        a=raw[ids["train"].index(rid)]
        for j in range(2):
            for q,(lo,hi) in enumerate([(0,2),(2,5)]):
                axes[j,q].plot(hours[lo:hi],a[lo:hi,j],marker="o",label=f"Cluster {c}")
    for j in range(2):
        for q,(lo,hi) in enumerate([(0,2),(2,5)]):
            ax=axes[j,q]; ax.set_xticks(hours[lo:hi]); ax.axhline(0,color="gray",lw=.7); ax.set_ylabel(["Volume change (%)","Travel time change (%)"][j]); ax.set_title(["Morning","Evening"][q]); ax.legend()
    fig.tight_layout(); fig.savefig(paths["plot"],dpi=180); plt.close(fig)
    for k,f in frames.items(): f.to_csv(paths[k],index=False,encoding="utf-8-sig")
    agreement={}
    for s,g in assignments.groupby("split"):
        v=g.dropna(subset=["dtw_cluster","kmeans_cluster"])
        agreement[s]={"comparable":len(v),"ari":float(adjusted_rand_score(v.kmeans_cluster,v.dtw_cluster)) if len(v)>1 else None}
    rho=float(spearmanr(eu,dt).statistic) if np.std(eu)>0 and np.std(dt)>0 else None
    report={"status":"성공","input":str(combined_path),"input_rows":len(d),"eligible_roads":{s:len(v) for s,v in ids.items()},"excluded_roads":{s:int((assignments.split.eq(s)&assignments.dtw_cluster.isna()).sum()) for s in ids},"min_paired_days_per_hour":3,"train_scaler_mean":scaler.mean_.tolist(),"train_scaler_scale":scaler.scale_.tolist(),"medoids":medoids_by,"scores":scores,"distance_rank_correlation":rho,"warping_reduced_pairs":int((dt<eu-1e-9).sum()),"road_pairs":len(eu),"dtw_euclidean_ari":float(adjusted_rand_score(labels_by["dtw"],labels_by["euclidean"])),"kmeans_comparison":agreement,"limitations":"대표 시간대 중앙값 패턴; 출근 2점은 DTW와 고정 거리 동일, 퇴근 3점만 최대 1슬롯 정렬 허용. 날짜 연속 시계열이나 실제 지연 시간 추정 아님. train 평균/표준편차 및 군집만 학습, eval 최근접 train medoid 배정. 최소 3일 기준은 탐색적 설정.","outputs":{k:str(p) for k,p in paths.items()}}
    paths["log"].write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return report


def visualize_dtw_map(root: Path = ROOT, data_root: Path = DATA_ROOT) -> dict:
    """[서연] 기존 DTW 군집 결과와 공식 링크 매핑을 지도에 표시한다."""
    import geopandas as gpd
    import folium
    import matplotlib.pyplot as plt
    def latest(folder, pattern):
        files = sorted(folder.glob(pattern))
        if not files:
            raise FileNotFoundError(f'입력 없음: {folder / pattern}')
        return files[-1]
    source = latest(root / 'outputs/tables', 'yang_seoyeon_dtw_clusters_*_(성공)_v*.csv')
    geo = latest(derived_of(data_root) / 'road_lines', 'road_lines_*_(성공).geojson')
    mapping = latest(TRAFFIC_DIR, 'link_hourly_*_v02.parquet')
    d = pd.read_csv(source, dtype={'road_id': 'string'})
    if d.duplicated(['split','road_id']).any():
        raise ValueError('DTW 도로 키 중복')
    train = d.loc[d.split.eq('train'), ['road_id','road_name','kmeans_cluster','dtw_cluster']].rename(columns={'dtw_cluster':'dtw_train'})
    ev = d.loc[d.split.eq('eval'), ['road_id','dtw_cluster']].rename(columns={'dtw_cluster':'dtw_eval'})
    roads = train.merge(ev,on='road_id',how='outer',validate='one_to_one')
    roads['transition'] = np.select([roads.dtw_train.isna() | roads.dtw_eval.isna(), roads.dtw_train.eq(roads.dtw_eval)], ['Unclassified','Unchanged'], default='Changed')
    links = gpd.read_file(geo)
    if links.crs is None: raise ValueError('GeoJSON CRS 누락')
    m = pd.read_parquet(mapping,columns=['link_id','road_id']).drop_duplicates()
    for f in [links,m]: f['link_id'] = f.link_id.astype(str)
    m['road_id'] = m.road_id.astype('string')
    if m.link_id.duplicated().any(): raise ValueError('링크-도로 매핑 중복')
    mapped = links[['link_id','geometry']].merge(m,on='link_id',validate='one_to_one').merge(roads,on='road_id',validate='many_to_one')
    mapped = gpd.GeoDataFrame(mapped,geometry='geometry',crs=links.crs).to_crs(4326)
    mapped = mapped.loc[mapped.geometry.notna() & ~mapped.geometry.is_empty].copy()
    if mapped.empty: raise ValueError('지도에 표시할 도로 없음')
    roads['mapped_links'] = roads.road_id.map(mapped.groupby('road_id').size()).fillna(0).astype(int)
    stamp = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%f') + '_(성공)_v01'
    figdir=root/'outputs/figures/ysy'; table=root/'outputs/tables'; logdir=root/'logs/ysy/traffic_analysis'
    for p in [figdir,table,logdir]: p.mkdir(parents=True,exist_ok=True)
    paths={'html':figdir/f'dtw_map_{stamp}.html','png':figdir/f'dtw_map_{stamp}.png','coverage':table/f'yang_seoyeon_dtw_map_coverage_{stamp}.csv','log':logdir/f'traffic_dtw_map_{stamp}.json'}
    if any(p.exists() for p in paths.values()): raise FileExistsError('출력 경로 중복')
    colors=['#2563eb','#ea580c','#16a34a','#9333ea','#0891b2','#be123c']
    def cluster_label(v): return 'Unclassified' if pd.isna(v) else f'Cluster {int(v)}'
    for c in ['kmeans_cluster','dtw_train','dtw_eval']: mapped[c+'_label']=mapped[c].map(cluster_label)
    def color(v):
        if v.startswith('Cluster '): return colors[int(v.split()[-1])%len(colors)]
        return {'Changed':'#dc2626','Unchanged':'#16a34a','Unclassified':'#9ca3af'}.get(v,'#9ca3af')
    layers=[('kmeans_cluster_label','K-means train'),('dtw_train_label','DTW train'),('dtw_eval_label','DTW eval'),('transition','DTW train/eval transition')]
    bounds=mapped.total_bounds
    fmap=folium.Map(location=[(bounds[1]+bounds[3])/2,(bounds[0]+bounds[2])/2],zoom_start=11)
    fields=['road_id','road_name','kmeans_cluster_label','dtw_train_label','dtw_eval_label','transition']
    fig,axes=plt.subplots(2,2,figsize=(13,12))
    for i,((col,title),ax) in enumerate(zip(layers,axes.flat)):
        layer=folium.FeatureGroup(name=title,show=i==1)
        for label,part in mapped.groupby(col):
            part.plot(ax=ax,color=color(label),linewidth=1.5,label=label)
            folium.GeoJson(part[fields+['geometry']],style_function=lambda f,c=color(label): {'color':c,'weight':4,'opacity':.85},tooltip=folium.GeoJsonTooltip(fields=fields)).add_to(layer)
        layer.add_to(fmap); ax.set_title(title); ax.set_axis_off(); ax.legend(loc='lower left',fontsize=8)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[bounds[1],bounds[0]],[bounds[3],bounds[2]]]); fmap.save(str(paths['html']))
    fig.tight_layout(); fig.savefig(paths['png'],dpi=180); plt.close(fig)
    roads.to_csv(paths['coverage'],index=False,encoding='utf-8-sig')
    report={'status':'성공','inputs':{'clusters':str(source),'geometry':str(geo),'mapping':str(mapping)},'roads':len(roads),'mapped_roads':int(roads.mapped_links.gt(0).sum()),'mapped_links':len(mapped),'unmapped_roads':roads.loc[roads.mapped_links.eq(0),'road_id'].tolist(),'transition_counts':roads.transition.value_counts().to_dict(),'limitations':'군집 번호는 방법 간 직접 대응하지 않음. 지도는 공식 매핑 링크에 한정; 공간 통계 검정 아님.','outputs':{k:str(v) for k,v in paths.items()}}
    paths['log'].write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def extract_road_control(root: Path, data_root: Path, news_dir: Path | None = None) -> dict:
    """[서연] 제목 기반 통제 및 위험 후보 추출. 실제 통제 여부는 수동 검증한다."""
    folder = news_dir or derived_of(data_root) / "naver_news"
    files = sorted(folder.glob("news_road_candidates_*_(성공)_v*.csv"))
    if not files:
        raise FileNotFoundError(folder)
    source = files[-1]
    data = pd.read_csv(source, encoding="utf-8-sig", dtype={"road_id": "string"})
    keys = ["split", "event_date", "road_id", "article_id"]
    required = keys + ["road_name", "title", "url", "date_status", "road_evidence", "case_status"]
    if set(required) - set(data.columns):
        raise ValueError(f"필수 열 누락: {set(required) - set(data.columns)}")
    title = data["title"].fillna("")
    control = title.str.contains(r"통제|통행\s*(?:제한|금지|차단)|도로\s*(?:폐쇄|차단)|차량\s*진입\s*금지", regex=True)
    hazard = title.str.contains(r"침수|결빙|블랙아이스", regex=True)
    out = data.loc[control | hazard].copy()
    out["control_keyword"] = control.loc[out.index]
    out["hazard_keyword"] = hazard.loc[out.index]
    out["candidate_type"] = np.where(out["control_keyword"], "통제 표현", "위험 표현만")
    out["verification_status"] = "미검증"
    out["verified_event_date"] = ""
    out["verified_control_status"] = ""
    out["verification_note"] = ""
    out = out.drop_duplicates(keys)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    tables = root / "outputs/tables"
    logs = root / "logs/ysy/traffic_analysis"
    tables.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    detail = tables / f"yang_seoyeon_road_control_candidates_{stamp}_(성공)_v01.csv"
    summary = tables / f"yang_seoyeon_road_control_summary_{stamp}_(성공)_v01.csv"
    log = logs / f"traffic_road_control_{stamp}_(성공)_v01.json"
    if any(p.exists() for p in (detail, summary, log)):
        raise FileExistsError("통제 도로 출력 경로 중복")
    out.to_csv(detail, index=False, encoding="utf-8-sig")
    direct = out.loc[out["control_keyword"] & out["date_status"].eq("in_window") & out["road_evidence"].eq("title")]
    grouped = direct.groupby(["split", "event_date", "road_id", "road_name"], dropna=False).agg(candidate_articles=("article_id", "nunique")).reset_index()
    grouped["verification_status"] = "미검증"
    grouped.to_csv(summary, index=False, encoding="utf-8-sig")
    log = logs / f"traffic_road_control_{stamp}_(성공)_v01.json"
    result = {"status": "성공", "source": str(source), "input_rows": len(data), "candidate_rows": len(out), "control_keyword_rows": int(out["control_keyword"].sum()), "hazard_only_rows": int((~out["control_keyword"]).sum()), "priority_control_rows": len(direct), "priority_unique_articles": int(direct["article_id"].nunique()), "priority_road_days": len(grouped), "confirmed_controls": 0, "outputs": {"detail": str(detail), "summary": str(summary), "log": str(log)}, "limitations": "제목의 통제 표현과 도로명이 함께 등장한 검토 후보. 해당 도로의 실제 통제, 해제 여부, 사건일 및 원인은 미검증. 위험 표현만 있는 행은 통제 요약에서 제외."}
    log.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="[서연] 교통량 변화·도로 집계·K-means")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--visualize-only", action="store_true", help="최신 성공 분석 결과의 표·그래프·지도만 생성")
    parser.add_argument("--combine-travel-time", action="store_true", help="최신 교통량과 v05 비교일 기준 통행시간 결합")
    parser.add_argument("--validate-relationships", action="store_true", help="결합 결과의 극단값 민감도와 하위집단별 안정성 검증")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--travel-path", type=Path,
                        default=_default_travel_path())
    parser.add_argument("--combine-news", action="store_true")
    parser.add_argument("--extract-road-control", action="store_true")
    parser.add_argument("--news-dir", type=Path)
    parser.add_argument("--combined-path", type=Path)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--analyze-dtw", action="store_true")
    parser.add_argument("--visualize-dtw-map", action="store_true")
    args = parser.parse_args()
    selected_modes = sum([args.extract_road_control, args.visualize_only, args.combine_travel_time, args.validate_relationships, args.combine_news, args.analyze_dtw, args.visualize_dtw_map])
    if selected_modes > 1:
        parser.error("실행 모드 옵션은 한 번에 하나만 사용할 수 있습니다.")
    if args.visualize_dtw_map:
        result = visualize_dtw_map(args.root, args.data_root)
    elif args.analyze_dtw:
        result = analyze_dtw(args.root, args.combined_path)
    elif args.extract_road_control:
        result = extract_road_control(args.root, args.data_root, args.news_dir)
    elif args.combine_news:
        result = combine_news(args.root, args.data_root, args.news_dir, args.combined_path, args.summary_path)
    elif args.visualize_only:
        result = visualize_latest(args.root, args.data_root)
    elif args.combine_travel_time:
        result = combine_travel_time(args.root, args.travel_path)
    elif args.validate_relationships:
        result = validate_relationships(args.root)
    else:
        result = run(args.root, args.input)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
