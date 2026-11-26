"""[서연] Read-only, reproducible spatial candidate diagnostics."""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import pandas as pd


def sha256(path):
    # [서연] Chunked input checksum for provenance.
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def analyze(path, output_dir):
    # [서연] Never mutate source data; make new timestamped output artifacts.
    df = pd.read_csv(path, dtype={'link_id': 'string', 'road_id': 'string'})
    required = {'coordinate_version_id', 'candidate_rank', 'road_name_match', 'road_id', 'road_name', 'link_road_name', 'distance_m', 'mapping_status'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if df.duplicated(['coordinate_version_id', 'candidate_rank']).any():
        raise ValueError('Duplicate coordinate-version candidate ranks')
    top = df.loc[df.candidate_rank.eq(1)].copy()
    if top.coordinate_version_id.duplicated().any():
        raise ValueError('Duplicate rank-1 candidate')
    top['road_name_match'] = top['road_name_match'].astype(str).str.lower().eq('true')
    df['road_name_match'] = df['road_name_match'].astype(str).str.lower().eq('true')
    mismatches = top.loc[~top.road_name_match].copy()
    alt = df.loc[df.candidate_rank.ne(1) & df.road_name_match & df.distance_m.le(50)]
    alt_ids = set(alt.coordinate_version_id)
    mismatch_ids = set(mismatches.coordinate_version_id)
    multi_road = df.groupby('coordinate_version_id').road_id.nunique().gt(1)
    mismatch_pairs = mismatches.groupby(['road_name', 'link_road_name']).size().reset_index(name='coordinate_versions').sort_values(['coordinate_versions','road_name','link_road_name'],ascending=[False,True,True])
    # [서연] A flagged version may have multiple reasons, so do not sum multi-label counts.
    detail = mismatches[['coordinate_version_id','spot_id','io_type','road_name','link_road_name','link_id','road_id','distance_m','mapping_status','review_reasons']].copy()
    detail['alternative_matching_candidate_within_50m'] = detail.coordinate_version_id.isin(alt_ids)
    detail['multiple_candidate_roads'] = detail.coordinate_version_id.map(multi_road).fillna(False)
    now = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%f%z')
    output_dir.mkdir(parents=True,exist_ok=True)
    report_path = output_dir / f'yang_seoyeon_candidate_diagnostics_{now}_(성공).json'
    pair_path = output_dir / f'yang_seoyeon_road_name_mismatch_pairs_{now}_v01.csv'
    detail_path = output_dir / f'yang_seoyeon_road_name_mismatch_review_{now}_v01.csv'
    for p in [report_path,pair_path,detail_path]:
        if p.exists():
            raise FileExistsError(p)
    mismatch_pairs.to_csv(pair_path,index=False,encoding='utf-8-sig',mode='x')
    detail.to_csv(detail_path,index=False,encoding='utf-8-sig',mode='x')
    summary = {
        'run_kind':'fresh_read_only_candidate_diagnostics',
        'generated_at':datetime.now().astimezone().isoformat(),
        'source':{'path':str(path.resolve()),'size_bytes':path.stat().st_size,'sha256':sha256(path)},
        'counts':{
            'candidate_rows':int(len(df)),
            'candidate_coordinate_versions':int(df.coordinate_version_id.nunique()),
            'nearest_road_name_mismatch_versions':int(len(mismatches)),
            'nearest_road_name_mismatch_within_50m':int(mismatches.distance_m.le(50).sum()),
            'nearest_mismatch_with_alternative_matching_candidate_within_50m':int(len(mismatch_ids & alt_ids)),
            'multiple_candidate_roads_versions':int(multi_road.sum()),
        },
        'notes':['No mapping statuses or selected links are changed.', 'Unmapped versions without candidates are not present in the candidate CSV; use the mapping review CSV to reconcile the full 649 versions.'],
        'outputs':{'mismatch_pairs':str(pair_path.resolve()),'mismatch_review':str(detail_path.resolve())},
    }
    with report_path.open('x',encoding='utf-8') as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    analyze(args.candidates,args.output_dir)
