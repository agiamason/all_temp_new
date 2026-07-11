#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pickle
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


CLUSTERS = (4, 8, 10)
XPS_TYPES = ("E", "P")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append per-cell RUL labels to the cluster 4/8/10 E/P CSV exports."
    )
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=Path("/home/all_temp_new/output"),
        help="Directory containing the previous cluster CSV exports.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("/home/all_temp_new/data/raw"),
        help="Directory containing raw cell pickle files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/all_temp_new/output"),
        help="Directory where the new RUL CSVs and zip are written.",
    )
    parser.add_argument(
        "--repo-output-dir",
        type=Path,
        default=Path("outputs"),
        help="Repo-local directory where generated artifacts are copied for Feishu upload.",
    )
    parser.add_argument(
        "--zip-name",
        default="cluster4_cluster8_cluster10_E_P_COPF_NMC_temperature_RUL_csvs.zip",
        help="Name of the generated zip archive.",
    )
    return parser.parse_args()


def calc_capacity(current: np.ndarray, time: np.ndarray) -> np.ndarray:
    return np.cumsum(np.diff(time, prepend=0) * current) / 3600.0


def max_discharge_capacity(cycle_data: dict) -> float:
    current = np.asarray(cycle_data.get("I_d", []), dtype=float)
    time = np.asarray(cycle_data.get("t_d", []), dtype=float)
    if current.size == 0 or time.size == 0:
        return float("nan")
    if current.size != time.size:
        raise ValueError(f"I_d/t_d length mismatch: {current.size} != {time.size}")
    q_d = calc_capacity(current, time)
    if q_d.size == 0 or np.all(np.isnan(q_d)):
        return float("nan")
    return float(np.nanmax(q_d))


def first_below(values: list[float], threshold: float) -> float:
    for idx, value in enumerate(values, start=1):
        if not math.isnan(value) and value < threshold:
            return float(idx)
    return float("nan")


def compute_rul_labels(raw_dir: Path) -> pd.DataFrame:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw data directory not found: {raw_dir}")

    rows: list[dict[str, float | str]] = []
    for cell_file in sorted(raw_dir.glob("*.pkl")):
        with cell_file.open("rb") as handle:
            cell_data = pickle.load(handle)

        capacities = [max_discharge_capacity(cycle) for cycle in cell_data]
        capacities = capacities[1:]
        relative_seed = [value for value in capacities[:5] if not math.isnan(value)]
        relative_threshold = float("nan")
        if relative_seed:
            relative_threshold = float(np.mean(relative_seed) * 0.8)

        rows.append(
            {
                "cell_id": cell_file.stem,
                "rul_fixed": first_below(capacities, 1.43),
                "rul_relative": first_below(capacities, relative_threshold)
                if not math.isnan(relative_threshold)
                else float("nan"),
            }
        )

    if not rows:
        raise ValueError(f"no raw pickle files found in {raw_dir}")
    return pd.DataFrame(rows)


def format_values(values: pd.Series) -> str:
    cleaned = values.dropna().astype(int).astype(str)
    return ";".join(cleaned.tolist())


def append_rul_to_detail(detail: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = detail.merge(labels, on="cell_id", how="left")
    missing = merged.loc[
        merged["rul_fixed"].isna() | merged["rul_relative"].isna(), "cell_id"
    ].tolist()
    if missing:
        unique_missing = sorted(set(missing))
        raise ValueError(f"missing RUL labels for cells: {unique_missing}")
    return merged


def append_rul_to_summary(summary: pd.DataFrame, detail: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    grouped = detail.groupby(cluster_col, dropna=False)
    additions = pd.DataFrame(
        {
            "rul_fixed_values": grouped["rul_fixed"].apply(format_values),
            "rul_fixed_mean": grouped["rul_fixed"].mean(),
            "rul_fixed_std": grouped["rul_fixed"].std(ddof=1),
            "rul_fixed_min": grouped["rul_fixed"].min(),
            "rul_fixed_max": grouped["rul_fixed"].max(),
            "rul_relative_values": grouped["rul_relative"].apply(format_values),
            "rul_relative_mean": grouped["rul_relative"].mean(),
            "rul_relative_std": grouped["rul_relative"].std(ddof=1),
            "rul_relative_min": grouped["rul_relative"].min(),
            "rul_relative_max": grouped["rul_relative"].max(),
        }
    )

    result = summary.copy()
    for column in additions.columns:
        result[column] = result[cluster_col].map(additions[column])
    return result


def find_cluster_column(frame: pd.DataFrame, cluster_count: int) -> str:
    for candidate in (f"cluster{cluster_count}", "cluster"):
        if candidate in frame.columns:
            return candidate
    raise KeyError(
        f"could not find a cluster column for cluster count {cluster_count}; "
        f"available columns: {list(frame.columns)}"
    )


def write_outputs(args: argparse.Namespace, labels: pd.DataFrame) -> tuple[list[Path], Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.repo_output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths: list[Path] = []
    for cluster_count in CLUSTERS:
        for xps_type in XPS_TYPES:
            detail_src = args.base_output_dir / (
                f"cluster{cluster_count}_cell_copf_temperature_NMC_type_{xps_type}.csv"
            )
            summary_src = args.base_output_dir / (
                f"cluster{cluster_count}_summary_copf_temperature_NMC_type_{xps_type}.csv"
            )
            if not detail_src.is_file():
                raise FileNotFoundError(f"detail CSV not found: {detail_src}")
            if not summary_src.is_file():
                raise FileNotFoundError(f"summary CSV not found: {summary_src}")

            detail = append_rul_to_detail(pd.read_csv(detail_src), labels)
            summary = pd.read_csv(summary_src)
            detail_cluster_col = find_cluster_column(detail, cluster_count)
            summary_cluster_col = find_cluster_column(summary, cluster_count)
            if detail_cluster_col != summary_cluster_col:
                detail = detail.rename(columns={detail_cluster_col: summary_cluster_col})
            summary = append_rul_to_summary(summary, detail, summary_cluster_col)

            detail_out = args.output_dir / (
                f"cluster{cluster_count}_cell_copf_temperature_NMC_RUL_type_{xps_type}.csv"
            )
            summary_out = args.output_dir / (
                f"cluster{cluster_count}_summary_copf_temperature_NMC_RUL_type_{xps_type}.csv"
            )
            detail.to_csv(detail_out, index=False)
            summary.to_csv(summary_out, index=False)
            csv_paths.extend([detail_out, summary_out])

    zip_path = args.output_dir / args.zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for csv_path in sorted(csv_paths):
            archive.write(csv_path, arcname=csv_path.name)

    repo_zip_path = args.repo_output_dir / args.zip_name
    shutil.copy2(zip_path, repo_zip_path)
    for csv_path in csv_paths:
        shutil.copy2(csv_path, args.repo_output_dir / csv_path.name)

    return csv_paths, zip_path, repo_zip_path


def main() -> int:
    args = parse_args()
    labels = compute_rul_labels(args.raw_dir)
    csv_paths, zip_path, repo_zip_path = write_outputs(args, labels)

    print(f"RUL labels computed: {len(labels)} cells")
    print(f"CSV files written: {len(csv_paths)}")
    for path in sorted(csv_paths):
        print(path)
    print(f"Zip written: {zip_path}")
    print(f"Repo zip copy: {repo_zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
