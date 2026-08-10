"""Validate fixed and automatically selected STS phase segmentation.

This script audits the public Boswell et al. processing code without inventing
human-annotated ground truth. The author's ``realign`` dictionary supplies only
partial reference checkpoints: 59 manually set sitting frames for 41 subjects.

Example
-------
python scripts/validate_phase_segmentation.py \
  --data-root sit2stand-analysis-main \
  --output results/phase_segmentation_validation
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sts_metrics import (  # noqa: E402
    MHIP,
    NECK,
    NOSE,
    RANK,
    angles_results,
    center_ts,
    fill_nan,
    get_segments,
    lowpass,
    select_magnitude,
    speed_stats,
    swap_lr,
    time_results,
)


GRID = np.geomspace(0.02, 2.0, 30)
FEATURES = [
    "time",
    "time_sit2stand",
    "time_stand2sit",
    "trunk_lean_max",
    "right_knee_range_mean",
    "right_hip_range_mean",
]
SPECIAL_CROPS = {
    "k4Zz5q1I": (75, 240),
    "hozGKSGr": (60, 250),
    "8iHK3CGi": (550, 1000),
    "9qluCnOn": (0, 400),
}
SPECIAL_MAGNITUDES = {
    "k4Zz5q1I": 0.1,
    "hozGKSGr": 0.1,
    "8iHK3CGi": 0.1,
    "9qluCnOn": 0.2,
}
SPECIAL_IDS = set(SPECIAL_CROPS) | {"pmYdj2Zc", "zyW3PPtt"}


def _literal_assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            out[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return out


def _prepare_trajectory(
    trajectory: np.ndarray,
    subject_id: str,
    tofix: dict[str, list[int]],
) -> tuple[np.ndarray, dict[str, object]]:
    """Reproduce official processing through first normalization."""
    res = np.asarray(trajectory, dtype=float).copy()
    if res.ndim != 2 or res.shape[1] != 75:
        raise ValueError(f"expected (frames, 75), got {res.shape}")

    if subject_id == "pmYdj2Zc":
        res = res[:-10, :]

    res[:, 1::3] = 50 + np.nanmax(res[:, 1::3]) - res[:, 1::3]
    md = np.nanmedian(
        res[:, MHIP * 3]
        - (res[:, 13 * 3] + res[:, 10 * 3]) / 2
    )
    orientation = "R" if md < 0 else "L"
    if orientation == "L":
        res[:, 0::3] = 1 + np.nanmax(res[:, 0::3]) - res[:, 0::3]
        res = swap_lr(res)

    first, last = 0, res.shape[0]
    if subject_id in SPECIAL_CROPS:
        first, last = SPECIAL_CROPS[subject_id]
    if subject_id == "zyW3PPtt":
        res[res[:, NOSE * 3 + 1] < -1, NOSE * 3 + 1] = np.nan
    if subject_id in tofix:
        first, last = map(int, tofix[subject_id])

    res = res[first:last, :]
    if res.shape[0] < 20:
        raise ValueError(f"too few frames after crop: {res.shape[0]}")
    res[res < 0.5] = np.nan
    res = np.apply_along_axis(fill_nan, 0, res)
    res = center_ts(res)
    return res, {
        "orientation": orientation,
        "first_frame": int(first),
        "last_frame": int(last),
        "crop_applied": bool(first != 0 or last != trajectory.shape[0]),
    }


def _candidate_diagnostics(
    res: np.ndarray,
    framerate: float,
    expected_reps: int = 5,
) -> dict[str, object]:
    hits: list[float] = []
    for magnitude in GRID:
        ups, downs = get_segments(res, framerate=framerate, magnitude=magnitude)
        if len(ups) == expected_reps and len(downs) in (
            expected_reps,
            expected_reps + 1,
        ):
            hits.append(float(magnitude))

    runs: list[list[float]] = []
    if hits:
        current = [hits[0]]
        for left, right in zip(hits, hits[1:]):
            if right / left < 1.35:
                current.append(right)
            else:
                runs.append(current)
                current = [right]
        runs.append(current)

    selected, n_hits = select_magnitude(
        res, framerate, expected_reps=expected_reps, grid=GRID
    )
    if selected is None:
        status = "no_candidate"
    elif n_hits == 1:
        status = "fragile_single_candidate"
    elif len(runs) > 1:
        status = "ambiguous_multiple_runs"
    else:
        status = "single_run"
    return {
        "selected": selected,
        "n_hits": int(n_hits),
        "n_runs": len(runs),
        "status": status,
        "hit_min": min(hits) if hits else np.nan,
        "hit_max": max(hits) if hits else np.nan,
    }


def _apply_realign(
    downs: np.ndarray,
    corrections: dict[int, int],
) -> tuple[np.ndarray, bool]:
    corrected = np.asarray(downs, dtype=int).copy()
    complete = True
    for index, frame in corrections.items():
        if int(index) >= len(corrected):
            complete = False
            continue
        corrected[int(index)] = int(frame)
    return np.sort(corrected), complete


def _metrics_from_boundaries(
    prepared: np.ndarray,
    ups: np.ndarray,
    downs: np.ndarray,
    framerate: float,
) -> dict[str, object]:
    """Continue the official pipeline from segmentation to paper-compatible metrics."""
    if len(downs) < 2 or len(ups) < 4:
        return {"error": "insufficient_boundaries"}

    all_breaks = sorted(np.asarray(ups, int).tolist() + np.asarray(downs, int).tolist())
    if len(all_breaks) > 1 and all_breaks[1] == int(downs[0]):
        all_breaks = all_breaks[1:]
    if not all_breaks or all_breaks[0] != int(downs[0]):
        return {"error": "first_boundary_is_not_sitting"}
    if len(all_breaks) % 2 == 1:
        all_breaks = all_breaks[:-1]
    all_breaks = np.asarray(all_breaks, dtype=int)

    start, end = int(ups[1]), int(ups[-2])
    segment = (
        prepared[start:end, 3 * NOSE:3 * NOSE + 2]
        - prepared[start:end, 3 * RANK:3 * RANK + 2]
    )
    if segment.shape[0] < 2:
        return {"error": "empty_height_segment"}
    height = float(np.quantile(np.sqrt(np.sum(segment**2, axis=1)), 0.95))
    if not np.isfinite(height) or height <= 0:
        return {"error": "invalid_height"}

    res = prepared.copy()
    for column in range(res.shape[1]):
        res[:, column] = lowpass(res[:, column], framerate, zero_lag=False)
    ankle = res[:, 3 * RANK:3 * RANK + 2].copy()
    for joint in range(25):
        sl = slice(3 * joint, 3 * joint + 2)
        res[:, sl] = (res[:, sl] - ankle) / height

    out: dict[str, object] = {}
    out.update(time_results(downs, framerate, alternate=0))
    out.update(time_results(all_breaks, framerate, alternate=1))
    out.update(time_results(all_breaks, framerate, alternate=-1))
    for boundaries, alternate in ((downs, 0), (all_breaks, 1), (all_breaks, -1)):
        out.update(
            angles_results(
                res,
                boundaries,
                framerate,
                alternate=alternate,
                paper_compat=True,
            )
        )
        for joint, name in ((MHIP, "pelvic"), (NECK, "neck")):
            out.update(
                speed_stats(
                    joint,
                    res,
                    boundaries,
                    framerate,
                    name,
                    alternate=alternate,
                )
            )
    return out


def _finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _strategy_fields(
    row: dict[str, object],
    prefix: str,
    ups: np.ndarray,
    downs: np.ndarray,
    metrics: dict[str, object],
    reference: pd.Series,
) -> None:
    row[f"{prefix}_up_count"] = len(ups)
    row[f"{prefix}_down_count"] = len(downs)
    row[f"{prefix}_relaxed_count_ok"] = bool(
        len(ups) == 5 and len(downs) in (5, 6)
    )
    row[f"{prefix}_strict_count_ok"] = bool(len(ups) == 5 and len(downs) == 6)
    row[f"{prefix}_metrics_ok"] = "error" not in metrics
    row[f"{prefix}_metrics_error"] = metrics.get("error", "")
    for feature in FEATURES:
        value = _finite(metrics.get(feature))
        ref_value = _finite(reference.get(feature))
        row[f"{prefix}_{feature}"] = value
        row[f"{prefix}_{feature}_abs_error"] = (
            abs(value - ref_value)
            if np.isfinite(value) and np.isfinite(ref_value)
            else np.nan
        )


def _cohort_summary(df: pd.DataFrame, checkpoint_df: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {"n": int(len(df))}
    for strategy in ("author", "fixed", "auto"):
        result[strategy] = {
            "relaxed_count_success_n": int(df[f"{strategy}_relaxed_count_ok"].sum()),
            "relaxed_count_success_rate": float(df[f"{strategy}_relaxed_count_ok"].mean()),
            "strict_count_success_n": int(df[f"{strategy}_strict_count_ok"].sum()),
            "strict_count_success_rate": float(df[f"{strategy}_strict_count_ok"].mean()),
            "metrics_success_n": int(df[f"{strategy}_metrics_ok"].sum()),
            "mae_vs_dataMovement": {
                feature: float(df[f"{strategy}_{feature}_abs_error"].mean())
                for feature in FEATURES
            },
        }
    result["auto_selection_status"] = {
        str(key): int(value)
        for key, value in df["auto_selection_status"].value_counts().items()
    }
    if not checkpoint_df.empty:
        available = checkpoint_df.dropna(
            subset=["fixed_abs_error_frames", "auto_abs_error_frames"]
        )
        result["realign_checkpoints"] = {
            "n_subjects": int(checkpoint_df["subjectid"].nunique()),
            "n_checkpoints": int(len(checkpoint_df)),
            "fixed_mae_frames": float(checkpoint_df["fixed_abs_error_frames"].mean()),
            "fixed_median_frames": float(checkpoint_df["fixed_abs_error_frames"].median()),
            "fixed_mae_seconds": float(checkpoint_df["fixed_abs_error_seconds"].mean()),
            "fixed_within_3_frames_rate": float(
                (checkpoint_df["fixed_abs_error_frames"] <= 3).mean()
            ),
            "paired_available_n": int(len(available)),
            "auto_mae_frames": float(available["auto_abs_error_frames"].mean()),
            "auto_median_frames": float(available["auto_abs_error_frames"].median()),
            "auto_mae_seconds": float(available["auto_abs_error_seconds"].mean()),
            "auto_within_3_frames_rate": float(
                (available["auto_abs_error_frames"] <= 3).mean()
            ),
            "auto_improved_n": int(
                (available["auto_abs_error_frames"]
                 < available["fixed_abs_error_frames"]).sum()
            ),
            "auto_equal_n": int(
                (available["auto_abs_error_frames"]
                 == available["fixed_abs_error_frames"]).sum()
            ),
            "auto_worsened_n": int(
                (available["auto_abs_error_frames"]
                 > available["fixed_abs_error_frames"]).sum()
            ),
        }
    return result


def validate(data_root: Path, output: Path, limit: int | None = None) -> dict[str, object]:
    started = time.perf_counter()
    edits = _literal_assignments(data_root / "edits.py")
    utils = _literal_assignments(data_root / "utils.py")
    tofix = {str(k): list(v) for k, v in dict(edits["tofix"]).items()}
    tocheck = set(map(str, edits["tocheck"]))
    toremove = set(map(str, edits["toremove"]))
    realign = {
        str(subject): {int(k): int(v) for k, v in corrections.items()}
        for subject, corrections in dict(utils["realign"]).items()
    }

    movement = pd.read_csv(data_root / "stats" / "dataMovement.csv")
    movement["subjectid"] = movement["subjectid"].astype(str)
    movement = movement.drop_duplicates("subjectid").set_index("subjectid")
    clean_ids = set(
        pd.read_csv(data_root / "stats" / "dataClean_text.csv")["subjectid"].astype(str)
    )
    npy_ids = {path.stem for path in (data_root / "videos" / "np").glob("*.npy")}
    subject_ids = sorted(set(movement.index) & npy_ids)
    if limit is not None:
        subject_ids = subject_ids[:limit]

    direct_intervention = set(tofix) | set(realign) | SPECIAL_IDS
    all_code_flags = direct_intervention | tocheck | toremove

    rows: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    for index, subject_id in enumerate(subject_ids, start=1):
        source = movement.loc[subject_id]
        framerate = _finite(source.get("framerate"))
        if not np.isfinite(framerate) or framerate <= 0:
            framerate = 30.0
        row: dict[str, object] = {
            "subjectid": subject_id,
            "in_final_405": subject_id in clean_ids,
            "direct_intervention": subject_id in direct_intervention,
            "code_flagged": subject_id in all_code_flags,
            "tofix": subject_id in tofix,
            "tocheck": subject_id in tocheck,
            "toremove": subject_id in toremove,
            "realign": subject_id in realign,
            "special_case": subject_id in SPECIAL_IDS,
            "framerate": framerate,
            "reference_n_sit2stand": _finite(source.get("n_sit2stand")),
        }
        try:
            trajectory = np.load(
                data_root / "videos" / "np" / f"{subject_id}.npy",
                allow_pickle=True,
            )
            prepared, prep_meta = _prepare_trajectory(trajectory, subject_id, tofix)
            row.update(prep_meta)

            author_magnitude = SPECIAL_MAGNITUDES.get(subject_id, 1.0)
            author_ups, author_downs_raw = get_segments(
                prepared, framerate=framerate, magnitude=author_magnitude
            )
            author_downs, author_realign_complete = _apply_realign(
                author_downs_raw, realign.get(subject_id, {})
            )
            row["author_magnitude"] = author_magnitude
            row["author_realign_complete"] = author_realign_complete
            author_metrics = _metrics_from_boundaries(
                prepared, author_ups, author_downs, framerate
            )
            _strategy_fields(
                row, "author", author_ups, author_downs, author_metrics, source
            )

            fixed_ups, fixed_downs = get_segments(
                prepared, framerate=framerate, magnitude=1.0
            )
            fixed_metrics = _metrics_from_boundaries(
                prepared, fixed_ups, fixed_downs, framerate
            )
            _strategy_fields(row, "fixed", fixed_ups, fixed_downs, fixed_metrics, source)

            auto_diag = _candidate_diagnostics(prepared, framerate)
            row["auto_magnitude"] = auto_diag["selected"]
            row["auto_magnitude_candidates"] = auto_diag["n_hits"]
            row["auto_magnitude_runs"] = auto_diag["n_runs"]
            row["auto_selection_status"] = auto_diag["status"]
            row["auto_hit_min"] = auto_diag["hit_min"]
            row["auto_hit_max"] = auto_diag["hit_max"]
            if auto_diag["selected"] is None:
                auto_ups = np.array([], dtype=int)
                auto_downs = np.array([], dtype=int)
                auto_metrics = {"error": "no_magnitude_candidate"}
            else:
                auto_ups, auto_downs = get_segments(
                    prepared,
                    framerate=framerate,
                    magnitude=float(auto_diag["selected"]),
                )
                auto_metrics = _metrics_from_boundaries(
                    prepared, auto_ups, auto_downs, framerate
                )
            _strategy_fields(row, "auto", auto_ups, auto_downs, auto_metrics, source)

            for boundary_index, reference_frame in realign.get(subject_id, {}).items():
                fixed_frame = (
                    float(fixed_downs[boundary_index])
                    if boundary_index < len(fixed_downs)
                    else np.nan
                )
                auto_frame = (
                    float(auto_downs[boundary_index])
                    if boundary_index < len(auto_downs)
                    else np.nan
                )
                checkpoints.append(
                    {
                        "subjectid": subject_id,
                        "boundary_index": int(boundary_index),
                        "reference_frame": int(reference_frame),
                        "framerate": framerate,
                        "fixed_frame": fixed_frame,
                        "fixed_abs_error_frames": (
                            abs(fixed_frame - reference_frame)
                            if np.isfinite(fixed_frame)
                            else np.nan
                        ),
                        "fixed_abs_error_seconds": (
                            abs(fixed_frame - reference_frame) / framerate
                            if np.isfinite(fixed_frame)
                            else np.nan
                        ),
                        "auto_frame": auto_frame,
                        "auto_abs_error_frames": (
                            abs(auto_frame - reference_frame)
                            if np.isfinite(auto_frame)
                            else np.nan
                        ),
                        "auto_abs_error_seconds": (
                            abs(auto_frame - reference_frame) / framerate
                            if np.isfinite(auto_frame)
                            else np.nan
                        ),
                    }
                )
            row["processing_error"] = ""
        except Exception as error:  # noqa: BLE001
            row["processing_error"] = f"{type(error).__name__}: {error}"
            for strategy in ("author", "fixed", "auto"):
                row[f"{strategy}_relaxed_count_ok"] = False
                row[f"{strategy}_strict_count_ok"] = False
                row[f"{strategy}_metrics_ok"] = False
            row.setdefault("auto_selection_status", "processing_error")
        rows.append(row)
        if index % 50 == 0 or index == len(subject_ids):
            print(f"processed {index}/{len(subject_ids)}", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    subject_df = pd.DataFrame(rows)
    checkpoint_df = pd.DataFrame(checkpoints)
    subject_df.to_csv(output / "subject_results.csv", index=False)
    checkpoint_df.to_csv(output / "realign_checkpoint_errors.csv", index=False)

    complete = subject_df[subject_df["processing_error"].fillna("") == ""].copy()
    final_npy = complete[complete["in_final_405"]].copy()
    auto_subset = final_npy[~final_npy["code_flagged"]].copy()
    summary = {
        "provenance": {
            "data_root_name": data_root.name,
            "official_code_commit": _git_head(data_root),
            "movement_table_n": int(len(movement)),
            "final_cohort_ids_n": int(len(clean_ids)),
            "available_npy_n": int(len(npy_ids)),
            "processed_intersection_n": int(len(subject_ids)),
            "tofix_n": len(tofix),
            "tocheck_n": len(tocheck),
            "toremove_n": len(toremove),
            "realign_subjects_n": len(realign),
            "realign_checkpoints_n": int(sum(len(v) for v in realign.values())),
            "direct_intervention_unique_n": len(direct_intervention),
            "all_code_flags_unique_n": len(all_code_flags),
            "direct_intervention_in_processed_n": int(
                complete["direct_intervention"].sum()
            ),
            "all_code_flags_in_processed_n": int(complete["code_flagged"].sum()),
            "runtime_seconds": time.perf_counter() - started,
        },
        "cohorts": {
            "movement_table_560_with_local_npy": _cohort_summary(
                complete, checkpoint_df
            ),
            "final_cohort_405_with_local_npy": _cohort_summary(
                final_npy,
                checkpoint_df[checkpoint_df["subjectid"].isin(final_npy["subjectid"])]
                if not checkpoint_df.empty
                else checkpoint_df,
            ),
            "final_npy_excluding_code_flags": _cohort_summary(
                auto_subset,
                checkpoint_df[
                    checkpoint_df["subjectid"].isin(auto_subset["subjectid"])
                ]
                if not checkpoint_df.empty
                else checkpoint_df,
            ),
            "direct_intervention_records": _cohort_summary(
                complete[complete["direct_intervention"]],
                checkpoint_df,
            ),
            "realign_records": _cohort_summary(
                complete[complete["realign"]],
                checkpoint_df,
            ),
        },
        "processing_errors": {
            str(key): int(value)
            for key, value in subject_df.loc[
                subject_df["processing_error"].fillna("") != "", "processing_error"
            ].value_counts().items()
        },
        "limits": [
            "realign supplies partial sitting-boundary checkpoints, not complete human-annotated STS boundaries",
            "indexed checkpoints may be ambiguous when a candidate detects a different number or order of sitting events",
            "dataMovement metrics reflect the author pipeline and interventions, not external motion-capture ground truth",
            "matching five repetitions does not establish boundary accuracy",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _git_head(path: Path) -> str | None:
    head = path / ".git" / "HEAD"
    if not head.exists():
        return None
    content = head.read_text(encoding="utf-8").strip()
    if content.startswith("ref: "):
        ref = path / ".git" / content[5:]
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else content
    return content


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase_segmentation_validation"),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    required = [
        args.data_root / "utils.py",
        args.data_root / "edits.py",
        args.data_root / "stats" / "dataMovement.csv",
        args.data_root / "stats" / "dataClean_text.csv",
        args.data_root / "videos" / "np",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("missing required inputs: " + ", ".join(missing))

    summary = validate(args.data_root, args.output, args.limit)
    provenance = summary["provenance"]
    print(
        json.dumps(
            {
                "processed": provenance["processed_intersection_n"],
                "runtime_seconds": round(float(provenance["runtime_seconds"]), 2),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
