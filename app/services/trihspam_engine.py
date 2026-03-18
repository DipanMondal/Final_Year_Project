from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class TriHSPAMConfig:
    min_I: int = 3
    min_J: int = 2
    min_K: int = 3
    disc_method: str = "eq_size"          # "eq_size" | "eq_width"
    n_bins: int = 5
    mv_method: str | None = None          # None | "locf"
    spm_algo: str = "fournier08closed"    # "fournier08closed" | "clospan" | "prefixspan" | "spam"
    time_relaxed: bool = False            # v1: keep False
    coherence_threshold: float = 0.5
    overlap_filter: float | None = 0.8    # None to disable
    jar_path: str | None = None           # defaults to app/services/spmf_vd.jar
    keep_temp_files: bool = False


# -----------------------------------------------------------------------------
# Public entry points
# -----------------------------------------------------------------------------

def run_weather_trihspam(
    cube_info: dict,
    config: TriHSPAMConfig,
) -> dict:
    """
    Main app-facing function.

    Expected cube_info format from build_trihspam_cube(...):
    {
        "cube": np.ndarray of shape (F, I, K),
        "feature_columns": [...],
        "numeric_features": [...],
        "symbolic_features": [...],
        "numeric_feature_indices": [...],
        "symbolic_feature_indices": [...],
        "windows_meta": [...],
        "window_ids": [...],
        "n_windows": ...,
        "window_size": ...
    }
    """
    _validate_cube_info(cube_info)
    _validate_config(config)

    if config.time_relaxed:
        raise NotImplementedError(
            "time_relaxed=True is not implemented in this first integration. "
            "Use aligned TC-triclusters first."
        )

    cube = np.array(cube_info["cube"], dtype=object, copy=True)
    feature_columns = list(cube_info["feature_columns"])
    numeric_features = list(cube_info["numeric_features"])
    symbolic_features = list(cube_info["symbolic_features"])
    numeric_feature_indices = list(cube_info["numeric_feature_indices"])
    symbolic_feature_indices = list(cube_info["symbolic_feature_indices"])
    windows_meta = list(cube_info.get("windows_meta", []))
    window_ids = list(cube_info.get("window_ids", list(range(cube.shape[1]))))

    if config.mv_method == "locf":
        cube = _impute_missing_with_locf_cube(cube)

    abstractions = _build_abstractions(
        cube=cube,
        numeric_feature_indices=numeric_feature_indices,
        symbolic_feature_indices=symbolic_feature_indices,
        disc_method=config.disc_method,
        n_bins=config.n_bins,
    )

    sequences = _cube_to_sequences(
        cube=cube,
        abstractions=abstractions,
        relaxed=config.time_relaxed,
    )

    item_to_int, reverse_map = _build_item_dictionary(sequences)

    patterns = _mine_patterns_with_spmf(
        sequences=sequences,
        item_to_int=item_to_int,
        reverse_map=reverse_map,
        n_observations=cube.shape[1],
        min_I=config.min_I,
        min_K=config.min_K,
        spm_algo=config.spm_algo,
        jar_path=config.jar_path,
        keep_temp_files=config.keep_temp_files,
    )

    triclusters = []
    for pattern_idx, pattern_row in enumerate(patterns):
        tric = _pattern_to_tricluster(
            pattern_row=pattern_row,
            cube=cube,
            feature_columns=feature_columns,
            numeric_feature_indices=numeric_feature_indices,
            symbolic_feature_indices=symbolic_feature_indices,
            windows_meta=windows_meta,
            window_ids=window_ids,
            min_I=config.min_I,
            min_J=config.min_J,
            min_K=config.min_K,
            coherence_threshold=config.coherence_threshold,
            relaxed=config.time_relaxed,
            tricluster_id=pattern_idx,
        )
        if tric is not None:
            triclusters.append(tric)

    # Remove exact duplicates
    triclusters = _deduplicate_triclusters(triclusters)

    # Optional overlap filtering
    if config.overlap_filter is not None:
        triclusters = _filter_overlapping_triclusters(
            triclusters,
            overlap_threshold=float(config.overlap_filter),
        )

    triclusters.sort(
        key=lambda x: (
            x["hvar3"],
            -x["shape"]["volume"],
            -x["shape"]["rows"],
            -x["shape"]["contexts"],
            -x["shape"]["features"],
        )
    )

    return {
        "config": asdict(config),
        "engine": {
            "aligned_only": True,
            "spm_algorithm": config.spm_algo,
            "n_input_observations": int(cube.shape[1]),
            "n_features": int(cube.shape[0]),
            "window_size": int(cube.shape[2]),
            "n_sequences": int(len(sequences)),
            "n_patterns_mined": int(len(patterns)),
            "n_triclusters": int(len(triclusters)),
        },
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "symbolic_features": symbolic_features,
        "abstractions": abstractions,
        "triclusters": triclusters,
    }


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def _validate_cube_info(cube_info: dict) -> None:
    required = [
        "cube",
        "feature_columns",
        "numeric_features",
        "symbolic_features",
        "numeric_feature_indices",
        "symbolic_feature_indices",
    ]
    missing = [k for k in required if k not in cube_info]
    if missing:
        raise ValueError(f"cube_info is missing keys: {missing}")

    cube = np.asarray(cube_info["cube"], dtype=object)
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D with shape (F, I, K). Got shape={cube.shape}")

    f = cube.shape[0]
    feature_columns = cube_info["feature_columns"]
    if len(feature_columns) != f:
        raise ValueError(
            f"feature_columns length ({len(feature_columns)}) does not match cube feature axis ({f})."
        )


def _validate_config(config: TriHSPAMConfig) -> None:
    if config.min_I <= 0:
        raise ValueError("min_I must be positive.")
    if config.min_J <= 0:
        raise ValueError("min_J must be positive.")
    if config.min_K <= 0:
        raise ValueError("min_K must be positive.")
    if config.n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    if config.disc_method not in {"eq_size", "eq_width"}:
        raise ValueError("disc_method must be 'eq_size' or 'eq_width'.")
    if config.spm_algo not in {"fournier08closed", "clospan", "prefixspan", "spam"}:
        raise ValueError("Unsupported spm_algo.")
    if config.mv_method not in {None, "locf"}:
        raise ValueError("mv_method must be None or 'locf'.")


# -----------------------------------------------------------------------------
# Missing values
# -----------------------------------------------------------------------------

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(np.isnan(value))
    except Exception:
        return False


def _impute_missing_with_locf_cube(cube: np.ndarray) -> np.ndarray:
    """
    LOCF along the time axis, for each feature-observation pair.
    cube shape: (F, I, K)
    """
    out = np.array(cube, dtype=object, copy=True)
    f_count, i_count, k_count = out.shape

    for f_idx in range(f_count):
        for i_idx in range(i_count):
            last_value = None
            for k_idx in range(k_count):
                val = out[f_idx, i_idx, k_idx]
                if _is_missing(val):
                    out[f_idx, i_idx, k_idx] = last_value
                else:
                    last_value = val
    return out


# -----------------------------------------------------------------------------
# Discretization / abstractions
# -----------------------------------------------------------------------------

def _build_abstractions(
    cube: np.ndarray,
    numeric_feature_indices: list[int],
    symbolic_feature_indices: list[int],
    disc_method: str,
    n_bins: int,
) -> dict:
    abstractions: dict[int, dict] = {}

    for f_idx in numeric_feature_indices:
        values = []
        flat = cube[f_idx].reshape(-1)
        for x in flat:
            if not _is_missing(x):
                values.append(float(x))

        values_arr = np.array(values, dtype=float)
        if values_arr.size == 0:
            abstractions[f_idx] = {
                "type": "numeric",
                "labels": ["bin0"],
                "edges": [],
            }
            continue

        if disc_method == "eq_size":
            qs = np.linspace(0.0, 1.0, n_bins + 1)
            edges = np.quantile(values_arr, qs)
        else:
            vmin = float(np.min(values_arr))
            vmax = float(np.max(values_arr))
            if math.isclose(vmin, vmax):
                edges = np.array([vmin, vmax], dtype=float)
            else:
                edges = np.linspace(vmin, vmax, n_bins + 1)

        edges = np.unique(np.array(edges, dtype=float))
        if edges.size < 2:
            edges = np.array([float(values_arr.min()), float(values_arr.max())], dtype=float)

        n_effective_bins = max(1, int(edges.size - 1))
        labels = [f"bin{i}" for i in range(n_effective_bins)]

        abstractions[f_idx] = {
            "type": "numeric",
            "labels": labels,
            "edges": edges.tolist(),
        }

    for f_idx in symbolic_feature_indices:
        vals = []
        flat = cube[f_idx].reshape(-1)
        for x in flat:
            if not _is_missing(x):
                vals.append(str(x))
        unique_vals = sorted(set(vals))
        abstractions[f_idx] = {
            "type": "symbolic",
            "values": unique_vals,
        }

    return abstractions


def _assign_numeric_bin(value: Any, abstraction: dict) -> str | None:
    if _is_missing(value):
        return None

    value = float(value)
    edges = np.array(abstraction.get("edges", []), dtype=float)
    labels = list(abstraction.get("labels", []))

    if edges.size < 2 or len(labels) <= 1:
        return "bin0"

    # searchsorted over inner edges gives 0..n_bins-1
    inner = edges[1:-1]
    idx = int(np.searchsorted(inner, value, side="right"))
    idx = max(0, min(idx, len(labels) - 1))
    return labels[idx]


def _assign_symbolic_value(value: Any) -> str | None:
    if _is_missing(value):
        return None
    return str(value)


# -----------------------------------------------------------------------------
# Sequence conversion
# -----------------------------------------------------------------------------

def _cube_to_sequences(
    cube: np.ndarray,
    abstractions: dict,
    relaxed: bool = False,
) -> dict[str, list[list[str]]]:
    """
    Convert cube (F, I, K) to observation-wise sequences.
    Returns:
        {
            "X0": [["f0_0#bin1", "f1_0#cold"], ["f0_1#bin2"], ...],
            "X1": ...
        }
    """
    f_count, i_count, k_count = cube.shape
    result: dict[str, list[list[str]]] = {}

    for obs_idx in range(i_count):
        itemsets: list[list[str]] = []
        for ctx_idx in range(k_count):
            items: list[str] = []
            for feat_idx in range(f_count):
                abstraction = abstractions[feat_idx]
                raw_val = cube[feat_idx, obs_idx, ctx_idx]

                if abstraction["type"] == "numeric":
                    symbol = _assign_numeric_bin(raw_val, abstraction)
                else:
                    symbol = _assign_symbolic_value(raw_val)

                if symbol is None:
                    continue

                feature_token = f"f{feat_idx}" if relaxed else f"f{feat_idx}_{ctx_idx}"
                items.append(f"{feature_token}#{symbol}")

            items.sort()
            itemsets.append(items)

        result[f"X{obs_idx}"] = itemsets

    return result


def _build_item_dictionary(sequences: dict[str, list[list[str]]]) -> tuple[dict[str, int], dict[int, str]]:
    item_to_int: dict[str, int] = {}
    next_id = 1

    for itemsets in sequences.values():
        for itemset in itemsets:
            for item in itemset:
                if item not in item_to_int:
                    item_to_int[item] = next_id
                    next_id += 1

    reverse_map = {v: k for k, v in item_to_int.items()}
    return item_to_int, reverse_map


def _write_spmf_input(
    sequences: dict[str, list[list[str]]],
    item_to_int: dict[str, int],
    output_path: Path,
    include_timestamps: bool,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for _, itemsets in sequences.items():
            parts: list[str] = []
            time_idx = 0
            for itemset in itemsets:
                if include_timestamps:
                    parts.append(f"<{time_idx}>")
                    time_idx += 1

                if itemset:
                    encoded_items = [str(item_to_int[item]) for item in itemset]
                    parts.extend(encoded_items)
                    parts.append("-1")

            parts.append("-2")
            f.write(" ".join(parts) + "\n")


# -----------------------------------------------------------------------------
# SPMF call + parsing
# -----------------------------------------------------------------------------

def _default_jar_path() -> Path:
    return Path(__file__).resolve().parent / "spmf_vd.jar"


def _mine_patterns_with_spmf(
    sequences: dict[str, list[list[str]]],
    item_to_int: dict[str, int],
    reverse_map: dict[int, str],
    n_observations: int,
    min_I: int,
    min_K: int,
    spm_algo: str,
    jar_path: str | None,
    keep_temp_files: bool,
) -> list[dict]:
    jar = Path(jar_path) if jar_path else _default_jar_path()
    if not jar.exists():
        raise FileNotFoundError(
            f"SPMF jar not found at: {jar}\n"
            "Copy spmf_vd.jar from the original TriHSPAM repo into app/services/ "
            "or pass config.jar_path explicitly."
        )

    algo_map = {
        "clospan": "CloSpan",
        "spam": "SPAM_AGP",
        "prefixspan": "PrefixSpan_AGP",
        "fournier08closed": "Fournier08-Closed+time",
    }
    spmf_name = algo_map[spm_algo]

    # Slightly safer than the original repo: avoid 0% minsup
    percent_min_I = max(1, int(math.ceil((float(min_I) / float(n_observations)) * 100.0)))

    if keep_temp_files:
        temp_dir = Path(tempfile.mkdtemp(prefix="trihspam_"))
        input_path = temp_dir / "spmf_input.txt"
        output_path = temp_dir / "spmf_output.txt"
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="trihspam_")
        temp_dir = Path(temp_dir_obj.name)
        input_path = temp_dir / "spmf_input.txt"
        output_path = temp_dir / "spmf_output.txt"

    include_timestamps = spm_algo == "fournier08closed"
    _write_spmf_input(
        sequences=sequences,
        item_to_int=item_to_int,
        output_path=input_path,
        include_timestamps=include_timestamps,
    )

    cmd = [
        "java",
        "-jar",
        str(jar),
        "run",
        spmf_name,
        str(input_path),
        str(output_path),
        f"{percent_min_I}%",
    ]

    if spm_algo == "fournier08closed":
        # Following the reference repo:
        # min interval = 1, max interval = 1, min whole interval = min_K-1, max whole interval = K-1
        # Since this is aligned TC-triclustering, we keep the same idea.
        cmd.extend(["1", "1", str(max(0, min_K - 1)), str(max(0, _infer_sequence_length(sequences) - 1))])
    else:
        cmd.append("true")

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "SPMF execution failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    patterns = _parse_spmf_output(output_path, reverse_map)

    if not keep_temp_files:
        temp_dir_obj.cleanup()

    return patterns


def _infer_sequence_length(sequences: dict[str, list[list[str]]]) -> int:
    if not sequences:
        return 0
    first_key = next(iter(sequences.keys()))
    return len(sequences[first_key])


def _parse_spmf_output(output_path: Path, reverse_map: dict[int, str]) -> list[dict]:
    """
    Returns a list of dicts:
    [
        {
            "pattern_string": "(f0_0#bin1 f3_0#hot) (f0_1#bin2)",
            "pattern_itemsets": [["f0_0#bin1","f3_0#hot"], ["f0_1#bin2"]],
            "support": 5,
            "subject_ids": [0,1,2,4,8]
        },
        ...
    ]
    """
    if not output_path.exists():
        return []

    rows = []
    with output_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            before_sup, support, subject_ids = _split_spmf_line(line)
            itemsets = _decode_pattern_tokens(before_sup.split(), reverse_map)
            if not itemsets:
                continue

            pattern_string = " ".join(
                "(" + " ".join(itemset) + ")" for itemset in itemsets
            )

            rows.append(
                {
                    "pattern_string": pattern_string,
                    "pattern_itemsets": itemsets,
                    "support": support,
                    "subject_ids": subject_ids,
                }
            )

    return rows


def _split_spmf_line(line: str) -> tuple[str, int, list[int]]:
    before_sup = line
    support = 0
    subject_ids: list[int] = []

    if "#SUP:" in line:
        before_sup, after_sup = line.split("#SUP:", 1)
        after_sup = after_sup.strip()

        if "#SID:" in after_sup:
            support_part, sid_part = after_sup.split("#SID:", 1)
            try:
                support = int(support_part.strip())
            except Exception:
                support = 0

            sid_tokens = sid_part.strip().split()
            for tok in sid_tokens:
                try:
                    subject_ids.append(int(tok))
                except Exception:
                    continue
        else:
            try:
                support = int(after_sup.strip().split()[0])
            except Exception:
                support = 0

    return before_sup.strip(), support, sorted(set(subject_ids))


def _decode_pattern_tokens(tokens: list[str], reverse_map: dict[int, str]) -> list[list[str]]:
    itemsets: list[list[str]] = []
    current: list[str] = []

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue

        # Ignore time annotations like <0>
        if tok.startswith("<") and tok.endswith(">"):
            continue

        if tok == "-1":
            if current:
                itemsets.append(sorted(current))
                current = []
            continue

        if tok == "-2":
            break

        try:
            item_id = int(tok)
        except Exception:
            continue

        if item_id in reverse_map:
            current.append(reverse_map[item_id])

    if current:
        itemsets.append(sorted(current))

    return itemsets


# -----------------------------------------------------------------------------
# Pattern -> tricluster
# -----------------------------------------------------------------------------

def _pattern_to_tricluster(
    pattern_row: dict,
    cube: np.ndarray,
    feature_columns: list[str],
    numeric_feature_indices: list[int],
    symbolic_feature_indices: list[int],
    windows_meta: list[dict],
    window_ids: list[int],
    min_I: int,
    min_J: int,
    min_K: int,
    coherence_threshold: float,
    relaxed: bool,
    tricluster_id: int,
) -> dict | None:
    if relaxed:
        raise NotImplementedError("Relaxed time mode is not implemented in this v1 engine.")

    rows_I = sorted(set(pattern_row["subject_ids"]))
    if len(rows_I) < min_I:
        return None

    cols_J: list[int] = []
    contx_K: list[int] = []

    for itemset in pattern_row["pattern_itemsets"]:
        for item in itemset:
            left = item.split("#", 1)[0]  # e.g. f3_12
            try:
                feat_part, ctx_part = left.split("_", 1)
                feat_idx = int(feat_part.replace("f", ""))
                ctx_idx = int(ctx_part)
            except Exception:
                continue

            if feat_idx not in cols_J:
                cols_J.append(feat_idx)
            if ctx_idx not in contx_K:
                contx_K.append(ctx_idx)

    rows_I.sort()
    cols_J.sort()
    contx_K.sort()

    if len(cols_J) < min_J or len(contx_K) < min_K:
        return None

    subcube_kij = _extract_subcube_kij(
        cube=cube,
        rows_I=rows_I,
        cols_J=cols_J,
        contx_K=contx_K,
    )

    numeric_local = [idx for idx, feat_idx in enumerate(cols_J) if feat_idx in set(numeric_feature_indices)]
    symbolic_local = [idx for idx, feat_idx in enumerate(cols_J) if feat_idx in set(symbolic_feature_indices)]

    h_score = float(h_var3(subcube_kij, numeric_local, symbolic_local))
    if h_score > coherence_threshold:
        return None

    has_num = len(numeric_local) > 0
    has_sym = len(symbolic_local) > 0
    if has_num and has_sym:
        tric_type = "mixed"
    elif has_num:
        tric_type = "numeric"
    else:
        tric_type = "symbolic"

    feature_names = [feature_columns[j] for j in cols_J]
    feature_groups = {
        "numeric": [feature_columns[j] for j in cols_J if j in set(numeric_feature_indices)],
        "symbolic": [feature_columns[j] for j in cols_J if j in set(symbolic_feature_indices)],
    }

    row_window_ids = [window_ids[i] for i in rows_I]
    row_windows_meta = [windows_meta[i] for i in rows_I] if len(windows_meta) == len(window_ids) else []

    volume = len(rows_I) * len(cols_J) * len(contx_K)

    return {
        "id": int(tricluster_id),
        "type": tric_type,
        "pattern_string": pattern_row["pattern_string"],
        "support": int(len(rows_I)),
        "hvar3": h_score,
        "rows": rows_I,
        "cols": cols_J,
        "contexts": contx_K,
        "row_window_ids": row_window_ids,
        "row_windows_meta": row_windows_meta,
        "feature_names": feature_names,
        "feature_groups": feature_groups,
        "shape": {
            "rows": int(len(rows_I)),
            "features": int(len(cols_J)),
            "contexts": int(len(contx_K)),
            "volume": int(volume),
        },
    }


def _extract_subcube_kij(
    cube: np.ndarray,
    rows_I: list[int],
    cols_J: list[int],
    contx_K: list[int],
) -> np.ndarray:
    """
    Returns subcube with shape (K, I, J), matching the original evaluation code.
    Input cube shape is (F, I, K).
    """
    out = np.empty((len(contx_K), len(rows_I), len(cols_J)), dtype=object)

    for k_pos, k_idx in enumerate(contx_K):
        for i_pos, i_idx in enumerate(rows_I):
            for j_pos, j_idx in enumerate(cols_J):
                out[k_pos, i_pos, j_pos] = cube[j_idx, i_idx, k_idx]

    return out


# -----------------------------------------------------------------------------
# HVar3
# -----------------------------------------------------------------------------

def h_var3(data_kij: np.ndarray, numeric_cols_idc: list[int], symbolic_cols_idc: list[int]) -> float:
    numeric_component = data_kij[:, :, numeric_cols_idc] if numeric_cols_idc else np.empty((0, 0, 0), dtype=float)
    symbolic_component = data_kij[:, :, symbolic_cols_idc] if symbolic_cols_idc else np.empty((0, 0, 0), dtype=object)

    numeric_metric = 0.0
    symbolic_metric = 0.0

    if numeric_component.size > 0:
        numeric_component = numeric_component.astype(float)
        numeric_metric = _coefficient_variation_numeric(numeric_component) / float(numeric_component.size)

    if symbolic_component.size > 0:
        symbolic_metric = _gini_impurity_3d(symbolic_component) / float(symbolic_component.size)

    return float(numeric_metric + symbolic_metric + _missing_values_ratio(data_kij))


def _coefficient_variation_numeric(array_3d: np.ndarray) -> float:
    """
    Reproduces the reference idea but adds numerical guards.
    data shape: (K, I, J)
    """
    if array_3d.size == 0:
        return 0.0

    std_x = np.nanstd(array_3d, axis=1)   # shape (K, J)
    mean_x = np.nanmean(array_3d, axis=1) # shape (K, J)

    with np.errstate(divide="ignore", invalid="ignore"):
        cv_x = np.where(np.abs(mean_x) < 1e-12, 0.0, std_x / np.abs(mean_x))

    cv_x = np.nan_to_num(cv_x, nan=0.0, posinf=0.0, neginf=0.0)
    avg_cv_x = np.mean(cv_x, axis=0)
    return float(np.mean(avg_cv_x))


def _gini_impurity_3d(data: np.ndarray) -> float:
    """
    data shape: (K, I, J)
    """
    if data.size == 0:
        return 0.0

    def calculate_gini(labels: np.ndarray) -> float:
        kept = []
        for x in labels:
            if not _is_missing(x):
                kept.append(str(x))

        if not kept:
            return 0.0

        values, counts = np.unique(np.array(kept, dtype=object), return_counts=True)
        probs = counts / counts.sum()
        return float(1.0 - np.sum(probs ** 2))

    gini_x = np.apply_along_axis(calculate_gini, axis=1, arr=data)  # shape (K, J)
    avg_gini_x = np.mean(gini_x, axis=0)
    return float(np.mean(avg_gini_x))


def _missing_values_ratio(data: np.ndarray) -> float:
    flat = data.reshape(-1)
    if flat.size == 0:
        return 0.0

    missing = 0
    for x in flat:
        if _is_missing(x):
            missing += 1
    return float(missing / float(flat.size))


# -----------------------------------------------------------------------------
# Post-processing
# -----------------------------------------------------------------------------

def _deduplicate_triclusters(triclusters: list[dict]) -> list[dict]:
    seen = set()
    out = []

    for tric in triclusters:
        key = (
            tuple(tric["rows"]),
            tuple(tric["cols"]),
            tuple(tric["contexts"]),
        )
        if key not in seen:
            seen.add(key)
            out.append(tric)

    return out


def _filter_overlapping_triclusters(
    triclusters: list[dict],
    overlap_threshold: float,
) -> list[dict]:
    if not triclusters:
        return []

    kept: list[dict] = []

    for candidate in sorted(
        triclusters,
        key=lambda x: (x["hvar3"], -x["shape"]["volume"])
    ):
        should_keep = True
        for chosen in kept:
            jacc = _tricluster_jaccard(candidate, chosen)
            if jacc >= overlap_threshold:
                should_keep = False
                break
        if should_keep:
            kept.append(candidate)

    return kept


def _tricluster_jaccard(t1: dict, t2: dict) -> float:
    i1, j1, k1 = set(t1["rows"]), set(t1["cols"]), set(t1["contexts"])
    i2, j2, k2 = set(t2["rows"]), set(t2["cols"]), set(t2["contexts"])

    inter = len(i1 & i2) * len(j1 & j2) * len(k1 & k2)
    size1 = len(i1) * len(j1) * len(k1)
    size2 = len(i2) * len(j2) * len(k2)
    union = size1 + size2 - inter

    if union <= 0:
        return 0.0
    return float(inter / union)