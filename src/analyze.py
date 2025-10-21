from __future__ import annotations
import argparse
import ast
import json
import os
import pickle
import sys
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

#!/usr/bin/env python3
"""
Calculate average rate of exact support recovery from datasets in results/raw.

Usage:
    python src/analyze.py [--dir results/raw] [--recursive]

The script looks for common file types (csv, json, npy, npz, pkl) and tries to
extract "true" and "estimated" supports or coefficient vectors. The overall
report is the total fraction of runs where estimated support exactly equals
true support (aggregated across all files).
"""



# Tolerance for treating a coefficient as nonzero
NONZERO_TOL = 1e-8

# Candidate key names for true/estimated supports or coefficient vectors
TRUE_SUPPORT_KEYS = {
    "true_support",
    "support_true",
    "S_true",
    "true_S",
    "support_true_indices",
    "true_indices",
}
EST_SUPPORT_KEYS = {
    "estimated_support",
    "support_est",
    "S_hat",
    "est_support",
    "hat_support",
    "support_est_indices",
    "est_indices",
}
TRUE_COEF_KEYS = {
    "beta_true",
    "beta0",
    "coef_true",
    "beta_true_vec",
    "beta_true_",
}
EST_COEF_KEYS = {
    "beta_hat",
    "beta",
    "coef_est",
    "beta_hat_vec",
    "beta_hat_",
    "beta_est",
}


def load_file(path: str) -> Any:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv", ".tsv"}:
        sep = "\t" if ext == ".tsv" else ","
        return pd.read_csv(path, sep=sep, dtype=object)
    if ext == ".json":
        with open(path, "r") as f:
            return json.load(f)
    if ext == ".npz":
        return dict(np.load(path, allow_pickle=True))
    if ext == ".npy":
        return np.load(path, allow_pickle=True)
    if ext in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            return pickle.load(f)
    # Fallback: try reading as text and parse JSON
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        raise ValueError(f"Unsupported or unreadable file type: {path}")


def as_array(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    if isinstance(x, (int, float, bool)):
        return np.asarray([x])
    # pandas Series or DataFrame column
    if isinstance(x, pd.Series):
        return x.to_numpy()
    # string that looks like a Python literal list/tuple
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            return as_array(val)
        except Exception:
            # last resort: split on commas
            parts = [p.strip() for p in x.strip("[]() ").split(",") if p.strip()]
            try:
                return np.asarray([float(p) for p in parts])
            except Exception:
                return np.asarray(parts)
    raise ValueError(f"Can't convert to array: {type(x)}")


def support_from_coef(vec: np.ndarray, tol: float = NONZERO_TOL) -> np.ndarray:
    vec = np.asarray(vec, dtype=float).ravel()
    return np.nonzero(np.abs(vec) > tol)[0].astype(int)


def parse_supports_from_object(obj: Any) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Try to extract (true_support, est_support) pairs from the object.

    Returns a list of pairs. Each pair corresponds to one run / record.
    """
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    # Case: a dict-like container with arrays
    if isinstance(obj, dict):
        keys = set(obj.keys())
        # Direct support arrays
        true_keys = keys & TRUE_SUPPORT_KEYS
        est_keys = keys & EST_SUPPORT_KEYS
        if true_keys and est_keys:
            tkey = sorted(true_keys)[0]
            ekey = sorted(est_keys)[0]
            t = as_array(obj[tkey])
            e = as_array(obj[ekey])
            # If these are 1D arrays of arrays (multiple runs), iterate
            if t.ndim == 1 and e.ndim == 1 and len(t) == len(e) and isinstance(t[0], (list, tuple, np.ndarray, str)):
                for tt, ee in zip(t, e):
                    pairs.append((support_as_indices(tt), support_as_indices(ee)))
                return pairs
            pairs.append((support_as_indices(t), support_as_indices(e)))
            return pairs

        # If coefficient vectors are present
        true_coef = keys & TRUE_COEF_KEYS
        est_coef = keys & EST_COEF_KEYS
        if true_coef and est_coef:
            tk = sorted(true_coef)[0]
            ek = sorted(est_coef)[0]
            t = as_array(obj[tk])
            e = as_array(obj[ek])
            # handle multiple runs if first dim matches and elements are vectors
            if t.ndim == 2 and e.ndim == 2 and t.shape[0] == e.shape[0]:
                for tt, ee in zip(t, e):
                    pairs.append((support_from_coef(tt), support_from_coef(ee)))
                return pairs
            # single vectors
            pairs.append((support_from_coef(t), support_from_coef(e)))
            return pairs

        # Maybe the dict maps run IDs to structures
        # Try to recursively scan values
        for v in obj.values():
            try:
                sub = parse_supports_from_object(v)
                if sub:
                    pairs.extend(sub)
            except Exception:
                continue
        return pairs

    # Case: pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        df = obj
        # If columns for supports exist
        col_names = set(df.columns.astype(str))
        true_cols = col_names & TRUE_SUPPORT_KEYS
        est_cols = col_names & EST_SUPPORT_KEYS
        if true_cols and est_cols:
            tcol = sorted(true_cols)[0]
            ecol = sorted(est_cols)[0]
            for tcell, ecell in zip(df[tcol], df[ecol]):
                pairs.append((support_as_indices(tcell), support_as_indices(ecell)))
            return pairs
        # If coef columns exist
        true_coef_cols = col_names & TRUE_COEF_KEYS
        est_coef_cols = col_names & EST_COEF_KEYS
        if true_coef_cols and est_coef_cols:
            tk = sorted(true_coef_cols)[0]
            ek = sorted(est_coef_cols)[0]
            for tcell, ecell in zip(df[tk], df[ek]):
                pairs.append((support_from_coef(as_array(tcell)), support_from_coef(as_array(ecell))))
            return pairs
        # Try per-row detection: each row may contain fields named like above
        for _, row in df.iterrows():
            rec = row.to_dict()
            sub = parse_supports_from_object(rec)
            if sub:
                pairs.extend(sub)
        return pairs

    # Case: numpy array of structured records or 2D arrays (pairs of vectors)
    if isinstance(obj, np.ndarray):
        if obj.dtype.names:  # structured array
            return parse_supports_from_object({n: obj[n] for n in obj.dtype.names})
        # 2D array: maybe stacked coef pairs: shape (n, 2*m) or tuple of pairs
        if obj.ndim == 2:
            # Heuristic: if even number of columns and two halves correspond to true/est
            ncols = obj.shape[1]
            if ncols % 2 == 0:
                half = ncols // 2
                left = obj[:, :half]
                right = obj[:, half:]
                for l, r in zip(left, right):
                    pairs.append((support_from_coef(l), support_from_coef(r)))
                return pairs
        # 1D array of objects (list-like runs)
        if obj.ndim == 1 and obj.size and isinstance(obj[0], (list, tuple, np.ndarray, dict)):
            for item in obj:
                sub = parse_supports_from_object(item)
                if sub:
                    pairs.extend(sub)
        return pairs

    # Case: simple tuple/list of two elements: treat as a single pair
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        a, b = obj
        try:
            a_arr = as_array(a)
            b_arr = as_array(b)
            # If elements are vectors -> treat as coef vectors
            if a_arr.ndim == 1 and b_arr.ndim == 1 and a_arr.shape == b_arr.shape:
                return [(support_from_coef(a_arr), support_from_coef(b_arr))]
            # Otherwise try to interpret as support indices
            return [(support_as_indices(a), support_as_indices(b))]
        except Exception:
            # fall through to recursive attempts
            pass

    return []


def support_as_indices(x: Any) -> np.ndarray:
    """
    Convert an arbitrary representation to an array of support indices (int).
    Accepts boolean masks, index lists, or coefficient vectors.
    """
    arr = None
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        try:
            arr = as_array(x)
        except Exception:
            pass
    if arr is None and isinstance(x, str):
        try:
            arr = as_array(x)
        except Exception:
            arr = None

    if arr is None:
        # Last resort: if x is scalar, treat as empty or single index
        if x is None:
            return np.array([], dtype=int)
        try:
            return np.asarray([int(x)], dtype=int)
        except Exception:
            return np.array([], dtype=int)

    # If boolean mask
    if arr.dtype == bool or (np.issubdtype(arr.dtype, np.number) and set(np.unique(arr).tolist()) <= {0, 1}):
        # boolean or 0/1 mask
        mask = arr.astype(bool).ravel()
        return np.nonzero(mask)[0].astype(int)
    # If numeric indices
    if np.issubdtype(arr.dtype, np.integer):
        return np.asarray(arr, dtype=int).ravel()
    # If floats -> maybe coefficients
    if np.issubdtype(arr.dtype, np.floating):
        return support_from_coef(arr)
    # If strings -> try literal eval
    if arr.dtype.type is np.str_ or arr.dtype.type is np.object_:
        out_idxs = []
        for el in arr:
            if isinstance(el, str):
                try:
                    parsed = ast.literal_eval(el)
                    out_idxs.extend(list(as_array(parsed).ravel()))
                except Exception:
                    try:
                        out_idxs.append(int(el))
                    except Exception:
                        continue
            else:
                try:
                    out_idxs.append(int(el))
                except Exception:
                    continue
        if out_idxs:
            return np.unique(np.asarray(out_idxs, dtype=int))
    return np.array([], dtype=int)


def exact_match(supp_a: np.ndarray, supp_b: np.ndarray) -> bool:
    a = np.unique(np.asarray(supp_a, dtype=int))
    b = np.unique(np.asarray(supp_b, dtype=int))
    return a.size == b.size and np.all(a == b)


def compute_rates_for_path(path: str) -> Tuple[int, int]:
    """
    Returns (n_correct, n_total) for the given file path.
    """
    try:
        obj = load_file(path)
    except Exception as e:
        print(f"Skipping {path}: load error: {e}", file=sys.stderr)
        return 0, 0

    try:
        pairs = parse_supports_from_object(obj)
    except Exception as e:
        print(f"Skipping {path}: parse error: {e}", file=sys.stderr)
        return 0, 0

    if not pairs:
        print(f"Skipping {path}: no support pairs found", file=sys.stderr)
        return 0, 0

    n_total = 0
    n_correct = 0
    for t, e in pairs:
        n_total += 1
        if exact_match(t, e):
            n_correct += 1
    return n_correct, n_total


def find_files(root: str, recursive: bool = True) -> List[str]:
    exts = ("*.csv", "*.tsv", "*.json", "*.npz", "*.npy", "*.pkl", "*.pickle")
    paths = []
    if recursive:
        for ext in exts:
            paths.extend(glob(os.path.join(root, "**", ext), recursive=True))
    else:
        for ext in exts:
            paths.extend(glob(os.path.join(root, ext)))
    return sorted(paths)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Average exact support recovery rate calculator")
    p.add_argument("--dir", "-d", default="results/raw", help="Directory containing raw result files")
    p.add_argument("--recursive", action="store_true", help="Search directory recursively")
    p.add_argument("--debug", action="store_true", help="Print per-file rates to stderr")
    args = p.parse_args(argv)

    root = args.dir
    if not os.path.isdir(root):
        print(f"Directory not found: {root}", file=sys.stderr)
        return 2

    files = find_files(root, recursive=args.recursive)
    if not files:
        print(f"No result files found under {root}", file=sys.stderr)
        return 1

    total_correct = 0
    total_runs = 0
    per_file = []
    for f in files:
        c, n = compute_rates_for_path(f)
        if n:
            per_file.append((f, c, n))
            total_correct += c
            total_runs += n
            if args.debug:
                rate = c / n
                print(f"{f}: {c}/{n} = {rate:.4f}", file=sys.stderr)

    if total_runs == 0:
        print("No runs with detectable support pairs found.", file=sys.stderr)
        return 1

    overall_rate = total_correct / total_runs
    # Print a short summary to stdout
    print(f"Total runs: {total_runs}")
    print(f"Exact support recovered in {total_correct} runs")
    print(f"Average exact support recovery rate: {overall_rate:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())