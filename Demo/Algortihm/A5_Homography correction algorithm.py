"""
Algorithm 5 (A5): Homography Correction (Estimation + Baseline Projection)

This script estimates a homography matrix H from the baseline PKL data and
applies H to the baseline marker coordinates (up to apply_homography stage).
The output is saved as a PKL file to be used by Algorithm 6.

Inputs:
- pkl_dir: directory containing per-frame PKL files with keys {"data", "ref_data"}

Outputs:
- homography_params.pkl containing:
  - H (3x3)
  - ref0 (N,2) baseline reference points
  - data0_f (N,2) filtered baseline data points
  - data0_h (N,2) baseline points after apply_homography (NOT aligned yet)
  - meta: grid size, square size, baseline file, etc.
"""

from __future__ import annotations

import os
import pickle
import argparse
import numpy as np
import pandas as pd

# ---- Project-specific imports (adjust paths if needed) ----
from lib.L1_Image_Conversion import *
from lib.L2_Point_Detection_Conversion import *
from lib.L3_Zhang_Camera_Calibration import *
from lib.L4_Pipeline_Utilities import *
from lib.L5_Visualization_Utilities import *


def estimate_homography_from_baseline(
    pkl_dir: str,
    square_size_mm: float = 100.0,
    grid_rows: int = 9,
    grid_cols: int = 12,
    baseline_index: int = 0,
    filter_thr: float = 50.0,
) -> dict:
    """Estimate homography H and apply it to baseline points (apply_homography stage)."""
    pkl_files = [f for f in os.listdir(pkl_dir) if f.lower().endswith(".pkl")]
    pkl_files.sort(key=natural_key)
    if not pkl_files:
        raise RuntimeError(f"No PKL found in: {pkl_dir}")

    base_file = pkl_files[baseline_index]
    with open(os.path.join(pkl_dir, base_file), "rb") as f:
        payload0 = pickle.load(f)

    data0 = ensure_2col(payload0.get("data"))
    ref0 = ensure_2col(payload0.get("ref_data"))
    if data0 is None or ref0 is None:
        raise RuntimeError(f"Baseline invalid: {base_file}")

    data0 = np.asarray(data0, dtype=float)
    ref0 = np.asarray(ref0, dtype=float)

    # Filtering (keep your pipeline)
    data0_f = iterative_filtering_12_9(data0, filter_thr)

    # Pre-homography scale estimation (kept for record)
    avg_dist_px0, _ = how_much_rect(pd.DataFrame(data0_f), grid_rows, grid_cols)
    um_per_pixel_pre = (6.0 * 1000.0) / avg_dist_px0

    # Object points (mm)
    obj_points_mm = Makeobjp(square_size_mm, grid_rows, grid_cols)
    obj_points_mm_xy = np.asarray(obj_points_mm)[:, :2]

    # Homography estimation
    H = compute_homography_noraml(data0_f, obj_points_mm_xy)

    # A5 ends at baseline homography projection
    data0_h = apply_homography(H, np.array(data0_f))
    data0_h = np.asarray(data0_h, dtype=float)

    return {
        "H": np.asarray(H, dtype=float),
        "ref0": ref0,
        "data0_f": np.asarray(data0_f, dtype=float),
        "data0_h": data0_h,
        "um_per_pixel_pre": float(um_per_pixel_pre),
        "baseline_file": base_file,
        "grid_rows": int(grid_rows),
        "grid_cols": int(grid_cols),
        "square_size_mm": float(square_size_mm),
        "filter_thr": float(filter_thr),
    }


def save_homography_params(out_path: str, payload: dict) -> None:
    """Save homography payload to PKL."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A5: Homography estimation + baseline projection")
    p.add_argument("--pkl_dir", required=True, help="Directory containing PKL files")
    p.add_argument("--out", required=True, help="Output PKL path (e.g., ./results/homography_params.pkl)")
    p.add_argument("--square_size_mm", type=float, default=100.0)
    p.add_argument("--grid_rows", type=int, default=9)
    p.add_argument("--grid_cols", type=int, default=12)
    p.add_argument("--baseline_index", type=int, default=0)
    p.add_argument("--filter_thr", type=float, default=50.0)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()

    payload = estimate_homography_from_baseline(
        pkl_dir=args.pkl_dir,
        square_size_mm=args.square_size_mm,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        baseline_index=args.baseline_index,
        filter_thr=args.filter_thr,
    )

    save_homography_params(args.out, payload)
    print(f"✅ Saved homography parameters: {args.out}")
