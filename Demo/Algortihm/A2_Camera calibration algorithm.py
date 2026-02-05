"""
A2: Camera Calibration using Zhang's Method with LM Refinement

This module performs intrinsic camera calibration using detected marker points
stored in PKL files. The calibration pipeline follows Zhang's method and applies
Levenberg–Marquardt optimization for refinement.

Outputs:
- Excel file containing intrinsic parameters and reprojection error
- PKL file containing calibration parameters and optional pose information
"""

from __future__ import annotations

import os
import pickle
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook

# ---- Project-specific imports ----
# Mode 1,2 uses:
from lib.L1_Image_Conversion import *
from lib.L2_Point_Detection_Conversion import *
from lib.L3_Zhang_Camera_Calibration import *
from lib.L4_Pipeline_Utilities import *
from lib.L5_Visualization_Utilities import *



# ============================================================
# Data loading utilities
# ============================================================
def load_pkl_points(folder_path: str) -> List[Tuple[str, object]]:
    """Load all PKL files from a folder."""
    files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".pkl"))
    data = []
    for fname in files:
        with open(os.path.join(folder_path, fname), "rb") as f:
            data.append((fname, pickle.load(f)))
    return data


def normalize_points(data, expected_n: int) -> np.ndarray:
    """Convert input data to (N, 2) float array."""
    if isinstance(data, pd.DataFrame):
        pts = data.iloc[:, :2].to_numpy(dtype=np.float64)
    else:
        pts = np.asarray(data, dtype=np.float64)

    pts = np.squeeze(pts)
    if pts.shape != (expected_n, 2):
        raise ValueError(f"Expected shape ({expected_n}, 2), got {pts.shape}")
    return pts


def build_image_points(
    folder_path: str,
    grid_size=(12, 9),
    apply_filter: bool = True,
    filter_thr: float = 50.0,
):
    """Build a list of image points from PKL files."""
    cols, rows = grid_size
    expected_n = cols * rows

    image_pts = []
    filenames = []

    for fname, data in load_pkl_points(folder_path):
        pts = normalize_points(data, expected_n)
        if apply_filter:
            pts = iterative_filtering_12_9(pts, filter_thr)
        image_pts.append(pts.astype(np.float64))
        filenames.append(fname)

    if not image_pts:
        raise RuntimeError("No valid point sets found.")

    return image_pts, filenames


# ============================================================
# Calibration runner
# ============================================================
def run_calibration(
    input_dir: str,
    output_dir: str,
    grid_size=(12, 9),
    spacing: float = 100.0,
    apply_filter: bool = True,
    filter_thr: float = 50.0,
):
    """Run Zhang + LM calibration on a folder of PKL files."""
    image_pts, filenames = build_image_points(
        input_dir, grid_size, apply_filter, filter_thr
    )

    K, dist, rvecs, tvecs, rmse, K_init = calibrate_zhang_then_lm(
        image_pts_list=image_pts,
        grid_size=grid_size,
        spacing=spacing,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_pkl = os.path.join(output_dir, "calibration_params.pkl")

    with open(out_pkl, "wb") as f:
        pickle.dump(
            {
                "camera_matrix": K,
                "dist_coeffs": dist,
                "RMSE_error": rmse,
                "camera_matrix_init": K_init,
            },
            f,
        )

    return K, dist, rmse, out_pkl


# ============================================================
# CLI entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration (Zhang + LM)")
    parser.add_argument("--input", required=True, help="Folder containing PKL files")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--spacing", type=float, default=100.0)
    args = parser.parse_args()

    K, dist, rmse, out_pkl = run_calibration(
        input_dir=args.input,
        output_dir=args.output,
        grid_size=(args.cols, args.rows),
        spacing=args.spacing,
    )

    print("Calibration completed.")
    print("RMSE (pixels):", rmse)
    print("Saved:", out_pkl)
