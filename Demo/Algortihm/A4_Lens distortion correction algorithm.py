"""
A4: Lens Distortion Correction

This module applies lens distortion correction to a batch of images using
precomputed intrinsic camera parameters. The distortion correction is performed
via image remapping based on the estimated camera matrix and distortion
coefficients.

Inputs:
- Distorted images (jpg/jpeg/png)
- Calibration parameter PKL file containing camera_matrix and dist_coeffs

Outputs:
- Undistorted images saved to the specified output directory
"""

from __future__ import annotations

import os
import pickle
import argparse
import multiprocessing
from functools import partial

from tqdm.contrib.concurrent import process_map

# ---- Project-specific imports ----
from lib.L1_Image_Conversion import *
from lib.L2_Point_Detection_Conversion import *
from lib.L3_Zhang_Camera_Calibration import *
from lib.L4_Pipeline_Utilities import *
from lib.L5_Visualization_Utilities import *

# ============================================================
# Core undistortion functions
# ============================================================
def undistort_single_image(
    image_path: str,
    out_path: str,
    camera_matrix,
    dist_coeffs,
) -> None:
    """
    Apply lens distortion correction to a single image.

    Args:
        image_path (str): Path to input image.
        out_path (str): Path to save undistorted image.
        camera_matrix (np.ndarray): Intrinsic camera matrix.
        dist_coeffs (np.ndarray): Distortion coefficients.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    undistort_image_remap(image_path, out_path, camera_matrix, dist_coeffs)


def _process_one_file(
    filename: str,
    input_dir: str,
    output_dir: str,
    camera_matrix,
    dist_coeffs,
) -> None:
    """Wrapper for parallel processing of a single image."""
    in_path = os.path.join(input_dir, filename)
    base, _ = os.path.splitext(filename)
    out_path = os.path.join(output_dir, f"{base}.jpg")

    if os.path.exists(out_path):
        return

    undistort_single_image(in_path, out_path, camera_matrix, dist_coeffs)


def process_directory(
    input_dir: str,
    output_dir: str,
    camera_matrix,
    dist_coeffs,
    n_jobs: int = -1,
    chunksize: int = 20,
) -> None:
    """
    Apply lens distortion correction to all images in a directory.

    Args:
        input_dir (str): Directory containing distorted images.
        output_dir (str): Directory to save corrected images.
        camera_matrix (np.ndarray): Intrinsic camera parameters.
        dist_coeffs (np.ndarray): Distortion coefficients.
        n_jobs (int): Number of worker processes (-1 = use all cores).
        chunksize (int): Chunk size for parallel processing.
    """
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        raise RuntimeError(f"No image files found in: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    if n_jobs is None or n_jobs <= 0:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)

    func = partial(
        _process_one_file,
        input_dir=input_dir,
        output_dir=output_dir,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )

    process_map(
        func,
        image_files,
        max_workers=n_jobs,
        chunksize=chunksize,
        desc="Lens distortion correction",
    )


# ============================================================
# CLI entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lens distortion correction using calibrated camera parameters"
    )
    parser.add_argument("--input", required=True, help="Directory with distorted images")
    parser.add_argument("--output", required=True, help="Directory to save corrected images")
    parser.add_argument(
        "--params",
        required=True,
        help="PKL file containing camera_matrix and dist_coeffs",
    )
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel workers")

    args = parser.parse_args()

    with open(args.params, "rb") as f:
        params = pickle.load(f)

    camera_matrix = params["camera_matrix"]
    dist_coeffs = params["dist_coeffs"]

    process_directory(
        input_dir=args.input,
        output_dir=args.output,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        n_jobs=args.n_jobs,
    )

    print("✅ Lens distortion correction completed.")
