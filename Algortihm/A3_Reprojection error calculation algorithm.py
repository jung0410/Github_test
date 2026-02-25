"""
A3: Reprojection Error Evaluation (Front-view RMSE)

This module evaluates reprojection error on a designated front-view point set.
For each calibration run (INIT and REFINED), it:
1) Estimates pose using solvePnP
2) Projects object points with cv2.projectPoints
3) Computes per-point residuals and RMSE in pixels

Pass/Fail criterion:
PASS  if Front_RMSE(INIT) <= Front_RMSE(REFINED)
FAIL  otherwise
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 12x9 grid helpers
# ============================================================
def build_objp(cols: int = 12, rows: int = 9, square_size_mm: float = 100.0) -> np.ndarray:
    ## in English
    objp = np.zeros((rows * cols, 3), np.float64)
    k = 0
    for y in range(1, rows + 1):
        for x in range(1, cols + 1):
            objp[k] = [x, y, 0.0]
            k += 1
    objp *= float(square_size_mm)
    return objp


def load_points_pkl(path: str, expected_n: int) -> np.ndarray:
    ## in English
    with open(path, "rb") as f:
        pts = pickle.load(f)

    if isinstance(pts, pd.DataFrame):
        pts = pts.iloc[:, :2].to_numpy(dtype=np.float64)

    pts = np.asarray(pts, dtype=np.float64)
    pts = np.squeeze(pts)

    if pts.ndim == 3 and pts.shape[1:] == (1, 2):
        pts = pts.reshape(-1, 2)

    if pts.shape != (expected_n, 2):
        raise ValueError(f"Expected ({expected_n},2), got {pts.shape} from {path}")

    return pts


def load_params_pkl(path: str) -> Dict[str, Any]:
    ## in English
    with open(path, "rb") as f:
        data = pickle.load(f)

    runs: List[Dict[str, Any]] = []

    # Case 1) Your pipeline schema (single payload)
    if isinstance(data, dict) and "camera_matrix" in data and "dist_coeffs" in data:
        K = np.array(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
        d = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1)

        K_init = data.get("K_init", None)
        if K_init is not None:
            K_init = np.array(K_init, dtype=np.float64).reshape(3, 3)

        runs.append({
            "K": K,
            "dist": d,
            "K_init": K_init,
            "RMSE_error(calib)": float(data.get("RMSE_error", np.nan)),
            "tag": "REFINED",
            "Number of image": None,
            "Count": None,
        })
        return {"runs": runs}

    # Case 2) "runs" schema (kept for compatibility)
    if isinstance(data, dict) and "runs" in data and isinstance(data["runs"], list):
        for r in data["runs"]:
            K = r.get("camera_matrix", r.get("Camera Matrix"))
            d = r.get("dist_coeffs", r.get("Distortion Coefficients"))
            K_init = r.get("K_init", r.get("camera_matrix_init", r.get("camera_matrix_init")))

            if K is None or d is None:
                continue

            run = {
                "K": np.array(K, dtype=np.float64).reshape(3, 3),
                "dist": np.array(d, dtype=np.float64).reshape(-1),
                "K_init": np.array(K_init, dtype=np.float64).reshape(3, 3) if K_init is not None else None,
                "RMSE_error(calib)": float(r.get("RMSE_error", np.nan)),
                "tag": r.get("tag", "REFINED"),
                "Number of image": r.get("Number of image", r.get("N", None)),
                "Count": r.get("Count", r.get("count", None)),
            }
            runs.append(run)

        return {"runs": runs}

    raise ValueError(f"Unsupported PKL structure: {path}")


# ============================================================
# Reprojection RMSE
# ============================================================
def solve_pnp_and_rmse(
    image_pts_px: np.ndarray,
    objp: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    ## in English
    if dist is None:
        dist = np.zeros(4, dtype=np.float64)

    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    if dist.size < 4:
        dist = np.pad(dist, (0, 4 - dist.size), mode="constant")

    img_pts = image_pts_px.astype(np.float64).reshape(-1, 1, 2)
    obj_pts = objp.astype(np.float64).reshape(-1, 1, 3)

    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return float("nan"), None, None, None

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)

    diff = image_pts_px.astype(np.float64) - proj
    rmse = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    return rmse, rvec, tvec, diff


def build_front_rmse_table(
    front_pts_px: np.ndarray,
    params_runs: List[Dict[str, Any]],
    cols: int = 12,
    rows: int = 9,
    square_size_mm: float = 100.0,
) -> pd.DataFrame:
    ## in English
    objp = build_objp(cols=cols, rows=rows, square_size_mm=square_size_mm)
    expected_n = cols * rows

    if front_pts_px.shape != (expected_n, 2):
        raise ValueError(f"Front points must be ({expected_n},2), got {front_pts_px.shape}")

    rows_out: List[Dict[str, Any]] = []

    for r in params_runs:
        K = r["K"]
        dist = r["dist"]
        tag = r.get("tag", "REFINED")

        rmse, rvec, tvec, diff = solve_pnp_and_rmse(front_pts_px, objp, K, dist)

        rows_out.append({
            "tag": tag,
            "Number of image": r.get("Number of image", None),
            "Count": r.get("Count", None),
            "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "k1": float(dist[0]) if dist is not None and dist.size > 0 else np.nan,
            "k2": float(dist[1]) if dist is not None and dist.size > 1 else np.nan,
            "p1": float(dist[2]) if dist is not None and dist.size > 2 else np.nan,
            "p2": float(dist[3]) if dist is not None and dist.size > 3 else np.nan,
            "RMSE_error(calib)": float(r.get("RMSE_error(calib)", np.nan)),
            "Front_RMSE": float(rmse),
        })

    return pd.DataFrame(rows_out)


def decide_pass_fail(df: pd.DataFrame) -> Tuple[str, float, float]:
    ## in English
    init_rows = df[df["tag"].astype(str).str.contains("INIT", case=False, na=False)]
    refined_rows = df[df["tag"].astype(str).str.contains("REFINED", case=False, na=False)]

    init_rmse = float(init_rows["Front_RMSE"].iloc[0]) if not init_rows.empty else np.nan
    refined_rmse = float(refined_rows["Front_RMSE"].iloc[0]) if not refined_rows.empty else np.nan

    if np.isnan(init_rmse) or np.isnan(refined_rmse):
        return "NOT_PASS", init_rmse, refined_rmse

    return ("PASS" if init_rmse <= refined_rmse else "NOT_PASS"), init_rmse, refined_rmse


def run_a3(
    params_pkl: str,
    front_pkl: str,
    out_dir: str,
    cols: int = 12,
    rows: int = 9,
    square_size_mm: float = 100.0,
) -> Dict[str, Any]:
    ## in English
    os.makedirs(out_dir, exist_ok=True)

    params = load_params_pkl(params_pkl)
    runs = params["runs"]
    if not runs:
        raise RuntimeError("No runs found in params PKL.")

    expected_n = cols * rows
    front_pts = load_points_pkl(front_pkl, expected_n=expected_n)

    ## in English
    K_init = runs[0].get("K_init", None)
    runs_for_eval: List[Dict[str, Any]] = []

    if K_init is not None:
        runs_for_eval.append({
            "K": K_init,
            "dist": np.zeros(4, dtype=np.float64),
            "tag": "INIT (dist=0)",
            "Number of image": runs[0].get("Number of image", None),
            "Count": -1,
            "RMSE_error(calib)": np.nan,
        })

    ## in English
    for r in runs:
        r = dict(r)
        r["tag"] = "REFINED"
        runs_for_eval.append(r)

    df = build_front_rmse_table(
        front_pts_px=front_pts,
        params_runs=runs_for_eval,
        cols=cols,
        rows=rows,
        square_size_mm=square_size_mm,
    )

    ## in English
    out_xlsx = os.path.join(out_dir, "A3_front_reprojection.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Front_RMSE", index=False)

    verdict, init_rmse, refined_rmse = decide_pass_fail(df)

    ## in English
    out_txt = os.path.join(out_dir, "A3_verdict.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"VERDICT: {verdict}\n")
        f.write(f"INIT_Front_RMSE: {init_rmse}\n")
        f.write(f"REFINED_Front_RMSE: {refined_rmse}\n")
        f.write(f"FRONT_PKL: {front_pkl}\n")
        f.write(f"PARAMS_PKL: {params_pkl}\n")

    print(f"✅ [A3] INIT Front_RMSE   : {init_rmse}")
    print(f"✅ [A3] REFINED Front_RMSE: {refined_rmse}")
    print(f"✅ [A3] VERDICT           : {verdict}")
    print(f"✅ [A3] Saved Excel        : {out_xlsx}")
    print(f"✅ [A3] Saved verdict      : {out_txt}")

    return {
        "df": df,
        "verdict": verdict,
        "init_rmse": init_rmse,
        "refined_rmse": refined_rmse,
        "excel": out_xlsx,
        "verdict_txt": out_txt,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A3 Front reprojection RMSE + PASS/FAIL")
    parser.add_argument("--params_pkl", required=True)
    parser.add_argument("--front_pkl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--square_size_mm", type=float, default=100.0)
    args = parser.parse_args()

    run_a3(
        params_pkl=args.params_pkl,
        front_pkl=args.front_pkl,
        out_dir=args.out_dir,
        cols=args.cols,
        rows=args.rows,
        square_size_mm=args.square_size_mm,
    )