import os
import re
import cv2
import pickle
from pathlib import Path
from typing import Any, List
import numpy as np


from lib.L1_Image_Conversion import *
from lib.L2_Point_Detection import *
from lib.L3_Zhangs_Calibration import *
# from lib.L5_Pipeline_utility import *
# from lib.L6_ETC_visualization import *

_IMG_EXTS = {".jpg", ".jpeg", ".png"}

def natural_key(path_str: str) -> List[Any]:
    name = Path(path_str).name
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def list_images(dir_path: str) -> List[str]:
    p = Path(dir_path)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    paths = [str(p / f) for f in os.listdir(p) if Path(f).suffix.lower() in _IMG_EXTS]
    paths.sort(key=natural_key)
    return paths

def read_gray(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to load image: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def save_pickle(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def ensure_2col(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            return None
        return arr.reshape(-1, 2)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    return None
import os
import logging
import multiprocessing
from pathlib import Path
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm.contrib.concurrent import process_map




def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        level=level
    )

# ============================================================
# [A1] Marker detection (calibration images) - saves GT_*.pkl
# ============================================================

def _A1_worker(file_path: str, out_dir: str, selection: Optional[str] = None):
    base = Path(file_path).stem
    out_path = str(Path(out_dir) / f"GT_{base}.pkl")
    if os.path.exists(out_path):
        return

    gray = read_gray(file_path)

    if selection is None or selection == "Circle_Centroid":
        Data, avg = find_Centroid_12_9(gray)
    elif selection == "Circle_ZernikeEdge":
        Data, avg = find_ZernikeEdge_curvefit_12_9(gray)
    elif selection == "cv":
        Data = find_cv_12_9(gray)
        avg = 0
    elif selection == "Square_Centroid":
        Data, avg = find_Square_12_9_Centroid(gray)
    else:
        Data, avg = find_Centroid_12_9(gray)

    if Data is None:
        logging.warning(f"[A1] No data: {Path(file_path).name}")
        return

    Data = np.asarray(Data, dtype=float).reshape(-1, 2)
    Data = iterative_filtering_12_9(Data, 50)
    save_pickle(out_path, Data)

def run_A1_marker_detection(input_dir: str, out_dir: str, selection=None, n_jobs=-1, chunksize=30):
    os.makedirs(out_dir, exist_ok=True)
    imgs = list_images(input_dir)
    if not imgs:
        raise RuntimeError(f"No images in {input_dir}")

    if n_jobs is None or n_jobs <= 0:
        n_jobs = max(1, multiprocessing.cpu_count() - 2)

    func = partial(_A1_worker, out_dir=out_dir, selection=selection)

    process_map(
        func,
        imgs,
        max_workers=n_jobs,
        chunksize=chunksize,
        desc=f"A1 Detect (calib): {Path(input_dir).name}",
    )

# ============================================================
# [A2] Camera calibration (from pkl) - returns dict
# ============================================================

def load_saved_data_pkl(pkl_dir: str) -> List[Tuple[str, np.ndarray]]:
    files = [f for f in os.listdir(pkl_dir) if f.lower().endswith(".pkl")]
    files.sort()
    out = []
    for fn in files:
        path = os.path.join(pkl_dir, fn)
        Data = load_pickle(path)
        out.append((fn, Data))
    return out

def perform_camera_calibration_from_data(
    saved_data: List[Tuple[str, np.ndarray]],
    image_size=(4000, 3000),
    square_size_mm=100.0,
):
    objpoints, imgpoints = [], []
    filenames = []

    rows, cols = 9, 12
    objp = np.zeros((rows * cols, 3), np.float32)
    idx = 0
    for y in range(1, rows + 1):
        for x in range(1, cols + 1):
            objp[idx] = [x, y, 0]
            idx += 1
    objp *= float(square_size_mm)

    for filename, Data in saved_data:
        if Data is None:
            continue
        Data = np.asarray(Data, dtype=np.float32).reshape(-1, 2)
        objpoints.append(objp)
        imgpoints.append(Data)
        filenames.append(filename)

    if not objpoints:
        raise RuntimeError("No valid pkl data for calibration")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=cv2.CALIB_FIX_K3
    )

    # RMSE
    total_sq, total_pts = 0.0, 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        proj = proj.reshape(-1, 2)
        err = (imgpoints[i] - proj)
        total_sq += float(np.sum(err**2))
        total_pts += proj.shape[0]
    rmse = np.sqrt(total_sq / max(1, total_pts))

    return {
        "ret": ret,
        "camera_matrix": K,
        "dist_coeffs": dist,
        "RMSE_error": float(rmse),
        "rvecs": rvecs,
        "tvecs": tvecs,
        "filenames": filenames,
    }

def save_calibration_params(calib_result: Dict[str, Any], out_path: str):
    payload = {
        "camera_matrix": calib_result["camera_matrix"],
        "dist_coeffs": calib_result["dist_coeffs"],
        "RMSE_error": calib_result["RMSE_error"],
    }
    save_pickle(out_path, payload)

# ============================================================
# [A4] Undistortion (ProcessPool-safe)
# ============================================================

def _undistort_worker(filename: str, input_dir: str, out_dir: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
    in_path = os.path.join(input_dir, filename)
    base = Path(filename).stem
    out_path = os.path.join(out_dir, f"{base}.jpg")
    if os.path.exists(out_path):
        return
    os.makedirs(out_dir, exist_ok=True)
    undistort_image_remap(in_path, out_path, camera_matrix, dist_coeffs)

def run_A4_undistort_images(input_dir, out_dir, camera_matrix, dist_coeffs, n_jobs=-1, chunksize=20):
    os.makedirs(out_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not image_files:
        raise RuntimeError(f"No images found in: {input_dir}")

    if n_jobs is None or n_jobs <= 0:
        n_jobs = max(1, multiprocessing.cpu_count() - 2)

    func = partial(
        _undistort_worker,
        input_dir=input_dir,
        out_dir=out_dir,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs
    )

    process_map(
        func,
        image_files,
        max_workers=n_jobs,
        desc=f"A4 Undistorting ({os.path.basename(input_dir)})",
        chunksize=chunksize
    )

# ============================================================
# [A1_plus_10] Detection on undistorted moved images
# ============================================================

def _save_payload_pickle(out_path: str, data: np.ndarray, ref_data: np.ndarray):
    payload = {"data": data, "ref_data": ref_data}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        import pickle
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

def process_first_image_plus10(img_path: str, out_dir: str) -> float:
    base = Path(img_path).stem
    out_path = str(Path(out_dir) / f"GT_{base}.pkl")
    gray = read_gray(img_path)
    Data_df, Ref_df, avg = find_Centroid_12_9_first_image_plus_ten(gray)
    if Data_df is None or Ref_df is None:
        raise RuntimeError(f"First image failed: {Path(img_path).name}")
    Data = np.asarray(Data_df, dtype=float).reshape(-1, 2)
    Ref  = np.asarray(Ref_df,  dtype=float).reshape(-1, 2)
    _save_payload_pickle(out_path, Data, Ref)
    return float(avg)

def _plus10_worker(img_path: str, out_dir: str, avg: float):
    base = Path(img_path).stem
    out_path = str(Path(out_dir) / f"GT_{base}.pkl")
    if os.path.exists(out_path):
        return
    gray = read_gray(img_path)
    Data_df, Ref_df, _ = find_Centroid_12_9_first_image_plus_ten(gray, avg)
    if Data_df is None or Ref_df is None:
        logging.warning(f"[A1+10] No data: {Path(img_path).name}")
        return
    Data = np.asarray(Data_df, dtype=float).reshape(-1, 2)
    Ref  = np.asarray(Ref_df,  dtype=float).reshape(-1, 2)
    _save_payload_pickle(out_path, Data, Ref)

def run_A1_plus10_marker_detection(input_dir: str, out_dir: str, n_jobs=-1, chunksize=30):
    os.makedirs(out_dir, exist_ok=True)
    imgs = list_images(input_dir)
    if not imgs:
        raise RuntimeError(f"No images in {input_dir}")

    avg = process_first_image_plus10(imgs[0], out_dir)
    rest = imgs[1:]
    if not rest:
        return

    if n_jobs is None or n_jobs <= 0:
        n_jobs = max(1, multiprocessing.cpu_count() - 2)

    func = partial(_plus10_worker, out_dir=out_dir, avg=avg)

    process_map(
        func,
        rest,
        max_workers=n_jobs,
        chunksize=chunksize,
        desc=f"A1+10 Detect (moved): {Path(input_dir).name}"
    )


def compute_displacement_from_pkl_dir(
    pkl_dir: str,
    square_size_mm: float = 100.0,
    grid_rows: int = 9,
    grid_cols: int = 12,
    baseline_index: int = 0,
    y_threshold_align: float = 50,
    x_threshold_align: float = 50,
):
    pkl_files = [f for f in os.listdir(pkl_dir) if f.lower().endswith(".pkl")]
    pkl_files.sort(key=natural_key)
    if not pkl_files:
        raise RuntimeError(f"No PKL found in: {pkl_dir}")

    base_file = pkl_files[baseline_index]
    import pickle
    with open(os.path.join(pkl_dir, base_file), "rb") as f:
        payload0 = pickle.load(f)

    data0 = ensure_2col(payload0.get("data"))
    ref0  = ensure_2col(payload0.get("ref_data"))
    if data0 is None or ref0 is None:
        raise RuntimeError(f"Baseline invalid: {base_file}")

    data0 = np.asarray(data0, dtype=float)
    data0 = iterative_filtering_12_9(data0, 50)

    avg_dist_px0, _ = how_much_rect(pd.DataFrame(data0), grid_rows, grid_cols)
    um_per_pixel = (6.0 * 1000.0) / avg_dist_px0

    obj_points_mm = Makeobjp(square_size_mm, grid_rows, grid_cols)
    obj_points_mm = np.asarray(obj_points_mm)
    obj_points_mm_xy = obj_points_mm[:, :2]

    H = compute_homography_noraml(data0, obj_points_mm_xy)

    data0_h = apply_homography(H, np.array(data0))
    data0_h = align_points_to_grid(data0_h, grid_rows, grid_cols, y_threshold_align, x_threshold_align)
    data0_h = np.asarray(data0_h, dtype=float)

    # scale update after H+align (너가 추가한 로직 유지)
    avg_dist_px0_h, _ = how_much_rect(pd.DataFrame(data0_h), grid_rows, grid_cols)
    um_per_pixel = (6.0 * 1000.0) / avg_dist_px0_h

    records = []
    for order, fname in enumerate(pkl_files):
        with open(os.path.join(pkl_dir, fname), "rb") as f:
            payload = pickle.load(f)

        data = ensure_2col(payload.get("data"))
        ref  = ensure_2col(payload.get("ref_data"))
        if data is None or ref is None:
            logging.warning(f"[PLOT] skip invalid: {fname}")
            continue

        data = np.asarray(data, dtype=float)
        ref  = np.asarray(ref, dtype=float)

        move = (ref - ref0).mean(axis=0)
        data_corr = data - move

        data_h = apply_homography(H, np.array(data_corr))
        data_h = align_points_to_grid(data_h, grid_rows, grid_cols, y_threshold_align, x_threshold_align)
        data_h = np.asarray(data_h, dtype=float)

        disp = data_h - data0_h
        disp_um = disp * um_per_pixel

        mag_each_um = np.hypot(disp_um[:, 0], disp_um[:, 1])
        mean_dx_um, mean_dy_um = np.nanmean(disp_um, axis=0)
        mean_disp_um = float(np.hypot(mean_dx_um, mean_dy_um))

        records.append({
            "order": order,
            "file": fname,
            "displacement_each_um": disp_um,
            "disp_mag_each_um": mag_each_um,
            "mean_dx_um": float(mean_dx_um),
            "mean_dy_um": float(mean_dy_um),
            "mean_disp_um": mean_disp_um,
            "um_per_pixel": float(um_per_pixel),
        })

    return pd.DataFrame(records)


def plot_mean_timeseries(df: pd.DataFrame, save_path=None, dpi=300):
    if df.empty:
        return
    plt.figure(figsize=(12, 6))
    plt.plot(df["order"].values, df["mean_disp_um"].values, linewidth=2.0, marker="o", markersize=4)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlabel("Frame (order)", fontsize=14, fontweight="bold")
    plt.ylabel("Displacement magnitude (µm)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
        plt.show()
        plt.close()
    else:
        plt.show()


def run_plot_from_moved_pkl(pkl_moved_dir: str, plot_dir: str, square_size_mm=100.0):
    os.makedirs(plot_dir, exist_ok=True)
    df = compute_displacement_from_pkl_dir(pkl_moved_dir, square_size_mm=square_size_mm)

    df_summary = df[["order", "file", "mean_dx_um", "mean_dy_um", "mean_disp_um", "um_per_pixel"]].copy()
    excel_out = os.path.join(plot_dir, "displacement_summary_noGT.xlsx")
    df_summary.to_excel(excel_out, index=False)

    plot_mean_timeseries(
        df,
        save_path=os.path.join(plot_dir, "mean_displacement_timeseries.png"),
        dpi=300
    )

    print("✅ Plot saved:", plot_dir)
    print("✅ Excel saved:", excel_out)
