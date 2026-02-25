import os
import re
import cv2
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
## in English
from typing import Any, List, Optional, Tuple, Dict
from lib.L1_Image_Conversion import how_much_rect, undistort_image_remap
from lib.L2_Point_Detection_Conversion import find_Centroid_12_9, find_ZernikeEdge_curvefit_12_9, find_cv_12_9, \
    find_Square_12_9_Centroid, iterative_filtering_12_9, find_Centroid_12_9_first_image_plus_ten, \
    compute_homography_noraml, Makeobjp, apply_homography, align_points_to_grid
from lib.L3_Zhang_Camera_Calibration import calibrate_zhang_then_lm

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
    for y in range(rows, 0, -1):
        for x in  range(1, cols + 1):
            objp[idx] = [x, y, 0]
            idx += 1
    objp *= float(square_size_mm)

    for filename, Data in saved_data:
        if Data is None:
            continue

        Data = np.asarray(Data, dtype=np.float32).reshape(-1, 2)
        Data = iterative_filtering_12_9(Data, 50)
        Data = np.asarray(Data, dtype=np.float32).reshape(-1, 2)
        objpoints.append(objp)

        imgpoints.append(Data)
        filenames.append(filename)

    if not objpoints:
        raise RuntimeError("No valid pkl data for calibration")

    K, dist, rvecs, tvecs, rmse, K_init = calibrate_zhang_then_lm(
        image_pts_list=imgpoints,
        grid_size=(12, 9),
        spacing=square_size_mm,
    )

    return {
        "ret": "calibration result",
        "camera_matrix": K,
        "camera_matrix_init": K_init,
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
        "K_init": calib_result["camera_matrix_init"]
    }
    save_pickle(out_path, payload)

# ============================================================
# [A3] Front-Reprojection
# ============================================================

def _A3_build_objp(square_size_mm: float, grid_rows: int = 9, grid_cols: int = 12) -> np.ndarray:
    ## in English
    objp = np.zeros((grid_rows * grid_cols, 3), np.float64)
    k = 0
    for y in range(1, grid_rows + 1):
        for x in range(1, grid_cols + 1):
            objp[k] = [x, y, 0.0]
            k += 1
    objp *= float(square_size_mm)
    return objp


def _A3_load_calib_params(calib_param_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    ## in English
    payload = load_pickle(calib_param_path)
    K = np.asarray(payload["camera_matrix"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(payload["dist_coeffs"], dtype=np.float64).reshape(-1)

    K_init = payload.get("K_init", None)
    if K_init is not None:
        K_init = np.asarray(K_init, dtype=np.float64).reshape(3, 3)

    return K, dist, K_init


def _A3_solve_pnp_rmse(
    image_pts_px: np.ndarray,
    objp: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    ## in English
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
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return float("nan"), None, None

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)

    diff = image_pts_px.astype(np.float64) - proj
    rmse = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    return rmse, rvec, tvec


def _A3_plot_front_rmse_bar(init_rmse: float, refined_rmse: float, save_path: str, dpi: int = 300) -> None:
    ## in English
    plt.figure(figsize=(4, 3))
    labels = ["INIT (dist=0)", "REFINED"]
    values = [init_rmse, refined_rmse]
    plt.bar(labels, values)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.ylabel("Reprojection error (px)", fontsize=20, fontweight="bold")
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def run_A3_reprojection_check(
    pkl_calib_dir: str,
    calib_param_path: str,
    out_dir: str,
    square_size_mm: float,
    grid_rows: int = 9,
    grid_cols: int = 12,
    front_index: int = 0,
) -> Dict[str, Any]:
    """
    A3:
    1) Select a front-view point set from pkl_calib_dir (e.g., first file).
    2) Compute Front_RMSE for INIT(dist=0) and REFINED(dist=calibrated).
    3) PASS if INIT_RMSE <= REFINED_RMSE else NOT_PASS.
    4) Save results (CSV/XLSX/TXT + simple bar plot).
    """
    ## in English
    os.makedirs(out_dir, exist_ok=True)

    ## in English
    pkl_files = [f for f in os.listdir(pkl_calib_dir) if f.lower().endswith(".pkl")]
    pkl_files.sort(key=natural_key)
    if not pkl_files:
        raise RuntimeError(f"[A3] No PKL found in: {pkl_calib_dir}")

    if front_index < 0 or front_index >= len(pkl_files):
        raise ValueError(f"[A3] front_index out of range: {front_index}")

    front_pkl = os.path.join(pkl_calib_dir, pkl_files[front_index])
    front_pts = load_pickle(front_pkl)
    front_pts = np.asarray(front_pts, dtype=np.float64).reshape(-1, 2)

    expected_n = grid_rows * grid_cols
    if front_pts.shape != (expected_n, 2):
        raise ValueError(f"[A3] Expected ({expected_n},2), got {front_pts.shape} from {front_pkl}")

    ## in English
    K_refined, dist_refined,K_init = _A3_load_calib_params(calib_param_path)

    ## in English
    dist_init = np.zeros(4, dtype=np.float64)

    ## in English
    objp = _A3_build_objp(square_size_mm=square_size_mm, grid_rows=grid_rows, grid_cols=grid_cols)

    init_rmse, _, _ = _A3_solve_pnp_rmse(front_pts, objp, K_init, dist_init)
    refined_rmse, _, _ = _A3_solve_pnp_rmse(front_pts, objp, K_refined, dist_refined)

    verdict = "PASS" if (np.isfinite(init_rmse) and np.isfinite(refined_rmse) and  refined_rmse <= init_rmse ) else "NOT_PASS"

    ## in English
    df = pd.DataFrame([
        {"tag": "INIT (dist=0)", "Front_RMSE": float(init_rmse), "front_pkl": os.path.basename(front_pkl)},
        {"tag": "REFINED", "Front_RMSE": float(refined_rmse), "front_pkl": os.path.basename(front_pkl)},
    ])

    out_csv = os.path.join(out_dir, "A3_front_rmse.csv")
    df.to_csv(out_csv, index=False)

    out_xlsx = os.path.join(out_dir, "A3_front_rmse.xlsx")
    df.to_excel(out_xlsx, index=False)

    out_txt = os.path.join(out_dir, "A3_verdict.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"VERDICT: {verdict}\n")
        f.write(f"INIT_Front_RMSE: {init_rmse}\n")
        f.write(f"REFINED_Front_RMSE: {refined_rmse}\n")
        f.write(f"FRONT_PKL: {front_pkl}\n")

    out_plot = os.path.join(out_dir, "A3_front_rmse_bar.png")
    _A3_plot_front_rmse_bar(float(init_rmse), float(refined_rmse), out_plot, dpi=300)

    print(f"✅ [A3] FRONT PKL       : {front_pkl}")
    print(f"✅ [A3] INIT Front_RMSE : {init_rmse}")
    print(f"✅ [A3] REF Front_RMSE  : {refined_rmse}")
    print(f"✅ [A3] VERDICT         : {verdict}")
    print(f"✅ [A3] Saved           : {out_dir}")

    return {
        "verdict": verdict,
        "init_rmse": float(init_rmse),
        "refined_rmse": float(refined_rmse),
        "front_pkl": front_pkl,
        "csv": out_csv,
        "xlsx": out_xlsx,
        "txt": out_txt,
        "plot": out_plot,
    }



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


def estimate_homography_from_baseline(
    pkl_dir: str,
    square_size_mm: float = 100.0,
    grid_rows: int = 9,
    grid_cols: int = 12,
    baseline_index: int = 0,
    filter_thr: float = 50.0,
):
    """
    A5: Estimate homography and apply it to the baseline (before grid alignment).
    """
    pkl_files = [f for f in os.listdir(pkl_dir) if f.lower().endswith(".pkl")]
    pkl_files.sort(key=natural_key)
    if not pkl_files:
        raise RuntimeError(f"No PKL found in: {pkl_dir}")

    base_file = pkl_files[baseline_index]
    with open(os.path.join(pkl_dir, base_file), "rb") as f:
        payload0 = pickle.load(f)

    data0 = ensure_2col(payload0.get("data"))
    ref0  = ensure_2col(payload0.get("ref_data"))
    if data0 is None or ref0 is None:
        raise RuntimeError(f"Baseline invalid: {base_file}")

    data0 = np.asarray(data0, dtype=float)
    ref0  = np.asarray(ref0, dtype=float)

    # filtering
    data0_f = iterative_filtering_12_9(data0, filter_thr)

    # initial scale (pre-homography)
    avg_dist_px0, _ = how_much_rect(pd.DataFrame(data0_f), grid_rows, grid_cols)
    um_per_pixel_pre = (6.0 * 1000.0) / avg_dist_px0

    # object points (mm)
    obj_points_mm = Makeobjp(square_size_mm, grid_rows, grid_cols)
    obj_points_mm_xy = np.asarray(obj_points_mm)[:, :2]

    # homography
    H = compute_homography_noraml(data0_f, obj_points_mm_xy)

    # ✅ A5 ends here: apply homography to baseline
    data0_h = apply_homography(H, np.array(data0_f))
    data0_h = np.asarray(data0_h, dtype=float)

    return {
        "H": np.asarray(H, dtype=float),
        "ref0": ref0,
        "data0_f": data0_f,
        "data0_h": data0_h,           # 🔑 A6 입력
        "um_per_pixel_pre": float(um_per_pixel_pre),
        "baseline_file": base_file,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "square_size_mm": float(square_size_mm),
        "filter_thr": float(filter_thr),
    }

def compute_displacement_using_homography(
    pkl_dir: str,
    homography_pkl_path: str,
    y_threshold_align: float = 50,
    x_threshold_align: float = 50,
):
    """
    A6: Deformation measurement after homography projection.
    """
    with open(homography_pkl_path, "rb") as f:
        H_payload = pickle.load(f)

    H = np.asarray(H_payload["H"], dtype=float)
    ref0 = np.asarray(H_payload["ref0"], dtype=float)
    data0_h = np.asarray(H_payload["data0_h"], dtype=float)

    grid_rows = int(H_payload["grid_rows"])
    grid_cols = int(H_payload["grid_cols"])

    # ---- A6 starts here ----
    data0_h = align_points_to_grid(
        data0_h, grid_rows, grid_cols,
        y_threshold_align, x_threshold_align
    )
    data0_h = np.asarray(data0_h, dtype=float)

    # scale update after homography + alignment
    avg_dist_px0_h, _ = how_much_rect(pd.DataFrame(data0_h), grid_rows, grid_cols)
    um_per_pixel = (6.0 * 1000.0) / avg_dist_px0_h

    pkl_files = [f for f in os.listdir(pkl_dir) if f.lower().endswith(".pkl")]
    pkl_files.sort(key=natural_key)

    records = []
    for order, fname in enumerate(pkl_files):
        with open(os.path.join(pkl_dir, fname), "rb") as f:
            payload = pickle.load(f)

        data = ensure_2col(payload.get("data"))
        ref  = ensure_2col(payload.get("ref_data"))
        if data is None or ref is None:
            continue

        data = np.asarray(data, dtype=float)
        ref  = np.asarray(ref, dtype=float)

        move = (ref - ref0).mean(axis=0)
        data_corr = data - move

        data_h = apply_homography(H, np.array(data_corr))
        data_h = align_points_to_grid(
            data_h, grid_rows, grid_cols,
            y_threshold_align, x_threshold_align
        )
        data_h = np.asarray(data_h, dtype=float)

        disp = data_h - data0_h
        disp_um = disp * um_per_pixel

        mean_dx_um, mean_dy_um = np.nanmean(disp_um, axis=0)
        mean_disp_um = float(np.hypot(mean_dx_um, mean_dy_um))

        records.append({
            "order": order,
            "file": fname,
            "mean_dx_um": float(mean_dx_um),
            "mean_dy_um": float(mean_dy_um),
            "mean_disp_um": mean_disp_um,
            "um_per_pixel": float(um_per_pixel),
        })

    return pd.DataFrame(records)

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

import os
import pickle

def run_A5_homography_from_moved_pkl(
    pkl_moved_dir: str,
    out_homography_pkl: str,
    square_size_mm: float = 60.0,
    grid_rows: int = 9,
    grid_cols: int = 12,
    baseline_index: int = 0,
    filter_thr: float = 50.0,
):
    """
    Pipeline Step A5:
    Estimate homography H from the baseline PKL and apply it to baseline (apply_homography stage).
    Saves homography_params.pkl used by A6.
    """
    payload = estimate_homography_from_baseline(
        pkl_dir=pkl_moved_dir,
        square_size_mm=square_size_mm,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        baseline_index=baseline_index,
        filter_thr=filter_thr,
    )

    os.makedirs(os.path.dirname(out_homography_pkl), exist_ok=True)
    with open(out_homography_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ [A5] Saved homography params: {out_homography_pkl}")
    return out_homography_pkl

import os
def plot_mean_timeseries(df: pd.DataFrame, save_path=None, dpi=300):
    ## in English
    if df is None or df.empty:
        return

    plt.figure(figsize=(12, 6))
    plt.plot(
        df["order"].values,
        df["mean_disp_um"].values,
        linewidth=2.0,
        marker="o",
        markersize=4,
    )
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlabel("Frame (order)", fontsize=14, fontweight="bold")
    plt.ylabel("Displacement magnitude (µm)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    else:
        plt.show()


def run_A6_displacement_and_plots(
    pkl_moved_dir: str,
    homography_pkl_path: str,
    out_dir: str,
    square_size_mm: float = 60.0,
    y_threshold_align: float = 50.0,
    x_threshold_align: float = 50.0,
):
    """
    Pipeline Step A6:
    Compute displacement using A5 homography results (A6 starts from alignment),
    then save CSV and plots.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) displacement table
    df = compute_displacement_using_homography(
        pkl_dir=pkl_moved_dir,
        homography_pkl_path=homography_pkl_path,
        y_threshold_align=y_threshold_align,
        x_threshold_align=x_threshold_align,
    )
    out_csv = os.path.join(out_dir, "displacement.csv")
    df.to_csv(out_csv, index=False)
    print(f"✅ [A6] Saved displacement table: {out_csv}")

    summary_cols = [c for c in ["order", "file", "mean_dx_um", "mean_dy_um", "mean_disp_um", "um_per_pixel"] if
                    c in df.columns]
    if summary_cols:
        out_xlsx = os.path.join(out_dir, "displacement_summary.xlsx")
        df[summary_cols].to_excel(out_xlsx, index=False)
        print(f"✅ [A6] Saved summary Excel: {out_xlsx}")

    ## in English
    plot_path = os.path.join(out_dir, "mean_displacement_timeseries.png")
    plot_mean_timeseries(df, save_path=plot_path, dpi=300)
    print(f"✅ [A6] Saved plot: {plot_path}")

    return out_csv
