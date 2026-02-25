# main.py
from __future__ import annotations


import argparse

from dataclasses import dataclass

import numpy as np
from tqdm.contrib.concurrent import process_map

# ----------------------------
# Your project imports
# ----------------------------
# Mode 1,2 uses:
from lib.L1_Image_Conversion import *
from lib.L2_Point_Detection_Conversion import *
from lib.L3_Zhang_Camera_Calibration import *
from lib.L4_Pipeline_Utilities import *
from lib.L5_Visualization_Utilities import *


# =========================
# Utilities
# =========================
_IMG_EXTS = {".jpg", ".jpeg", ".png"}


def setup_logging(level: int) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        level=level,
    )


def natural_key(path_str: str) -> List[Any]:
    name = Path(path_str).name
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_images(dir_path: str | Path) -> List[str]:
    p = Path(dir_path)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")
    paths = [str(p / f) for f in os.listdir(p) if (p / f).suffix.lower() in _IMG_EXTS]
    paths.sort(key=natural_key)
    return paths


def get_all_image_paths(root_dir: str) -> Dict[str, List[str]]:
    """Recursively collect image paths by folder."""
    image_paths_by_folder: Dict[str, List[str]] = {}
    for root, _, files in os.walk(root_dir):
        image_list = [
            os.path.join(root, f)
            for f in files
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if image_list:
            image_list.sort(key=natural_key)
            image_paths_by_folder[root] = image_list
    return image_paths_by_folder


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# =========================
# Mode 1: simple (your first code)
# =========================
def mode1_process_single_image(img_path: str, out_dir: str, selection: Optional[str]) -> None:
    base = Path(img_path).stem
    out_path = str(Path(out_dir) / f"GT_{base}.pkl")
    if os.path.exists(out_path):
        return

    gray = read_gray(img_path)

    # ---- your selection routing (keep same behavior) ----
    if selection is None or selection == "Circle_Centroid":
        Data_df, avg = find_Centroid_12_9(gray)
    elif selection == "Circle_ZernikeEdge":
        Data_df, avg = find_ZernikeEdge_curvefit_12_9(gray)
    elif selection == "cv":
        Data_df = find_cv_12_9(gray)
    elif selection == "Square_Centroid":
        Data_df, avg = find_Square_12_9_Centroid(gray)
    else:
        raise ValueError(f"Unknown selection: {selection}")

    if Data_df is None:
        logging.warning(f"[Mode1] No data: {Path(img_path).name}")
        return

    data = np.asarray(Data_df).reshape(-1, 2)

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def mode1_run_folder(img_paths: List[str], out_dir: str, selection: Optional[str], n_jobs: int, chunksize: int) -> None:
    if n_jobs is None or n_jobs <= 0:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)

    func = partial(mode1_process_single_image, out_dir=out_dir, selection=selection)

    process_map(
        func,
        img_paths,
        max_workers=n_jobs,
        chunksize=chunksize,
        desc=f"[Mode1] {Path(out_dir).name}",
    )


# =========================
# Mode 2: bootstrap (your second code)
# =========================
def save_payload(out_path: str, data: np.ndarray, ref_data: np.ndarray, black, circular) -> None:
    payload = {"data": data, "ref_data": ref_data, "black_count": black, "circularity_list": circular}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def mode2_process_first_image(img_path: str, out_dir: str) -> float:
    base = Path(img_path).stem
    out_path = str(Path(out_dir) / f"GT_{base}.pkl")

    gray = read_gray(img_path)

    # Your project function:
    Data_df, Ref_df, avg, black_count_list, circularity_list = find_Centroid_12_9_first_image_plus_ten_count(gray)

    if Data_df is None or Ref_df is None:
        raise RuntimeError(f"[Mode2] First image failed: {Path(img_path).name}")

    data = np.asarray(Data_df, dtype=float).reshape(-1, 2)
    ref = np.asarray(Ref_df, dtype=float).reshape(-1, 2)

    save_payload(out_path, data, ref, black_count_list, circularity_list)
    return float(avg)


def mode2_process_single_image(img_path: str, out_dir: str, avg: float) -> None:
    base = Path(img_path).stem
    out_path = str(Path(out_dir) / f"GT_{base}.pkl")
    if os.path.exists(out_path):
        return

    try:
        gray = read_gray(img_path)
        Data_df, Ref_df, _, black_pixel, circularity_list = find_Centroid_12_9_first_image_plus_ten_count(gray, avg)

        if Data_df is None or Ref_df is None:
            logging.warning(f"[Mode2] No data: {Path(img_path).name}")
            return

        data = np.asarray(Data_df, dtype=float).reshape(-1, 2)
        ref = np.asarray(Ref_df, dtype=float).reshape(-1, 2)
        save_payload(out_path, data, ref, black_pixel, circularity_list)

    except Exception as e:
        logging.exception(f"[Mode2] Fail {Path(img_path).name}: {e}")


def mode2_run_folder(img_paths: List[str], out_dir: str, n_jobs: int, chunksize: int) -> None:
    if not img_paths:
        return

    os.makedirs(out_dir, exist_ok=True)

    # first image bootstrap
    avg = mode2_process_first_image(img_paths[0], out_dir)

    # rest parallel
    rest = img_paths[1:]
    if not rest:
        return

    if n_jobs is None or n_jobs <= 0:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)

    func = partial(mode2_process_single_image, out_dir=out_dir, avg=avg)
    process_map(
        func,
        rest,
        max_workers=n_jobs,
        chunksize=chunksize,
        desc=f"[Mode2] {Path(out_dir).name}",
    )


# =========================
# Top-level runner
# =========================
@dataclass
class RunConfig:
    input_dir: str
    output_dir: str
    mode: str                  # "simple" or "bootstrap"
    selection: Optional[str]   # only for mode1
    n_jobs: int
    chunksize: int
    folder_workers: int        # folder-level parallelism


def run(cfg: RunConfig) -> None:
    folders = get_all_image_paths(cfg.input_dir)
    if not folders:
        logging.warning(f"No images found under {cfg.input_dir}")
        return

    os.makedirs(cfg.output_dir, exist_ok=True)

    def _process_one_folder(folder_path: str) -> str:
        folder_name = Path(folder_path).name
        out_dir = str(Path(cfg.output_dir) / folder_name)
        imgs = folders[folder_path]

        if cfg.mode == "simple":
            mode1_run_folder(imgs, out_dir, cfg.selection, cfg.n_jobs, cfg.chunksize)
        elif cfg.mode == "bootstrap":
            mode2_run_folder(imgs, out_dir, cfg.n_jobs, cfg.chunksize)
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")

        return f"[DONE] {folder_name}"

    # Folder-level parallelism (avoid nested parallelism explosion)
    # If you set folder_workers>1, set per-folder n_jobs=1 for disk-heavy jobs.
    folder_workers = max(1, int(cfg.folder_workers))

    if folder_workers == 1:
        for fp in sorted(folders.keys()):
            print(_process_one_folder(fp))
    else:
        # Avoid nested parallelism: strongly recommend cfg.n_jobs=1 here.
        results = process_map(
            _process_one_folder,
            sorted(folders.keys()),
            max_workers=folder_workers,
            chunksize=1,
            desc="Processing folders",
        )
        for r in results:
            print(r)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Root input directory (recursive).")
    p.add_argument("--output", required=True, help="Root output directory.")
    p.add_argument("--mode", choices=["simple", "bootstrap"], default="simple")
    p.add_argument("--selection", default=None, help="Mode1 detector selection (e.g., Circle_Centroid, cv, ...)")
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--chunksize", type=int, default=30)
    p.add_argument("--folder_workers", type=int, default=1)
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    setup_logging(getattr(logging, args.log_level))

    cfg = RunConfig(
        input_dir=args.input,
        output_dir=args.output,
        mode=args.mode,
        selection=args.selection,
        n_jobs=args.n_jobs,
        chunksize=args.chunksize,
        folder_workers=args.folder_workers,
    )
    run(cfg)
