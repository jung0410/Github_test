from lib.L1_Image_Conversion import *
from lib.L2_Point_Detection_Conversion import *
from lib.L3_Zhang_Camera_Calibration import *
from lib.L4_Pipeline_Utilities import *
from lib.L5_Visualization_Utilities import *

def run_pipeline(
    calib_image_dir: str,
    moved_image_dir: str,
    work_dir: str,
    selection=None,
    image_size=(4000, 3000),
    square_size_mm=60.0,
    n_jobs=-1,
):
    setup_logging()

    pkl_calib_dir = os.path.join(work_dir, "pkl_calib")
    calib_param_path = os.path.join(work_dir, "calib_params.pkl")
    undist_dir = os.path.join(work_dir, "undistorted")
    pkl_moved_dir = os.path.join(work_dir, "pkl_moved_plus10")

    # NEW outputs for A5/A6
    homography_param_path = os.path.join(work_dir, "homography_params.pkl")
    a6_out_dir = os.path.join(work_dir, "A6_results")

    print("\n========== [1/6] A1 Marker detection (calibration images) ==========")
    run_A1_marker_detection(calib_image_dir, pkl_calib_dir, selection=selection, n_jobs=n_jobs)

    print("\n========== [2/6] A2 Camera calibration ==========")
    saved_data = load_saved_data_pkl(pkl_calib_dir)
    calib_result = perform_camera_calibration_from_data(
        saved_data,
        image_size=image_size,
        square_size_mm=square_size_mm
    )
    save_calibration_params(calib_result, calib_param_path)

    A = calib_result["camera_matrix"]
    dist = calib_result["dist_coeffs"]
    print("RMSE:", calib_result["RMSE_error"])
    print("Saved:", calib_param_path)

    print("\n========== [3/6] A4 Undistort moved images ==========")
    run_A4_undistort_images(moved_image_dir, undist_dir, A, dist, n_jobs=n_jobs)

    print("\n========== [4/6] A1+10 Marker detection (undistorted moved images) ==========")
    run_A1_plus10_marker_detection(undist_dir, pkl_moved_dir, n_jobs=n_jobs)

    print("\n========== [5/6] A5 Homography estimation (baseline + apply_homography) ==========")
    run_A5_homography_from_moved_pkl(
        pkl_moved_dir=pkl_moved_dir,
        out_homography_pkl=homography_param_path,
        square_size_mm=square_size_mm,
        grid_rows=9,
        grid_cols=12,
        baseline_index=0,
        filter_thr=50.0,
    )

    print("\n========== [6/6] A6 Displacement measurement (alignment + displacement) ==========")
    run_A6_displacement_and_plots(
        pkl_moved_dir=pkl_moved_dir,
        homography_pkl_path=homography_param_path,
        out_dir=a6_out_dir,
        square_size_mm=square_size_mm,
        y_threshold_align=50.0,
        x_threshold_align=50.0,
    )

    print("\n✅ Finished.")
    print(" - PKL (calib):", pkl_calib_dir)
    print(" - Calib:", calib_param_path)
    print(" - Undistorted:", undist_dir)
    print(" - PKL (moved+10):", pkl_moved_dir)
    print(" - A5 homography:", homography_param_path)
    print(" - A6 output:", a6_out_dir)



if __name__ == "__main__":
    import argparse
    import multiprocessing

    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Run full CV displacement pipeline (A1-A6)")
    parser.add_argument("--calib_image_dir", required=True)
    parser.add_argument("--moved_image_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--image_width", type=int, default=4000)
    parser.add_argument("--image_height", type=int, default=3000)
    parser.add_argument("--square_size_mm", type=float, default=60.0)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--selection", default=None)

    args = parser.parse_args()

    run_pipeline(
        calib_image_dir=args.calib_image_dir,
        moved_image_dir=args.moved_image_dir,
        work_dir=args.work_dir,
        selection=args.selection,
        image_size=(args.image_width, args.image_height),
        square_size_mm=args.square_size_mm,
        n_jobs=args.n_jobs,
    )

### cmd 들어가서 아래 파일 입력

###cd /d C:\Users\Win\PycharmProjects\Final_algorithm
# "C:\Users\Win\OneDrive\Research_1_Computer vision\Final_algorithm\Scripts\python.exe" main.py ^
#   --calib_image_dir "D:\demo\all_process_for_one\Input\20_images" ^
#   --moved_image_dir "D:\demo\all_process_for_one\Input\Displacement_image" ^
#   --work_dir "D:\demo\all_process_for_one\output" ^
#   --image_width 4000 ^
#   --image_height 3000 ^
#   --square_size_mm 60 ^
#   --n_jobs -1