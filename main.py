from lib.L4_Pipeline_utility import *

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

    print("\n========== [1/5] A1 Marker detection (calibration images) ==========")
    run_A1_marker_detection(calib_image_dir, pkl_calib_dir, selection=selection, n_jobs=n_jobs)

    print("\n========== [2/5] A2 Camera calibration ==========")
    saved_data = load_saved_data_pkl(pkl_calib_dir)
    calib_result = perform_camera_calibration_from_data(saved_data, image_size=image_size, square_size_mm=square_size_mm)
    save_calibration_params(calib_result, calib_param_path)

    A = calib_result["camera_matrix"]
    dist = calib_result["dist_coeffs"]
    print("A:", A)
    print("dist:", dist)
    print("RMSE:", calib_result["RMSE_error"])
    print("Saved:", calib_param_path)

    print("\n========== [3/5] A4 Undistort moved images ==========")
    run_A4_undistort_images(moved_image_dir, undist_dir, A, dist, n_jobs=n_jobs)

    print("\n========== [4/5] A1+10 Marker detection (undistorted moved images) ==========")
    run_A1_plus10_marker_detection(undist_dir, pkl_moved_dir, n_jobs=n_jobs)

    print("\n========== [5/5] Plot displacement from moved PKL (no GT) ==========")
    plot_dir = os.path.join(work_dir, "plots_noGT")
    run_plot_from_moved_pkl(pkl_moved_dir, plot_dir, square_size_mm=square_size_mm)

    print("\n✅ Finished.")
    print(" - PKL (calib):", pkl_calib_dir)
    print(" - Calib:", calib_param_path)
    print(" - Undistorted:", undist_dir)
    print(" - PKL (moved+10):", pkl_moved_dir)
    print(" - Plots:", plot_dir)


if __name__ == "__main__":
    multiprocessing.freeze_support()

    calib_image_dir = r"D:\demo\all_process_for_one\Input\20_images"
    moved_image_dir = r"D:\demo\all_process_for_one\Input\Displacement_image"
    work_dir = r"D:\demo\all_process_for_one\output"

    run_pipeline(
        calib_image_dir=calib_image_dir,
        moved_image_dir=moved_image_dir,
        work_dir=work_dir,
        selection=None,
        image_size=(4000, 3000),
        square_size_mm=60.0,
        n_jobs=-1
    )
