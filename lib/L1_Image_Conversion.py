import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


import matplotlib.pyplot as plt
import os




### Load a single image file
def process_single_image(directory_path):
    # Read an image from the given path (kept as-is for backward compatibility).
    image = cv2.imread(directory_path)
    return image


### Load multiple images from a directory
### Note: This function does not preserve filenames reliably; use with caution.
def process_images_in_directory(directory_path):
    image_files = [f for f in os.listdir(directory_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    images = []

    for file in image_files:
        file_path = os.path.join(directory_path, file)
        image = cv2.imread(file_path)
        images.append(image)

    return images


def calculate_distance_Y_from_data(df):
    # Compute Y-direction differences for a fixed set of index pairs and append their mean.
    index_pairs = [(30, 0), (31, 1), (32, 2), (33, 3), (34, 4), (35, 5)]
    distances = []

    for idx1, idx2 in index_pairs:
        y1 = df.loc[idx1, "Y"]
        y2 = df.loc[idx2, "Y"]
        distance = y1 - y2
        distances.append(distance)

    average_distance = sum(distances) / len(distances)
    distances.append(average_distance)

    return distances


def calculate_distance_Y_from_data_2(df):
    # Compute Y-direction differences for a second set of index pairs and append their mean.
    index_pairs = [(18, 12), (19, 13), (20, 14), (21, 15), (22, 16), (23, 17)]
    distances = []

    for idx1, idx2 in index_pairs:
        y1 = df.loc[idx1, "Y"]
        y2 = df.loc[idx2, "Y"]
        distance = y1 - y2
        distances.append(distance)

    average_distance = sum(distances) / len(distances)
    distances.append(average_distance)

    return distances


def calculate_distance_X_from_data(df):
    # Compute X-direction differences for a fixed set of index pairs and append their mean.
    index_pairs = [(5, 0), (11, 6), (17, 12), (23, 18), (29, 24), (35, 30)]
    distances = []

    for idx1, idx2 in index_pairs:
        X1 = df.loc[idx1, "X"]
        X2 = df.loc[idx2, "X"]
        distance = X1 - X2
        distances.append(distance)

    average_distance = sum(distances) / len(distances)
    distances.append(average_distance)

    return distances

def undistort_image_remap(image_path,out_path, camera_matrix, dist_coeffs):
    image = cv2.imread(image_path)
    lastfolder = os.path.basename(image_path)
    height, width = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (width, height), 5)
    undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_path, undistorted_image)


def calculate_distance_X_from_data_2(df):
    # Compute X-direction differences for a second set of index pairs and append their mean.
    index_pairs = [(3, 2), (9, 8), (15, 14), (21, 20), (27, 26), (33, 32)]
    distances = []

    for idx1, idx2 in index_pairs:
        X1 = df.loc[idx1, "X"]
        X2 = df.loc[idx2, "X"]
        distance = X1 - X2
        distances.append(distance)

    average_distance = sum(distances) / len(distances)
    distances.append(average_distance)

    return distances


def calculate_euclidean_distance_from_data(df):
    # Compute Euclidean distances for a fixed set of index pairs and append their mean.
    index_pairs = [(30, 0), (31, 1), (32, 2), (33, 3), (34, 4), (35, 5)]
    distances = []

    for idx1, idx2 in index_pairs:
        x1, y1 = df.loc[idx1, "X"], df.loc[idx1, "Y"]
        x2, y2 = df.loc[idx2, "X"], df.loc[idx2, "Y"]
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        distances.append(distance)

    average_distance = sum(distances) / len(distances)
    distances.append(average_distance)

    return distances


def findcorners(image, b, max_sigma, step_sigma):
    # Detect a symmetric circles grid and return ordered corner coordinates.

    obj_points = []  # Placeholder for 3D object points (not used here)
    img_points = []  # Placeholder for 2D image points (not used here)

    rows = 6
    cols = 6

    sigmaX = 50
    found = False
    corners = None

    while sigmaX <= max_sigma:
        # Larger kernel size can improve detection for blurred/low-contrast targets.
        b = 101

        gray1 = cv2.GaussianBlur(gray1, (b, b), sigmaX)

        params = cv2.SimpleBlobDetector_Params()

        # Note: Extremely large values may cause detection instability.
        # This setting was inspired by community discussions on overlapping keypoints.
        params.maxArea = 10000

        detector = cv2.SimpleBlobDetector_create(params)
        ret, corners = cv2.findCirclesGrid(
            gray1, (cols, rows),
            flags=(cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING),
            blobDetector=detector
        )

        if ret:
            found = True

            break

        sigmaX += step_sigma

    if found and corners is not None:
        corners = corners.reshape(-1, 2)

        # Run blob detection again with color filtering to target black circles.
        params.filterByColor = True
        params.blobColor = 0
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(image)

        df = pd.DataFrame(corners, columns=['X', 'Y'])

        # Sort by Y first, then group rows and sort by X within each group.
        df = df.sort_values(by='Y').reset_index(drop=True)
        df['Group'] = df.index // 9  # 9 corresponds to cols for this grid
        df = df.sort_values(by=['Group', 'X']).drop('Group', axis=1).reset_index(drop=True)

        print("Corner detection succeeded.")
        return df, sigmaX
    else:
        # Return a dummy grid (zeros) to avoid downstream failures when detection fails.
        D = 0
        objp = np.zeros((rows * cols, 2), np.float32)
        objp[:, 0] = np.repeat(np.arange(rows - 1, -1, -1), cols) * D
        objp[:, 1] = np.tile(np.arange(cols), rows) * D
        objp = objp.reshape(-1, 2)
        df = pd.DataFrame(objp, columns=['X', 'Y'])

        print("Corner detection failed.")
        return df, sigmaX



### Compute RMSD and maximum absolute error
def calculate_rmsd_and_max_error(measured_df, actual_df):
    # Validate input shapes.
    if measured_df.shape != actual_df.shape:
        raise ValueError("The input data frames must have the same shape.")

    measured_values = measured_df.to_numpy()
    actual_values = actual_df.to_numpy()

    rmsd = np.sqrt(np.mean((measured_values - actual_values) ** 2))
    max_error = np.max(np.abs(measured_values - actual_values))

    return rmsd, max_error


def calculate_rmsd_and_max_error_with_error(error_df):
    # Compute RMSD and error extrema from an error DataFrame.

    error_values = error_df.to_numpy()

    rmsd = np.sqrt(np.mean((error_values) ** 2))
    max_error = np.max(np.abs(error_values))
    high_error = np.max(error_values)
    low_error = np.min(error_values)

    return rmsd, max_error, high_error, low_error


def calculate_rmsd_and_max_error_with_constant(measured_df, n):
    # Build a constant reference DataFrame and compute RMSD/max error against it.

    shape = measured_df.shape
    actual_df = pd.DataFrame(np.full(shape, n))

    measured_values = measured_df.to_numpy()
    actual_values = actual_df.to_numpy()

    rmsd = np.sqrt(np.mean((measured_values - actual_values) ** 2))
    max_error = np.max(np.abs(measured_values - actual_values))

    return rmsd, max_error


def how_much_rect(corners, r, c):
    # Duplicate definition kept as-is; computes mean and std of neighbor distances.

    movement_df = pd.DataFrame()
    corners = pd.DataFrame(corners)
    df = corners
    rows = r
    cols = c
    distances = []

    for i in range(0, df.shape[0] - 1, 1):
        if (i + 1) % cols == 0:
            continue

        movement_x = df.iloc[i + 1, 0] - df.iloc[i, 0]
        movement_y = df.iloc[i + 1, 1] - df.iloc[i, 1]
        euclidean_distance_1 = (np.sqrt(movement_x ** 2 + movement_y ** 2))
        distances.append(euclidean_distance_1)

    for i in range(0, df.shape[0] - 2, 1):
        if (i + cols) >= df.shape[0]:
            break

        movement_x = df.iloc[i + cols, 0] - df.iloc[i, 0]
        movement_y = df.iloc[i + cols, 1] - df.iloc[i, 1]
        euclidean_distance_2 = (np.sqrt(movement_x ** 2 + movement_y ** 2))
        distances.append(euclidean_distance_2)

    average_distance = np.mean(distances)
    std_deviation = np.std(distances)
    movement_df['Distance'] = distances
    return average_distance, std_deviation

def crop_images_from_coordinates(image, coordinates, crop_size=(50, 50)):
    # Crop local patches around each (X, Y) coordinate and return their top-left origins.

    cropped_images = []
    h, w = crop_size

    data_list_c = []
    for index, row in coordinates.iterrows():
        x = int(row['X'])
        y = int(row['Y'])

        # Empirical scale used for cropping extent.
        D = (w / 2.25)

        x_start = int(max(0, x - D))
        y_start = int(max(0, y - D))
        x_end = int(min(image.shape[1], x + D))
        y_end = int(min(image.shape[0], y + D))

        data = {"X": [x_start], "Y": [y_start]}
        df = pd.DataFrame(data)
        data_list_c.append(df)

        cropped_image = image[y_start:y_end, x_start:x_end]
        cropped_images.append(cropped_image)

    Data_x_Y = pd.concat(data_list_c, ignore_index=True)
    return cropped_images, Data_x_Y, h


def pre_process_image(image, sigmaX, avg):
    # Convert to grayscale, smooth, and binarize using Otsu thresholding.

    if len(image.shape) == 2:
        grayscale_image = image
    else:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Padding is currently disabled but returned for downstream compatibility.
    padding_size = 0

    blurred_image = cv2.GaussianBlur(grayscale_image, (int(kernel_size), int(kernel_size)), 0)

    # Binarize using Otsu's method.
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_image, blurred_image, padding_size


# Detect (sub-)pixel corners by extracting connected contours and using moment-based centers.
def detect_corners(image):
    # Note: This currently uses contour centroids rather than the Zernike-based method.

    _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    subpixel_corners = []
    for contour in contours:
        # Compute contour centroid using image moments.
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            subpixel_corners.append((cX, cY))

    return subpixel_corners


# Compute the center of a contour by averaging its coordinates (utility function).
def calculate_contour_center(contour):
    x_values = [point[0][0] for point in contour]
    y_values = [point[0][1] for point in contour]

    center_middle = np.mean(x_values)

    # Note: This returns (x_mean, x_mean) in the original implementation.
    return center_middle, center_middle


def crop_images_from_coordinates_2(image, coordinates, crop_size=(50, 50)):
    # Crop local patches using full width/height offsets.

    cropped_images = []
    h, w = crop_size

    data_list_c = []
    for index, row in coordinates.iterrows():
        x = (row['X'])
        y = (row['Y'])

        x_start = int(max(0, x - w))
        y_start = int(max(0, y - h))
        x_end = int(min(image.shape[1], x + w))
        y_end = int(min(image.shape[0], y + h))

        data = {"X": [x_start], "Y": [y_start]}
        df = pd.DataFrame(data)
        data_list_c.append(df)

        cropped_image = image[y_start:y_end, x_start:x_end]
        cropped_images.append(cropped_image)

    Data_x_Y = pd.concat(data_list_c, ignore_index=True)
    return cropped_images, Data_x_Y


def sort_points(df):
    # Sort exactly 36 points into 6 rows by iteratively selecting the smallest Y values.

    sorted_points = pd.DataFrame()

    for _ in range(6):
        selected_points = df.nsmallest(6, 'Y')
        selected_points = selected_points.sort_values('X')

        sorted_points = pd.concat([sorted_points, selected_points], ignore_index=True)
        df = df.drop(selected_points.index)

    return sorted_points


def crop_images_from_coordinates_3(image, coordinates, crop_size):
    # Crop patches around each coordinate using (h, w) half-extent convention.

    cropped_images = []
    h, w = crop_size

    data_list_c = []
    for index, row in coordinates.iterrows():
        x = int(row['X'])
        y = int(row['Y'])

        x_start = int(max(0, round(x - w)))
        y_start = int(max(0, round(y - h)))
        x_end = int(min(image.shape[1], round(x + w)))
        y_end = int(min(image.shape[0], round(y + h)))

        data = {"X": [x_start], "Y": [y_start]}
        df = pd.DataFrame(data)
        data_list_c.append(df)

        cropped_image = image[y_start:y_end, x_start:x_end]
        cropped_images.append(cropped_image)

    Data_x_Y = pd.concat(data_list_c, ignore_index=True)
    return cropped_images, Data_x_Y


### Corner detection for a 12x9 symmetric circles grid
def findcorners_12_9(image, b, max_sigma, step_sigma):
    obj_points = []  # Placeholder for 3D object points (not used here)
    img_points = []  # Placeholder for 2D image points (not used here)

    rows = 9
    cols = 12

    # Input is assumed to be already grayscale in this pipeline.
    gray1 = image

    found = False
    corners = None

    B = 1000  # Reserved for optional padding (currently unused)

    # Override search range (kept as-is).
    max_sigma = 4000
    sigmaX = 21
    step_sigma = 10

    while sigmaX <= max_sigma:
        # Larger blur can improve detection under noise, but may merge nearby blobs.
        b = 51

        gray1 = cv2.GaussianBlur(gray1, (sigmaX, sigmaX), 0)

        # Configure blob detector for circular targets.
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 500
        params.maxArea = 10000
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByConvexity = True
        params.minConvexity = 0.5
        params.minDistBetweenBlobs = 50

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(image)

        # Increase drawn keypoint size for visualization (debug).
        for kp in keypoints:
            kp.size *= 2

        im_with_keypoints = cv2.drawKeypoints(
            image, keypoints, np.array([]), (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        ret, corners = cv2.findCirclesGrid(
            gray1, (cols, rows),
            flags=(cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING),
            blobDetector=detector
        )

        # Reject detections containing near-duplicate points.
        if ret:
            pts = corners.reshape(-1, 2)

            def count_duplicates(pts, threshold=3):
                count = 0
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        dist = np.linalg.norm(pts[i] - pts[j])
                        if dist < threshold:
                            count += 1
                return count

            dup_count = count_duplicates(pts)
            if dup_count > 0:
                ret = False
                print(f"⚠️ Duplicate points detected: {dup_count}")

        if ret:
            found = True
            break

        sigmaX += step_sigma

    if found and corners is not None:
        corners = corners.reshape(-1, 2)

        # Optionally re-run blob detection targeting black circles.
        params.filterByColor = True
        params.blobColor = 0
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(image)

        df = pd.DataFrame(corners, columns=['X', 'Y'])

        # Sort by Y, then by X within each row-group.
        df = df.sort_values(by='Y').reset_index(drop=True)
        df['Group'] = df.index // 9
        df = df.sort_values(by=['Group', 'X']).drop('Group', axis=1).reset_index(drop=True)

        return df, sigmaX
    else:
        # Return a dummy grid (zeros) to avoid downstream failures when detection fails.
        D = 0
        objp = np.zeros((rows * cols, 2), np.float32)
        objp[:, 0] = np.repeat(np.arange(rows - 1, -1, -1), cols) * D
        objp[:, 1] = np.tile(np.arange(cols), rows) * D
        objp = objp.reshape(-1, 2)
        df = pd.DataFrame(objp, columns=['X', 'Y'])

        print("Corner detection failed.")
        return df, sigmaX


def findcorners_12_9_SigmaX(image, sigmaX):
    # Detect blob centers using a sigmaX derived from a previous frame.

    rows = 9
    cols = 12

    gray1 = image

    found = False
    corners = None

    B = 1000  # Reserved for optional padding (currently unused)

    max_sigma = 4000
    sigmaX = float(sigmaX * 2 - 1)
    step_sigma = 10

    while sigmaX <= max_sigma:
        b = 51

        gray1 = cv2.GaussianBlur(gray1, (int(sigmaX), int(sigmaX)), 0)

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 1000
        params.maxArea = 90000
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByConvexity = True
        params.minConvexity = 0.5
        params.minDistBetweenBlobs = 50
        params.filterByInertia = True
        params.minInertiaRatio = 0.15

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(gray1)

        im_with_keypoints = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        max_dist = 4000

        def filter_by_nearest_distance_vectorized(keypoints, scale=2.0):
            # Filter keypoints by comparing their nearest-neighbor distance to the global mean.
            if len(keypoints) == 0:
                return []

            points = np.array([kp.pt for kp in keypoints])

            diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)

            np.fill_diagonal(dists, np.inf)

            min_dists = np.min(dists, axis=1)
            mean_dist = np.mean(min_dists)

            keep_indices = np.where(min_dists < scale * mean_dist)[0]
            keypoints_filtered = [keypoints[i] for i in keep_indices]

            return keypoints_filtered

        keypoints = filter_by_nearest_distance_vectorized(keypoints, scale=1.5)

        def draw_filled_keypoints(image, keypoints, color=(0, 255, 0)):
            # Draw filled circles at keypoint locations for visualization.
            if len(image.shape) == 2 or image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            img_copy = image.copy()

            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                radius = int(kp.size / 2)
                cv2.circle(img_copy, (x, y), radius, color, thickness=-1)

            return img_copy

        im_with_keypoints = draw_filled_keypoints(image, keypoints)

        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = fr"D:\Real_test_2025_08\std/keypoints_filled_{timestamp}.png"

        scale_percent = 10
        width = int(im_with_keypoints.shape[1] * scale_percent / 100)
        height = int(im_with_keypoints.shape[0] * scale_percent / 100)
        resized_img = cv2.resize(im_with_keypoints, (width, height), interpolation=cv2.INTER_AREA)

        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)

        if len(points) > 0:
            found = True
            break

        sigmaX += step_sigma

    if found and points is not None:
        points = points.reshape(-1, 2)

        params.filterByColor = True
        params.blobColor = 0

        df = pd.DataFrame(points, columns=['X', 'Y'])

        # Sort by Y, then by X within each row-group.
        df = df.sort_values(by='Y').reset_index(drop=True)
        df['Group'] = df.index // 9
        df = df.sort_values(by=['Group', 'X']).drop('Group', axis=1).reset_index(drop=True)

        return df, sigmaX
    else:
        # Return a dummy grid (zeros) to avoid downstream failures when detection fails.
        D = 0
        objp = np.zeros((rows * cols, 2), np.float32)
        objp[:, 0] = np.repeat(np.arange(rows - 1, -1, -1), cols) * D
        objp[:, 1] = np.tile(np.arange(cols), rows) * D
        objp = objp.reshape(-1, 2)
        df = pd.DataFrame(objp, columns=['X', 'Y'])

        print("Corner detection failed.")
        return df, sigmaX


def findcorners_12_9_plus_10(image):
    # Detect two blob-size groups (small grid dots vs. larger reference dots) and return both.

    obj_points = []  # Placeholder for 3D object points (not used here)
    img_points = []  # Placeholder for 2D image points (not used here)

    rows = 9
    cols = 12

    gray1 = image

    found = False
    corners = None

    B = 1000  # Reserved for optional padding (currently unused)

    max_sigma = 4000
    sigmaX = 21
    step_sigma = 10

    while sigmaX <= max_sigma:
        b = 51

        gray1 = cv2.GaussianBlur(gray1, (sigmaX, sigmaX), 0)
        _, gray1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 1500
        params.maxArea = 90000
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByConvexity = True
        params.minConvexity = 0.5
        params.minDistBetweenBlobs = 50
        params.filterByInertia = True
        params.minInertiaRatio = 0.15

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray1)

        # Increase drawn keypoint size for visualization (debug).
        for kp in keypoints:
            kp.size *= 2

        im_with_keypoints = cv2.drawKeypoints(
            image, keypoints, np.array([]), (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)

        if len(points) > 0:
            found = True

        if found:
            pts = points.reshape(-1, 2)

            def count_duplicates(pts, threshold=3):
                count = 0
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        dist = np.linalg.norm(pts[i] - pts[j])
                        if dist < threshold:
                            count += 1
                return count

            dup_count = count_duplicates(pts)
            if dup_count > 0:
                ret = False

            # Split keypoints into two groups based on kp.size using a simple 1D k-means (k=2).
            sizes = np.array([kp.size for kp in keypoints], dtype=np.float32)

            if len(sizes) == 0:
                print("No blobs were detected.")
                small_kps, large_kps = [], []
            else:
                c1, c2 = np.percentile(sizes, [10, 90]).astype(np.float32)
                for _ in range(20):
                    d1 = np.abs(sizes - c1)
                    d2 = np.abs(sizes - c2)
                    labels = (d2 < d1).astype(np.int32)

                    if np.any(labels == 0):
                        c1 = sizes[labels == 0].mean()
                    if np.any(labels == 1):
                        c2 = sizes[labels == 1].mean()

                if c1 < c2:
                    small_mask = (labels == 0)
                    large_mask = (labels == 1)
                else:
                    small_mask = (labels == 1)
                    large_mask = (labels == 0)

                small_kps = [kp for kp, m in zip(keypoints, small_mask) if m]
                large_kps = [kp for kp, m in zip(keypoints, large_mask) if m]

            break

    if found and pts is not None:

        def kps_to_df(kps):
            # Convert a keypoint list into an (X, Y) DataFrame.
            pts = [(kp.pt[0], kp.pt[1]) for kp in kps]
            if len(pts) == 0:
                return pd.DataFrame(columns=['X', 'Y'])
            df_local = pd.DataFrame(pts, columns=['X', 'Y'])
            return df_local

        df_small = kps_to_df(small_kps)

        # If the expected number of small dots is not reached, apply nearest-neighbor filtering.
        if len(df_small) != 108:

            def filter_by_nearest_distance_vectorized(keypoints, scale=2.0):
                if len(keypoints) == 0:
                    return []

                points = np.array([kp.pt for kp in keypoints])

                diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
                dists = np.linalg.norm(diffs, axis=2)
                np.fill_diagonal(dists, np.inf)

                min_dists = np.min(dists, axis=1)
                mean_dist = np.mean(min_dists)

                keep_indices = np.where(min_dists < scale * mean_dist)[0]
                keypoints_filtered = [keypoints[i] for i in keep_indices]

                return keypoints_filtered

            df_small = filter_by_nearest_distance_vectorized(small_kps, scale=1.5)
            df_small = kps_to_df(df_small)

        df_big = kps_to_df(large_kps)

        # Sort the small-dot grid into row-major order.
        df = df_small.sort_values(by='Y').reset_index(drop=True)
        df['Group'] = df.index // 9
        df_small = df.sort_values(by=['Group', 'X']).drop('Group', axis=1).reset_index(drop=True)

        return df_small, df_big, sigmaX
    else:
        # Return a dummy grid (zeros) to avoid downstream failures when detection fails.
        D = 0
        objp = np.zeros((rows * cols, 2), np.float32)
        objp[:, 0] = np.repeat(np.arange(rows - 1, -1, -1), cols) * D
        objp[:, 1] = np.tile(np.arange(cols), rows) * D
        objp = objp.reshape(-1, 2)
        df = pd.DataFrame(objp, columns=['X', 'Y'])

        print("Corner detection failed.")
        return df, sigmaX
