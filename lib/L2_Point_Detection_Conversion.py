import math

from scipy.optimize import least_squares

from lib.L1_Image_Conversion import *

# ============================================================
# IO
# ============================================================

def load_image(path: str):
    """
    Load a single image from a file path.
    Returns: image (numpy array) or None if fail
    """
    return cv2.imread(path)


def load_images_from_dir(directory_path: str, exts=(".jpg", ".jpeg", ".png")):
    """
    Load all images in a directory. (order: filename sort)
    Returns: list of (filename, image)
    """
    files = [f for f in os.listdir(directory_path) if f.lower().endswith(exts)]
    files.sort()
    images = []
    for f in files:
        img = cv2.imread(os.path.join(directory_path, f))
        images.append((f, img))
    return images


# ============================================================
# Distance utilities for a 6x6 grid (index-based)
# ============================================================

def _pairwise_diff(df: pd.DataFrame, index_pairs, col: str):
    vals = []
    for i1, i2 in index_pairs:
        v1 = df.loc[i1, col]
        v2 = df.loc[i2, col]
        vals.append(v1 - v2)
    vals.append(sum(vals) / len(vals) if len(vals) > 0 else np.nan)
    return vals


def calculate_distance_y_6x6_top_bottom(df: pd.DataFrame):
    """
    Y distance between last row (30~35) and first row (0~5).
    Returns: 6 diffs + avg
    """
    pairs = [(30, 0), (31, 1), (32, 2), (33, 3), (34, 4), (35, 5)]
    return _pairwise_diff(df, pairs, "Y")


def calculate_distance_y_6x6_mid(df: pd.DataFrame):
    """
    Y distance between row (18~23) and (12~17).
    Returns: 6 diffs + avg
    """
    pairs = [(18, 12), (19, 13), (20, 14), (21, 15), (22, 16), (23, 17)]
    return _pairwise_diff(df, pairs, "Y")


def calculate_distance_x_6x6_edges(df: pd.DataFrame):
    """
    X distance between right edge and left edge for each row.
    Returns: 6 diffs + avg
    """
    pairs = [(5, 0), (11, 6), (17, 12), (23, 18), (29, 24), (35, 30)]
    return _pairwise_diff(df, pairs, "X")


def calculate_distance_x_6x6_inner(df: pd.DataFrame):
    """
    X distance between near points in each row.
    Returns: 6 diffs + avg
    """
    pairs = [(3, 2), (9, 8), (15, 14), (21, 20), (27, 26), (33, 32)]
    return _pairwise_diff(df, pairs, "X")


def calculate_euclidean_distance_pairs(df: pd.DataFrame, index_pairs):
    """
    Euclidean distance for given index pairs.
    Returns: list of distances + avg
    """
    dists = []
    for i1, i2 in index_pairs:
        x1, y1 = df.loc[i1, "X"], df.loc[i1, "Y"]
        x2, y2 = df.loc[i2, "X"], df.loc[i2, "Y"]
        dists.append(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
    dists.append(np.mean(dists) if len(dists) > 0 else np.nan)
    return dists


# ============================================================
# Grid quality: neighbor distances statistics
# ============================================================

def grid_neighbor_distance_stats(points, rows: int, cols: int):
    """
    Compute neighbor distances on a (rows x cols) grid-like ordered points.
    points: DataFrame or ndarray of shape (rows*cols, 2)
    Returns: (mean, std)
    """
    if isinstance(points, np.ndarray):
        df = pd.DataFrame(points, columns=["X", "Y"])
    else:
        df = points.copy()

    distances = []

    # horizontal neighbors (same row)
    for i in range(rows * cols):
        if (i + 1) % cols == 0:
            continue
        dx = df.iloc[i + 1, 0] - df.iloc[i, 0]
        dy = df.iloc[i + 1, 1] - df.iloc[i, 1]
        distances.append(np.sqrt(dx * dx + dy * dy))

    # vertical neighbors (same col)
    for i in range(rows * cols):
        j = i + cols
        if j >= rows * cols:
            break
        dx = df.iloc[j, 0] - df.iloc[i, 0]
        dy = df.iloc[j, 1] - df.iloc[i, 1]
        distances.append(np.sqrt(dx * dx + dy * dy))

    return float(np.mean(distances)), float(np.std(distances))


def grid_neighbor_distance_df(points, rows: int, cols: int):
    """
    Same as grid_neighbor_distance_stats, but returns a DataFrame of distances.
    """
    if isinstance(points, np.ndarray):
        df = pd.DataFrame(points, columns=["X", "Y"])
    else:
        df = points.copy()

    distances = []

    for i in range(rows * cols):
        if (i + 1) % cols == 0:
            continue
        dx = df.iloc[i + 1, 0] - df.iloc[i, 0]
        dy = df.iloc[i + 1, 1] - df.iloc[i, 1]
        distances.append(np.sqrt(dx * dx + dy * dy))

    for i in range(rows * cols):
        j = i + cols
        if j >= rows * cols:
            break
        dx = df.iloc[j, 0] - df.iloc[i, 0]
        dy = df.iloc[j, 1] - df.iloc[i, 1]
        distances.append(np.sqrt(dx * dx + dy * dy))

    return pd.DataFrame({"Distance": distances})


# ============================================================
# RMSD / Errors
# ============================================================

def calculate_rmsd_and_max_error(measured_df: pd.DataFrame, actual_df: pd.DataFrame):
    if measured_df.shape != actual_df.shape:
        raise ValueError("The input data frames must have the same shape.")

    measured = measured_df.to_numpy()
    actual = actual_df.to_numpy()

    rmsd = float(np.sqrt(np.mean((measured - actual) ** 2)))
    max_error = float(np.max(np.abs(measured - actual)))
    return rmsd, max_error


def calculate_rmsd_and_error_range_from_error_df(error_df: pd.DataFrame):
    """
    error_df itself is error values (measured - actual)
    """
    e = error_df.to_numpy()
    rmsd = float(np.sqrt(np.mean(e ** 2)))
    max_error = float(np.max(np.abs(e)))
    high_error = float(np.max(e))
    low_error = float(np.min(e))
    return rmsd, max_error, high_error, low_error


def calculate_rmsd_and_max_error_with_constant(measured_df: pd.DataFrame, n: float):
    actual = np.full(measured_df.shape, n, dtype=np.float64)
    measured = measured_df.to_numpy()
    rmsd = float(np.sqrt(np.mean((measured - actual) ** 2)))
    max_error = float(np.max(np.abs(measured - actual)))
    return rmsd, max_error


# ============================================================
# Cropping utilities
# ============================================================

def crop_images_from_coordinates(image, coordinates: pd.DataFrame, crop_size=(50, 50), scale_div=2.25):
    """
    Crop small images around (X, Y) centers.

    crop_size: (h, w) nominal. Original code used D = w/2.25 (we keep same behavior).
    Returns:
        cropped_images: list of cropped images
        top_left_df: DataFrame of (x_start, y_start)
        h: crop_size[0]
    """
    cropped_images = []
    h, w = crop_size

    top_left_list = []
    D = w / scale_div

    for _, row in coordinates.iterrows():
        x = int(row["X"])
        y = int(row["Y"])

        x_start = int(max(0, x - D))
        y_start = int(max(0, y - D))
        x_end = int(min(image.shape[1], x + D))
        y_end = int(min(image.shape[0], y + D))

        top_left_list.append({"X": x_start, "Y": y_start})
        cropped_images.append(image[y_start:y_end, x_start:x_end])

    return cropped_images, pd.DataFrame(top_left_list), h


def crop_images_from_coordinates_hw(image, coordinates: pd.DataFrame, crop_hw=(50, 50)):
    """
    Crop using explicit half-width/half-height (w, h) style from original _2 function.
    """
    cropped_images = []
    h, w = crop_hw

    top_left_list = []
    for _, row in coordinates.iterrows():
        x = float(row["X"])
        y = float(row["Y"])

        x_start = int(max(0, x - w))
        y_start = int(max(0, y - h))
        x_end = int(min(image.shape[1], x + w))
        y_end = int(min(image.shape[0], y + h))

        top_left_list.append({"X": x_start, "Y": y_start})
        cropped_images.append(image[y_start:y_end, x_start:x_end])

    return cropped_images, pd.DataFrame(top_left_list)


def crop_images_from_coordinates_round(image, coordinates: pd.DataFrame, crop_hw):
    """
    Crop using rounding behavior from original _3.
    crop_hw: (h, w) half sizes
    """
    cropped_images = []
    h, w = crop_hw

    top_left_list = []
    for _, row in coordinates.iterrows():
        x = int(row["X"])
        y = int(row["Y"])

        x_start = int(max(0, round(x - w)))
        y_start = int(max(0, round(y - h)))
        x_end = int(min(image.shape[1], round(x + w)))
        y_end = int(min(image.shape[0], round(y + h)))

        top_left_list.append({"X": x_start, "Y": y_start})
        cropped_images.append(image[y_start:y_end, x_start:x_end])

    return cropped_images, pd.DataFrame(top_left_list)


# ============================================================
# Pre-processing
# ============================================================

# def pre_process_image(image):
#     """
#     Convert to grayscale, blur, Otsu binarize.
#     Returns: binary_image, blurred_image, padding_size(kept=0 for compatibility)
#     """
#     if image is None:
#         raise ValueError("image is None")
#
#     if len(image.shape) == 2:
#         gray = image
#     else:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     kernel_size = 5
#     if kernel_size % 2 == 0:
#         kernel_size += 1
#
#     blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
#     _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     padding_size = 0
#     return binary, blurred, padding_size
#

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


# ============================================================
# Contours -> centers (simple)
# ============================================================

def detect_contour_centers(binary_image):
    """
    Detect contour centers (pixel-level) from a binary image.
    Returns: list of (cX, cY)
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
    return centers


# ============================================================
# CirclesGrid helpers (ordering)
# ============================================================

def _order_grid_points_df(df: pd.DataFrame, cols_for_group: int):
    """
    Order grid points: sort by Y, then chunk by cols_for_group and sort each chunk by X.
    """
    df = df.sort_values(by="Y").reset_index(drop=True)
    df["Group"] = df.index // cols_for_group
    df = df.sort_values(by=["Group", "X"]).drop(columns=["Group"]).reset_index(drop=True)
    return df


# ============================================================
# findCirclesGrid (6x6)  -- fixed gray initialization
# ============================================================

def find_circles_grid_6x6(image, max_sigma=300, step_sigma=10):
    """
    Try findCirclesGrid on blurred versions while increasing sigma.
    Returns: (df_points, used_sigma)
    """
    rows, cols = 6, 6

    if image is None:
        raise ValueError("image is None")

    # Ensure grayscale for findCirclesGrid
    gray0 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    sigmaX = 50
    found = False
    corners = None

    while sigmaX <= max_sigma:
        b = 101  # keep original behavior
        gray = cv2.GaussianBlur(gray0, (b, b), sigmaX)

        params = cv2.SimpleBlobDetector_Params()
        params.maxArea = 10000  # keep original
        detector = cv2.SimpleBlobDetector_create(params)

        ret, corners = cv2.findCirclesGrid(
            gray, (cols, rows),
            flags=(cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING),
            blobDetector=detector
        )

        if ret:
            found = True
            break

        sigmaX += step_sigma

    if found and corners is not None:
        pts = corners.reshape(-1, 2)
        df = pd.DataFrame(pts, columns=["X", "Y"])
        # NOTE: original used df.index//9 which is suspicious for 6x6,
        # but we keep intent: group by cols (6).
        df = _order_grid_points_df(df, cols_for_group=cols)
        return df, sigmaX

    # fallback: zeros
    objp = np.zeros((rows * cols, 2), np.float32)
    df = pd.DataFrame(objp, columns=["X", "Y"])
    return df, sigmaX


# ============================================================
# findCirclesGrid (12x9) - simplified wrapper
# ============================================================
##findcorners_12_9???find_circles_grid_12x9
def findcorners_12_9(image, sigma_start=21, sigma_max=4000, sigma_step=10):
    """
    12x9 circles grid detection using findCirclesGrid.
    Returns: (df_points, used_sigma)
    """
    rows, cols = 9, 12

    if image is None:
        raise ValueError("image is None")

    gray0 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    sigmaX = sigma_start
    found = False
    corners = None

    while sigmaX <= sigma_max:
        # OpenCV GaussianBlur kernel size must be odd and positive
        k = int(sigmaX)
        if k % 2 == 0:
            k += 1

        gray = cv2.GaussianBlur(gray0, (k, k), 0)

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

        ret, corners = cv2.findCirclesGrid(
            gray, (cols, rows),
            flags=(cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING),
            blobDetector=detector
        )

        if ret:
            # optional: duplicate check
            pts = corners.reshape(-1, 2)
            dup = _count_duplicates(pts, threshold=3)
            if dup == 0:
                found = True
                break

        sigmaX += sigma_step

    if found and corners is not None:
        pts = corners.reshape(-1, 2)
        df = pd.DataFrame(pts, columns=["X", "Y"])
        # original used df.index//9 (wrong for 12 columns). Here we use cols=12.
        df = _order_grid_points_df(df, cols_for_group=cols)
        return df, sigmaX

    df = pd.DataFrame(np.zeros((rows * cols, 2), np.float32), columns=["X", "Y"])
    return df, sigmaX


def _count_duplicates(pts, threshold=3):
    count = 0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) < threshold:
                count += 1
    return count


def residuals(params, x, y):
    # Residual function used by least-squares ellipse fitting.
    def ellipse_equation(params, x, y):
        # Implicit conic (ellipse) model: A x^2 + B x y + C y^2 + D x + E y + 1 = 0
        A, B, C, D, E = params
        return A * x ** 2 + B * x * y + C * y ** 2 + D * x + E * y + 1

    return ellipse_equation(params, x, y)


def subpixel_edge_zernike(image, edge_coordinates, radius=5):
    """
    Estimate subpixel edge coordinates using Zernike moments on a local patch.

    Args:
        image: Input (grayscale/binary) image.
        edge_coordinates: Iterable of integer pixel edge coordinates (x, y).
        radius: Half-size of the local patch (patch size = 2*radius + 1).

    Returns:
        List of (subpixel_x, subpixel_y) coordinates.
    """
    subpixel_coords = []
    for coord in edge_coordinates:
        x, y = coord
        N = (2 * radius + 1)

        # Extract a local patch centered at (x, y). Skip if it goes out-of-bounds.
        local_patch = image[max(y - radius, 0):y + radius + 1, max(x - radius, 0):x + radius + 1]
        if local_patch.shape[0] < (2 * radius + 1) or local_patch.shape[1] < (2 * radius + 1):
            continue

        def zernike_moment(patch, n, m, radius):
            # Compute the Zernike moment Z_nm on a normalized unit disk.
            patch = patch.astype(np.float32)
            y, x = np.indices(patch.shape)
            cx, cy = patch.shape[1] // 2, patch.shape[0] // 2

            # Normalize coordinates to [-1, 1] range w.r.t. the given radius.
            x = (x - cx) / radius
            y = (y - cy) / radius

            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan2(y, x)

            # Use only pixels inside the unit disk.
            mask = r <= 1
            r = r[mask]
            theta = theta[mask]
            patch = patch[mask]

            def zernike_radial_polynomial(n, m, r):
                """
                Compute the Zernike radial polynomial R_n^m(r).

                Args:
                    n: Order (non-negative integer).
                    m: Repetition (integer, |m| <= n and (n - |m|) even).
                    r: Radial coordinate in [0, 1] (vector).

                Returns:
                    R: Radial polynomial values evaluated at r.
                """
                R = np.zeros_like(r)
                for s in range((n - abs(m)) // 2 + 1):
                    # Coefficient from the closed-form Zernike radial polynomial definition.
                    c = ((-1) ** s) * math.factorial(n - s) / (
                        math.factorial(s)
                        * math.factorial((n + abs(m)) // 2 - s)
                        * math.factorial((n - abs(m)) // 2 - s)
                    )
                    R += c * r ** (n - 2 * s)

                return R

            # Evaluate the radial polynomial and construct the complex Zernike basis.
            R = zernike_radial_polynomial(n, m, r)
            V = R * np.exp(-1j * m * theta)

            # Compute the complex Zernike moment with standard normalization.
            Z = (patch * V).sum() * (n + 1) / np.pi

            return Z, V

        # Zernike moments for edge orientation/offset estimation.
        Z00, V00 = zernike_moment(local_patch, 0, 0, N / 2)
        Z10, V10 = zernike_moment(local_patch, 1, 0, N / 2)

        # Z11 is the key term for estimating local edge orientation.
        Z11, V11 = zernike_moment(local_patch, 1, 1, N / 2)
        Z20, V20 = zernike_moment(local_patch, 2, 0, N / 2)

        # Edge orientation derived from the phase of Z11.
        phi = np.arctan2(np.imag(Z11), np.real(Z11))

        # Estimate edge displacement magnitude using Z20 and Z11 (heuristic/derived relation).
        l = Z20 * np.exp(-1j * phi) / Z11
        l = abs(l)

        # Convert estimated displacement to subpixel coordinate update.
        subpixel_y = y + (N / 2) * np.cos(phi) * l
        subpixel_x = x + (N / 2) * np.sin(phi) * l

        subpixel_coords.append((subpixel_x, subpixel_y))

    return subpixel_coords


def find_Centroid_12_9(image):
    i = 0

    cor_1, sigmaX = findcorners_12_9(image, 135, 800, 10)


    # cor_1=iterative_filtering_12_9(cor_1,50)

    cor_1 = pd.DataFrame(cor_1, columns=["X", "Y"])


    S, Stdev = how_much_rect(cor_1, r=12, c=9)
    S = int(S)
    avg = int(0.25 * S)

    # Crop local regions around detected grid corners for centroid estimation.
    cropped_images, Data_x_Y = crop_images_from_coordinates_3(image, cor_1, crop_size=(int(avg), int(avg)))
    if cropped_images is None or len(cropped_images) == 0 or cropped_images[0].size == 0:
        print("No Coners")
        return cor_1

    processed_images = []
    # List to accumulate per-patch centroid results as DataFrames.
    data_list_m = []
    data = []

    for image_c in cropped_images:
        # Preprocess each cropped patch (thresholding/denoising/padding handled internally).
        processed_image, blurred_image, padding_size = pre_process_image(image_c, 79, avg)
        # display_image(image_c)
        black_pixels = np.where(processed_image == 0)
        y_indices, x_indices = black_pixels
        height, width = processed_image.shape[:2]

        # Detect whether any black pixels touch the patch boundary (likely cropping failure / partial blob).
        if (
            np.any(y_indices == 0) or
            np.any(y_indices == height - 1) or
            np.any(x_indices == 0) or
            np.any(x_indices == width - 1)
        ):
            print("Error: Black pixel is out of pixel.")

            # Analyze connected components to handle border-touching blobs.
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                (processed_image == 0).astype(np.uint8), connectivity=8
            )

            border_touching_labels = []
            for label in range(1, num_labels):  # label 0 is background
                x, y, w, h, area = stats[label]
                if x == 0 or y == 0 or (x + w) == width or (y + h) == height:
                    border_touching_labels.append(label)

            # Case 1: only one foreground blob (plus background) touches the border -> abort this patch.
            if len(border_touching_labels) >= 1 and num_labels == 2:
                break

            # Case 2: multiple blobs exist; remove the one touching the bottom border.
            elif len(border_touching_labels) == 1 and num_labels == 3:
                for label in border_touching_labels:
                    x, y, w, h, area = stats[label]
                    if y + h >= height - 1:
                        processed_image[labels == label] = 255  # remove by setting to white

                # Recompute black pixels after cleanup.
                height, width = processed_image.shape[:2]
                black_pixels = np.where(processed_image == 0)
                y_indices, x_indices = black_pixels

        # Compute centroid as the mean position of black pixels (in patch coordinates).
        distances_x = []
        distances_y = []
        for y, x in zip(*black_pixels):
            distances_x.append(x)
            distances_y.append(y)

        average_x = sum(distances_x) / len(distances_x) if distances_x else None
        average_y = sum(distances_y) / len(distances_y) if distances_y else None

        # Undo padding offset to map back to the cropped patch coordinate frame.
        data = {"X": [average_x - padding_size], "Y": [average_y - padding_size]}
        df = pd.DataFrame(data)
        data_list_m.append(df)

    # Concatenate per-patch centroids and add them to the original crop origin coordinates.
    middle_df = pd.concat(data_list_m, ignore_index=True)

    Data = Data_x_Y + middle_df
    return Data, avg


def find_Centroid_12_9_first_image(image):
    i = 0

    cor_1, sigmaX = findcorners_12_9(image, 135, 800, 10)

    cor_1 = pd.DataFrame(cor_1, columns=["X", "Y"])
    S, Stdev = how_much_rect(cor_1, r=12, c=9)

    S = int(S)

    avg = int(0.25 * S)

    # Crop patches for centroid estimation (used for the first image in a sequence).
    cropped_images, Data_x_Y = crop_images_from_coordinates_3(image, cor_1, crop_size=(int(avg), int(avg)))
    if cropped_images is None or len(cropped_images) == 0 or cropped_images[0].size == 0:
        print("No Coners")
        return cor_1

    data_list_m = []

    for image_c in cropped_images:
        processed_image, blurred_image, padding_size = pre_process_image(image_c, 79, avg)

        # Locate black pixels (foreground).
        black_pixels = np.where(processed_image == 0)
        y_indices, x_indices = black_pixels

        height, width = processed_image.shape[:2]
        # Abort if the blob touches the patch boundary.
        if (
            np.any(y_indices == 0) or
            np.any(y_indices == height - 1) or
            np.any(x_indices == 0) or
            np.any(x_indices == width - 1)
        ):
            # print("Error: Black pixel is out of pixel.")
            display_image(processed_image,)
            break

        # Centroid as the mean of black-pixel coordinates.
        distances_x = []
        distances_y = []
        for y, x in zip(*black_pixels):
            distances_x.append(x)
            distances_y.append(y)

        average_x = sum(distances_x) / len(distances_x) if distances_x else None
        average_y = sum(distances_y) / len(distances_y) if distances_y else None

        data = {"X": [average_x - padding_size], "Y": [average_y - padding_size]}
        df = pd.DataFrame(data)
        data_list_m.append(df)

    middle_df = pd.concat(data_list_m, ignore_index=True)
    Data = Data_x_Y + middle_df

    return Data, avg, sigmaX


def find_Centroid_12_9_Rest_image(image, avg, sigmaX):
    i = 0

    # Re-detect corners using sigmaX estimated from the first frame.
    cor_1, sigmaX = findcorners_12_9_SigmaX(image, sigmaX)

    cropped_images, Data_x_Y = crop_images_from_coordinates_3(image, cor_1, crop_size=(int(avg), int(avg)))
    if cropped_images is None or len(cropped_images) == 0 or cropped_images[0].size == 0:
        print("No Coners")
        return cor_1

    data_list_m = []

    for image_c in cropped_images:
        processed_image, blurred_image, padding_size = pre_process_image(image_c, 79, avg)

        # Locate black pixels (foreground).
        black_pixels = np.where(processed_image == 0)
        y_indices, x_indices = black_pixels

        height, width = processed_image.shape[:2]
        # If the blob touches the boundary, try to remove border-touching connected components.
        if (
            np.any(y_indices == 0) or
            np.any(y_indices == height - 1) or
            np.any(x_indices == 0) or
            np.any(x_indices == width - 1)
        ):
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                (processed_image == 0).astype(np.uint8), connectivity=8
            )

            border_touching_labels = []
            for label in range(1, num_labels):  # label 0 is background
                x, y, w, h, area = stats[label]
                if x == 0 or y == 0 or (x + w) == width or (y + h) == height:
                    border_touching_labels.append(label)

            # Case 1: only one blob touches the border -> keep as-is (do not break the pipeline).
            if len(border_touching_labels) >= 1 and num_labels == 2:
                pass

            # Case 2: multiple blobs touch the border -> remove all border-touching blobs.
            elif len(border_touching_labels) >= 1 and num_labels >= 3:
                margin = 0
                for label in border_touching_labels:
                    x, y, w, h, area = stats[label]

                    touch_top = (y <= margin)
                    touch_bottom = (y + h >= height - 1 - margin)
                    touch_left = (x <= margin)
                    touch_right = (x + w >= width - 1 - margin)

                    if touch_top or touch_bottom or touch_left or touch_right:
                        processed_image[labels == label] = 255

                black_pixels = np.where(processed_image == 0)

        # Centroid as the mean of black-pixel coordinates.
        distances_x = []
        distances_y = []
        for y, x in zip(*black_pixels):
            distances_x.append(x)
            distances_y.append(y)

        average_x = sum(distances_x) / len(distances_x) if distances_x else None
        average_y = sum(distances_y) / len(distances_y) if distances_y else None

        if average_y != None:
            data = {"X": [average_x - padding_size], "Y": [average_y - padding_size]}

        df = pd.DataFrame(data)
        data_list_m.append(df)

    middle_df = pd.concat(data_list_m, ignore_index=True)
    Data = Data_x_Y + middle_df
    return Data


def find_Centroid_12_9_first_image_plus_ten(image, avg=None):
    i = 0
    df_small, df_big, sigmaX = findcorners_12_9_plus_10(image)
    cor_1 = pd.DataFrame(df_small, columns=["X", "Y"])

    # Estimate crop size from the grid spacing if not provided.
    if avg is None:
        S, Stdev = how_much_rect(cor_1, r=12, c=9)
        k = 1
        avg = int(0.25 * int(S))  # 25% of the estimated spacing
    else:
        avg = int(avg)
        k = 2

    # ----- Small crop branch -----
    cropped_images, Data_x_Y = crop_images_from_coordinates_3(image, cor_1, crop_size=(int(avg), int(avg)))
    if cropped_images is None or len(cropped_images) == 0 or cropped_images[0].size == 0:
        print("No Coners")
        return cor_1

    data_list_m = []
    black_count_list = []

    for image_c in cropped_images:
        processed_image, blurred_image, padding_size = pre_process_image(image_c, 79, avg)

        # Locate black pixels (foreground).
        black_pixels = np.where(processed_image == 0)
        y_indices, x_indices = black_pixels
        height, width = processed_image.shape[:2]

        # If border-touching, attempt to remove the border blobs.
        if (
            np.any(y_indices == 0) or
            np.any(y_indices == height - 1) or
            np.any(x_indices == 0) or
            np.any(x_indices == width - 1)
        ):
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                (processed_image == 0).astype(np.uint8), connectivity=8
            )

            border_touching_labels = []
            for label in range(1, num_labels):  # label 0 is background
                x, y, w, h, area = stats[label]
                if x == 0 or y == 0 or (x + w) == width or (y + h) == height:
                    border_touching_labels.append(label)

            if len(border_touching_labels) >= 1 and num_labels == 2:
                pass
            elif len(border_touching_labels) >= 1 and num_labels >= 3:
                margin = 0
                for label in border_touching_labels:
                    x, y, w, h, area = stats[label]

                    touch_top = (y <= margin)
                    touch_bottom = (y + h >= height - 1 - margin)
                    touch_left = (x <= margin)
                    touch_right = (x + w >= width - 1 - margin)

                    if touch_top or touch_bottom or touch_left or touch_right:
                        processed_image[labels == label] = 255

                black_pixels = np.where(processed_image == 0)

        # Centroid as the mean of black-pixel coordinates.
        distances_x = []
        distances_y = []
        for y, x in zip(*black_pixels):
            distances_x.append(x)
            distances_y.append(y)

        average_x = sum(distances_x) / len(distances_x) if distances_x else None
        average_y = sum(distances_y) / len(distances_y) if distances_y else None

        if average_y != None:
            data = {"X": [average_x - padding_size], "Y": [average_y - padding_size]}

        df = pd.DataFrame(data)
        data_list_m.append(df)

    middle_df = pd.concat(data_list_m, ignore_index=True)
    Data_small = Data_x_Y + middle_df

    # ----- Large crop branch -----
    # Verified that a 2.5x crop does not clip the blob in typical cases.
    cor_1 = df_big
    cropped_images, Data_x_Y = crop_images_from_coordinates_3(
        image, cor_1, crop_size=(int(avg * 2.5), int(avg * 2.5))
    )
    if cropped_images is None or len(cropped_images) == 0 or cropped_images[0].size == 0:
        print("No Coners")
        return cor_1

    data_list_m = []

    for image_c in cropped_images:
        processed_image, blurred_image, padding_size = pre_process_image(image_c, 79, avg)

        # Locate black pixels (foreground).
        black_pixels = np.where(processed_image == 0)
        y_indices, x_indices = black_pixels
        height, width = processed_image.shape[:2]

        # If border-touching, attempt to remove border blobs or abort for invalid single-blob cases.
        if (
            np.any(y_indices == 0) or
            np.any(y_indices == height - 1) or
            np.any(x_indices == 0) or
            np.any(x_indices == width - 1)
        ):
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                (processed_image == 0).astype(np.uint8), connectivity=8
            )

            border_touching_labels = []
            for label in range(1, num_labels):  # label 0 is background
                x, y, w, h, area = stats[label]
                if x == 0 or y == 0 or (x + w) == width or (y + h) == height:
                    border_touching_labels.append(label)

            # Case 1: single blob touches the border -> abort this patch.
            if len(border_touching_labels) >= 1 and num_labels == 2:
                break
                pass

            # Case 2: multiple blobs -> remove border-touching blobs.
            elif len(border_touching_labels) >= 1 and num_labels >= 3:
                margin = 0
                for label in border_touching_labels:
                    x, y, w, h, area = stats[label]

                    touch_top = (y <= margin)
                    touch_bottom = (y + h >= height - 1 - margin)
                    touch_left = (x <= margin)
                    touch_right = (x + w >= width - 1 - margin)

                    if touch_top or touch_bottom or touch_left or touch_right:
                        processed_image[labels == label] = 255

                black_pixels = np.where(processed_image == 0)

        # Centroid as the mean of black-pixel coordinates.
        distances_x = []
        distances_y = []
        for y, x in zip(*black_pixels):
            distances_x.append(x)
            distances_y.append(y)

        average_x = sum(distances_x) / len(distances_x) if distances_x else None
        average_y = sum(distances_y) / len(distances_y) if distances_y else None

        # Abort if centroid cannot be computed.
        if average_y == None or average_x == None:
            break

        data = {"X": [average_x - padding_size], "Y": [average_y - padding_size]}
        df = pd.DataFrame(data)
        data_list_m.append(df)

        ###black count
        black_count = int(len(black_pixels[0]))  # ## in English
        black_count_list.append(black_count)

        # Keep only valid DataFrames (defensive programming for downstream concat).
        valid = [df for df in data_list_m if isinstance(df, pd.DataFrame) and not df.empty]

    # Concatenate per-patch results and add to crop origins.
    valid = [df for df in data_list_m if isinstance(df, pd.DataFrame) and not df.empty]
    middle_df = pd.concat(data_list_m, ignore_index=True)
    Data_big = Data_x_Y + middle_df

    return Data_small, Data_big, avg


def find_Centroid_12_9_first_image_plus_ten_count(image, avg=None):
    i = 0
    df_small, df_big, sigmaX = findcorners_12_9_plus_10(image)
    cor_1 = pd.DataFrame(df_small, columns=["X", "Y"])

    # Estimate crop size from the grid spacing if not provided.
    if avg is None:
        S, Stdev = how_much_rect(cor_1, r=12, c=9)
        k = 1
        avg = int(0.25 * int(S))  # 25% of the estimated spacing
    else:
        avg = int(avg)
        k = 2

    # ----- Small crop branch -----
    cropped_images, Data_x_Y = crop_images_from_coordinates_3(image, cor_1, crop_size=(int(avg), int(avg)))
    if cropped_images is None or len(cropped_images) == 0 or cropped_images[0].size == 0:
        print("No Coners")
        return cor_1

    data_list_m = []
    black_count_list = []
    circularity_list = []

    for image_c in cropped_images:
        processed_image, blurred_image, padding_size = pre_process_image(image_c, 79, avg)

        # Locate black pixels (foreground).
        black_pixels = np.where(processed_image == 0)
        y_indices, x_indices = black_pixels
        height, width = processed_image.shape[:2]

        # If border-touching, attempt to remove the border blobs.
        if (
            np.any(y_indices == 0) or
            np.any(y_indices == height - 1) or
            np.any(x_indices == 0) or
            np.any(x_indices == width - 1)
        ):
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                (processed_image == 0).astype(np.uint8), connectivity=8
            )

            border_touching_labels = []
            for label in range(1, num_labels):  # label 0 is background
                x, y, w, h, area = stats[label]
                if x == 0 or y == 0 or (x + w) == width or (y + h) == height:
                    border_touching_labels.append(label)

            if len(border_touching_labels) >= 1 and num_labels == 2:
                pass
            elif len(border_touching_labels) >= 1 and num_labels >= 3:
                margin = 0
                for label in border_touching_labels:
                    x, y, w, h, area = stats[label]

                    touch_top = (y <= margin)
                    touch_bottom = (y + h >= height - 1 - margin)
                    touch_left = (x <= margin)
                    touch_right = (x + w >= width - 1 - margin)

                    if touch_top or touch_bottom or touch_left or touch_right:
                        processed_image[labels == label] = 255

                black_pixels = np.where(processed_image == 0)

        # Centroid as the mean of black-pixel coordinates.
        distances_x = []
        distances_y = []
        for y, x in zip(*black_pixels):
            distances_x.append(x)
            distances_y.append(y)

        average_x = sum(distances_x) / len(distances_x) if distances_x else None
        average_y = sum(distances_y) / len(distances_y) if distances_y else None

        if average_y != None:
            data = {"X": [average_x - padding_size], "Y": [average_y - padding_size]}

        df = pd.DataFrame(data)
        data_list_m.append(df)

        ###black count
        black_count = int(len(black_pixels[0]))
        black_count_list.append(black_count)

        cnts, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            circularity = np.nan
        else:
            # cnt = max(cnts, key=cv2.contourArea)
            cnt = max(cnts, key=cv2.contourArea)
            A = black_count ## Area
            P = float(cv2.arcLength(cnt, True))
            if P < 1e-9 or A < 1e-9:
                circularity = np.nan
            else:
                circularity = float(4.0 * np.pi * A / (P * P))

        circularity_list.append(circularity)



    middle_df = pd.concat(data_list_m, ignore_index=True)
    Data_small = Data_x_Y + middle_df

    # ----- Large crop branch -----
    # Verified that a 2.5x crop does not clip the blob in typical cases.
    cor_1 = df_big
    cropped_images, Data_x_Y = crop_images_from_coordinates_3(
        image, cor_1, crop_size=(int(avg * 2.5), int(avg * 2.5))
    )
    if cropped_images is None or len(cropped_images) == 0 or cropped_images[0].size == 0:
        print("No Coners")
        return cor_1

    data_list_m = []

    for image_c in cropped_images:
        processed_image, blurred_image, padding_size = pre_process_image(image_c, 79, avg)

        # Locate black pixels (foreground).
        black_pixels = np.where(processed_image == 0)
        y_indices, x_indices = black_pixels
        height, width = processed_image.shape[:2]

        # If border-touching, attempt to remove border blobs or abort for invalid single-blob cases.
        if (
            np.any(y_indices == 0) or
            np.any(y_indices == height - 1) or
            np.any(x_indices == 0) or
            np.any(x_indices == width - 1)
        ):
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                (processed_image == 0).astype(np.uint8), connectivity=8
            )

            border_touching_labels = []
            for label in range(1, num_labels):  # label 0 is background
                x, y, w, h, area = stats[label]
                if x == 0 or y == 0 or (x + w) == width or (y + h) == height:
                    border_touching_labels.append(label)

            # Case 1: single blob touches the border -> abort this patch.
            if len(border_touching_labels) >= 1 and num_labels == 2:
                break
                pass

            # Case 2: multiple blobs -> remove border-touching blobs.
            elif len(border_touching_labels) >= 1 and num_labels >= 3:
                margin = 0
                for label in border_touching_labels:
                    x, y, w, h, area = stats[label]

                    touch_top = (y <= margin)
                    touch_bottom = (y + h >= height - 1 - margin)
                    touch_left = (x <= margin)
                    touch_right = (x + w >= width - 1 - margin)

                    if touch_top or touch_bottom or touch_left or touch_right:
                        processed_image[labels == label] = 255

                black_pixels = np.where(processed_image == 0)

        # Centroid as the mean of black-pixel coordinates.
        distances_x = []
        distances_y = []
        for y, x in zip(*black_pixels):
            distances_x.append(x)
            distances_y.append(y)

        average_x = sum(distances_x) / len(distances_x) if distances_x else None
        average_y = sum(distances_y) / len(distances_y) if distances_y else None

        # Abort if centroid cannot be computed.
        if average_y == None or average_x == None:
            break

        # print(average_x, average_y)
        data = {"X": [average_x - padding_size], "Y": [average_y - padding_size]}
        df = pd.DataFrame(data)
        data_list_m.append(df)


        # Keep only valid DataFrames (defensive programming for downstream concat).
        valid = [df for df in data_list_m if isinstance(df, pd.DataFrame) and not df.empty]

    # Concatenate per-patch results and add to crop origins.
    valid = [df for df in data_list_m if isinstance(df, pd.DataFrame) and not df.empty]
    middle_df = pd.concat(data_list_m, ignore_index=True)
    Data_big = Data_x_Y + middle_df

    return Data_small, Data_big, avg,black_count_list,circularity_list

def find_contour_for_find_12_9(image):
    i = 0
    cor_1, sigmaX = findcorners_12_9(image, 135, 800, 10)
    cor_1 = pd.DataFrame(cor_1, columns=["X", "Y"])
    S, Stdev = how_much_rect(cor_1, r=12, c=9)

    S = int(S)

    # Note: Using avg=0.4*S can cause cropping artifacts; 0.25*S is used here.
    avg = int(0.25 * S)

    # Crop patches around detected corners for contour-based ellipse fitting.
    cropped_images, Data_x_Y = crop_images_from_coordinates_3(image, cor_1, crop_size=(int(avg), int(avg)))

    if cropped_images is None or len(cropped_images) == 0 or cropped_images[0].size == 0:
        print("No Coners")
        return cor_1

    processed_images = []
    # List to accumulate ellipse-center estimates per patch.
    data_list_m = []
    data = []

    for image_c in cropped_images:
        processed_image, blurred_image, padding_size = pre_process_image(image_c, 79, avg)

        # Abort if the foreground touches the patch boundary (likely clipped contour).
        black_pixels = np.where(processed_image == 0)
        y_indices, x_indices = black_pixels
        height, width = processed_image.shape[:2]
        if (
            np.any(y_indices == 0) or
            np.any(y_indices == height - 1) or
            np.any(x_indices == 0) or
            np.any(x_indices == width - 1)
        ):
            break

        # Extract all contour points from the binary patch.
        contours, hierarchy = cv2.findContours(processed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        coords = []
        for contour in contours:
            for point in contour:
                # OpenCV contour points are expected to be shaped as (1, 2).
                if isinstance(point, np.ndarray) and point.shape == (1, 2):
                    x = point[0][0]
                    y = point[0][1]
                    coords.append((x, y))
                else:
                    raise ValueError("Unexpected point structure: {}".format(point))

        coords = np.array(coords)

        # Fit an implicit ellipse to contour points using non-linear least squares.
        x_data = coords[:, 0]
        y_data = coords[:, 1]
        initial_params = [1, 0, 1, 0, 0]  # initial guess for (A, B, C, D, E)

        result = least_squares(residuals, initial_params, args=(x_data, y_data))
        fitted_params = result.x

        A, B, C, D, E = fitted_params

        # Compute ellipse center (x0, y0) from conic parameters in implicit form.
        xo = (B * E - 2 * C * D) / (4 * A * C - B ** (2)) - padding_size
        yo = (B * D - 2 * A * E) / (4 * A * C - B ** (2)) - padding_size

        data = {"X": [xo], "Y": [yo]}
        df = pd.DataFrame(data)
        data_list_m.append(df)

    middle_df = pd.concat(data_list_m, ignore_index=True)
    Data = Data_x_Y + middle_df

    Data3 = Data
    return Data3, avg


def find_ZernikeEdge_curvefit_12_9(image):
    i = 0

    cor_1, sigmaX = findcorners_12_9(image, 135, 800, 10)
    cor_1 = pd.DataFrame(cor_1, columns=["X", "Y"])
    S, Stdev = how_much_rect(cor_1, r=12, c=9)
    S = int(S)

    avg = int(0.25 * S)

    # Crop patches around detected grid corners.
    cropped_images, Data_x_Y = crop_images_from_coordinates_3(image, cor_1, crop_size=(int(avg), int(avg)))

    if cropped_images is None or len(cropped_images) == 0 or cropped_images[0].size == 0:
        print("No Coners")
        return cor_1

    processed_images = []
    # List to accumulate ellipse-center estimates per patch.
    data_list_m = []
    data = []

    for image_c in cropped_images:
        processed_image, blurred_image, padding_size = pre_process_image(image_c, 79, avg)

        black_pixels = np.where(processed_image == 0)
        y_indices, x_indices = black_pixels
        height, width = processed_image.shape[:2]

        # If boundary-touching occurs, attempt connected-component cleanup (same logic as centroid path).
        if (
            np.any(y_indices == 0) or
            np.any(y_indices == height - 1) or
            np.any(x_indices == 0) or
            np.any(x_indices == width - 1)
        ):
            print("Error: Black pixel is out of pixel.")

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                (processed_image == 0).astype(np.uint8), connectivity=8
            )

            border_touching_labels = []
            for label in range(1, num_labels):  # label 0 is background
                x, y, w, h, area = stats[label]
                if x == 0 or y == 0 or (x + w) == width or (y + h) == height:
                    border_touching_labels.append(label)

            if len(border_touching_labels) >= 1 and num_labels == 2:
                break

            elif len(border_touching_labels) == 1 and num_labels == 3:
                for label in border_touching_labels:
                    x, y, w, h, area = stats[label]
                    if y + h >= height - 1:
                        processed_image[labels == label] = 255

                # Convert to a simplified binary encoding for contour extraction.
                processed_image = np.where(processed_image == 0, 255, 0).astype(np.uint8)

        # Extract contours and then refine them to subpixel precision using Zernike moments.
        contours, hierarchy = cv2.findContours(processed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        coords = []
        for contour in contours:
            for point in contour:
                if isinstance(point, np.ndarray) and point.shape == (1, 2):
                    x = point[0][0]
                    y = point[0][1]
                    coords.append((x, y))
                else:
                    raise ValueError("Unexpected point structure: {}".format(point))

        # Compute coarse centroid of contour points (debug/diagnostic).
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        x_mean = np.mean(x_coords)
        y_mean = np.mean(y_coords)
        average_coords = (x_mean, y_mean)

        # Subpixel refinement of contour points.
        sub_contours = subpixel_edge_zernike(processed_image, coords, radius=10)
        sub_contours = np.array(sub_contours)

        # Compute subpixel contour mean (debug/diagnostic).
        x_coords = [coord[0] for coord in sub_contours]
        y_coords = [coord[1] for coord in sub_contours]
        x_mean = np.mean(x_coords)
        y_mean = np.mean(y_coords)
        average_coords = (x_mean, y_mean)

        # Fit ellipse using subpixel contour points.
        x_data = sub_contours[:, 0]
        y_data = sub_contours[:, 1]
        initial_params = [1, 0, 1, 0, 0]

        result = least_squares(residuals, initial_params, args=(x_data, y_data))
        fitted_params = result.x

        A, B, C, D, E = fitted_params

        # Compute ellipse center and undo padding offset.
        xo = (B * E - 2 * C * D) / (4 * A * C - B ** (2)) - padding_size
        yo = (B * D - 2 * A * E) / (4 * A * C - B ** (2)) - padding_size

        data = {"X": [xo], "Y": [yo]}
        df = pd.DataFrame(data)
        data_list_m.append(df)

    middle_df = pd.concat(data_list_m, ignore_index=True)
    Data = Data_x_Y + middle_df

    S, std = how_much_rect(Data, 6, 6)

    return Data, avg


def find_cv_12_9(image):
    # Preprocess image for blob detection and circle-grid extraction.
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Configure OpenCV SimpleBlobDetector for circular dot targets.
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 100000
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0.5

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect a symmetric 12x9 circles grid.
    ret, corners = cv2.findCirclesGrid(
        image, (12, 9),
        flags=(cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING),
        blobDetector=detector
    )

    corners = corners.reshape(-1, 2)
    df = pd.DataFrame(corners, columns=['X', 'Y'])

    return df


def find_Square_12_9_GT(image):
    i = 0
    r, c = 12, 9

    cor_1, sigmaX = findcorners_12_9(image, 135, 800, 10)
    cor_1 = pd.DataFrame(cor_1, columns=["X", "Y"])


    S, Stdev = how_much_rect(cor_1, r=12, c=9)
    S = int(S)
    avg = int(0.25 * S)

    # Crop patches around corners; here "GT" uses raw black pixels without preprocessing.
    cropped_images, Data_x_Y = crop_images_from_coordinates_3(image, cor_1, crop_size=(int(avg), int(avg)))
    if cropped_images is None or len(cropped_images) == 0 or cropped_images[0].size == 0:
        print("No Coners")
        return cor_1

    processed_images = []
    data_list_m = []
    data = []

    for image_c in cropped_images:
        # Compute centroid directly from black pixels in the raw cropped patch.
        black_pixels = np.where(image_c == 0)

        distances_x = []
        distances_y = []
        for y, x in zip(*black_pixels):
            distances_x.append(x)
            distances_y.append(y)

        padding_size = 0
        average_x = sum(distances_x) / len(distances_x) if distances_x else None
        average_y = sum(distances_y) / len(distances_y) if distances_y else None

        data = {"X": [average_x - padding_size], "Y": [average_y - padding_size]}
        df = pd.DataFrame(data)
        data_list_m.append(df)

    middle_df = pd.concat(data_list_m, ignore_index=True)
    Data = Data_x_Y + middle_df
    return Data, avg


def find_Square_12_9_Centroid(image):
    i = 0

    cor_1, sigmaX = findcorners_12_9(image, 135, 800, 10)
    cor_1 = pd.DataFrame(cor_1, columns=["X", "Y"])
    print(cor_1)

    S, Stdev = how_much_rect(cor_1, r=12, c=9)
    S = int(S)
    avg = int(0.25 * S)

    # Crop patches around corners and estimate centroid from preprocessed binary blobs.
    cropped_images, Data_x_Y = crop_images_from_coordinates_3(image, cor_1, crop_size=(int(avg), int(avg)))
    if cropped_images is None or len(cropped_images) == 0 or cropped_images[0].size == 0:
        print("No Coners")
        return cor_1

    processed_images = []
    data_list_m = []
    data = []

    for image_c in cropped_images:
        processed_image, blurred_image, padding_size = pre_process_image(image_c, 79, avg)

        black_pixels = np.where(processed_image == 0)
        y_indices, x_indices = black_pixels
        height, width = processed_image.shape[:2]

        # If border-touching occurs, attempt connected-component cleanup.
        if (
            np.any(y_indices == 0) or
            np.any(y_indices == height - 1) or
            np.any(x_indices == 0) or
            np.any(x_indices == width - 1)
        ):
            print("Error: Black pixel is out of pixel.")

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                (processed_image == 0).astype(np.uint8), connectivity=8
            )

            border_touching_labels = []
            for label in range(1, num_labels):  # label 0 is background
                x, y, w, h, area = stats[label]
                if x == 0 or y == 0 or (x + w) == width or (y + h) == height:
                    border_touching_labels.append(label)

            if len(border_touching_labels) >= 1 and num_labels == 2:
                break

            elif len(border_touching_labels) == 1 and num_labels == 3:
                for label in border_touching_labels:
                    x, y, w, h, area = stats[label]
                    if y + h >= height - 1:
                        processed_image[labels == label] = 255

                black_pixels = np.where(processed_image == 0)

        # Compute centroid as the mean of black-pixel coordinates.
        distances_x = []
        distances_y = []
        for y, x in zip(*black_pixels):
            distances_x.append(x)
            distances_y.append(y)

        padding_size = 0
        average_x = sum(distances_x) / len(distances_x) if distances_x else None
        average_y = sum(distances_y) / len(distances_y) if distances_y else None

        data = {"X": [average_x - padding_size], "Y": [average_y - padding_size]}
        df = pd.DataFrame(data)
        data_list_m.append(df)

    middle_df = pd.concat(data_list_m, ignore_index=True)
    Data = Data_x_Y + middle_df
    return Data, avg


### Point normalization

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull


from lib.L1_Image_Conversion import *
from lib.L2_Point_Detection_Conversion import *
# from lib.L3_Point_Conversion import *
# from lib.L4_Zhangs_Calibration import *
# from lib.L5_Pipeline_utility import *
from lib.L5_Visualization_Utilities import *

def normalize_points(points):
    """
    Normalize 2D points for numerically stable homography estimation.

    Args:
        points: (N, 2) numpy array of (X, Y).

    Returns:
        points_normalized: (N, 2) normalized points
        T: (3, 3) normalization transform matrix
    """
    # Robust normalization based on median and mean distance to the median.
    median_vals = np.median(points, axis=0)
    mean_dist = np.mean(np.linalg.norm(points - median_vals, axis=1))

    scale = 1 / mean_dist if mean_dist > 0 else 1
    T = np.array([
        [scale, 0, -median_vals[0] * scale],
        [0, scale, -median_vals[1] * scale],
        [0, 0, 1]
    ])

    # Convert to homogeneous coordinates and apply transform.
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_normalized = (T @ points_homogeneous.T).T

    return points_normalized[:, :2], T


def compute_homography_noraml(points_src, points_dst):
    """
    Compute homography from source points to destination points with normalization (DLT).

    Args:
        points_src: (N, 2) array-like or DataFrame (measured points, e.g., from circle grid)
        points_dst: (N, 2) array-like or DataFrame (ideal grid points)

    Returns:
        H: (3, 3) homography matrix normalized so that H[-1, -1] == 1
    """
    if isinstance(points_src, pd.DataFrame):
        points_src = points_src.to_numpy(dtype=np.float64)
    else:
        points_src = np.asarray(points_src, dtype=np.float64)

    if isinstance(points_dst, pd.DataFrame):
        points_dst = points_dst.to_numpy(dtype=np.float64)
    else:
        points_dst = np.asarray(points_dst, dtype=np.float64)

    # 1) Normalize coordinates
    points_src_norm, T_src = normalize_points(points_src)
    points_dst_norm, T_dst = normalize_points(points_dst)

    num_points = points_src_norm.shape[0]
    A = np.zeros((2 * num_points, 9))

    # 2) Build DLT matrix A
    for i in range(num_points):
        X_src, Y_src = points_src_norm[i]
        X_dst, Y_dst = points_dst_norm[i]

        A[2 * i] = [-X_src, -Y_src, -1, 0, 0, 0, X_src * X_dst, Y_src * X_dst, X_dst]
        A[2 * i + 1] = [0, 0, 0, -X_src, -Y_src, -1, X_src * Y_dst, Y_src * Y_dst, Y_dst]

    # 3) Solve via SVD
    U, S, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)

    # 4) Denormalize
    H = np.linalg.inv(T_dst) @ H_norm @ T_src

    # Zero-out extremely small elements for neatness
    H[np.abs(H) < 1e-10] = 0

    return H / H[-1, -1]


def compute_homography(points_src, points_dst):
    """
    Compute homography from source points to destination points (DLT, no normalization).

    Args:
        points_src: DataFrame of source points (X, Y)
        points_dst: DataFrame of destination points (X, Y)

    Returns:
        H: (3, 3) homography matrix normalized so that H[-1, -1] == 1
    """
    points_src = points_src.to_numpy()
    points_dst = points_dst.to_numpy()

    points_dst = np.squeeze(points_dst)

    num_points = points_src.shape[0]
    A = np.zeros((2 * num_points, 9))

    for i in range(num_points):
        X_src, Y_src = points_src[i]
        X_dst, Y_dst = points_dst[i]

        A[2 * i] = [-X_src, -Y_src, -1, 0, 0, 0, X_src * X_dst, Y_src * X_dst, X_dst]
        A[2 * i + 1] = [0, 0, 0, -X_src, -Y_src, -1, X_src * Y_dst, Y_src * Y_dst, Y_dst]

    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    H[np.abs(H) < 1e-10] = 0
    return H / H[-1, -1]


def compute_homography_rmse(points_src, points_dst, H):
    """
    Apply homography to source points and compute RMSE to destination points.
    Also aligns the first point via translation before RMSE computation.

    Args:
        points_src: (N, 2) numpy array
        points_dst: DataFrame with columns ['X','Y'] (or compatible)
        H: (3, 3) homography matrix

    Returns:
        rmse: float
    """
    ones = np.ones((points_src.shape[0], 1))
    homogenous_points_src = np.hstack([points_src, ones])

    transformed_points = np.dot(H, homogenous_points_src.T).T
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2][:, np.newaxis]

    # Align the first transformed point to the first destination point
    adjustment_point = points_dst.iloc[0].values
    translation = adjustment_point - transformed_points[0]
    adjusted_transformed_points = transformed_points + translation



    rmse = np.sqrt(mean_squared_error(points_dst, adjusted_transformed_points))
    return rmse


def compute_homography_diff(points_src, points_dst, H):
    """
    Apply homography to source points and compute mean Euclidean error to destination points.
    Aligns the first point via translation before error computation.

    Args:
        points_src: (N, 2) numpy array
        points_dst: DataFrame with columns ['X','Y']
        H: (3, 3) homography matrix

    Returns:
        mean_distance: Series or scalar (mean of distances)
    """
    ones = np.ones((points_src.shape[0], 1))
    homogenous_points_src = np.hstack([points_src, ones])

    transformed_points = np.dot(H, homogenous_points_src.T).T
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2][:, np.newaxis]

    # Optional: align first point
    adjustment_point = points_dst.iloc[0].values
    translation = adjustment_point - transformed_points[0]
    transformed_points = transformed_points + translation

    Error_dst = (points_dst - transformed_points)

    df = pd.DataFrame()
    df['Distance'] = np.sqrt(Error_dst['X'] ** 2 + Error_dst['Y'] ** 2)

    df = np.mean(df)
    return df


def compute_rmse(points_src, points_dst):
    """
    Compute RMSE directly between two point sets (no homography).

    Args:
        points_src: array-like
        points_dst: array-like

    Returns:
        rmse: float
    """
    rmse = np.sqrt(mean_squared_error(points_dst, points_src))
    return rmse


def compute_homography_rmse_point(points_src, points_dst, H):
    """
    Apply homography to source points and compute RMSE to destination points.
    Returns RMSE and transformed points.

    Args:
        points_src: (N, 2) numpy array
        points_dst: (N, 2) numpy array
        H: (3, 3) homography matrix

    Returns:
        rmse: float
        transformed_points: (N, 2) numpy array
    """
    ones = np.ones((points_src.shape[0], 1))
    homogenous_points_src = np.hstack([points_src, ones])

    transformed_points = np.dot(H, homogenous_points_src.T).T
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2][:, np.newaxis]

    rmse = np.sqrt(mean_squared_error(points_dst, transformed_points))
    return rmse, transformed_points


def apply_homography(H, points):
    """
    Apply homography to a set of 2D points.

    Args:
        H: (3, 3) homography matrix
        points: (N, 2) array-like

    Returns:
        transformed_points: (N, 2) numpy array
    """
    ones = np.ones(shape=(len(points), 1))
    homogenous_points = np.hstack([points, ones])

    transformed_points = np.dot(H, homogenous_points.T).T
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2][:, np.newaxis]

    return transformed_points


def Makeobjp(D, r, c):
    """
    Build an ideal planar grid (r x c) with spacing D.

    Args:
        D: spacing
        r: number of rows
        c: number of cols

    Returns:
        df: DataFrame with columns ['X','Y']
    """
    obj_points = []

    objp = np.zeros((r * c, 2), np.float32)

    objp[:, 0] = np.tile(np.arange(1, c + 1, 1), r) * D
    objp[:, 1] = np.repeat(np.arange(1, r + 1, 1), c) * D

    obj_points.append(objp)
    df = pd.DataFrame(objp, columns=['X', 'Y'])
    return df


def create_grid_cylinder(D, n=6):
    """
    Create a "cylindrical" grid where X positions follow a sine-based spacing rule.

    Args:
        D: vertical spacing step
        n: grid dimension (n x n)

    Returns:
        df: DataFrame with columns ['X','Y']
    """
    points = []
    gap_H = D / 6

    def calculate_x_coords():
        X = 6 / 50
        x_coords = [0]
        for i in range(1, n):
            x = 50 * (np.sin(5 * X / 2) - np.sin(3 * X / 2))
            if i >= 2:
                x += 50 * (np.sin(3 * X / 2) - np.sin(X / 2))
            if i >= 3:
                x += 2 * 50 * (np.sin(X / 2))
            if i >= 4:
                x += 50 * (np.sin(3 * X / 2) - np.sin(X / 2))
            if i >= 5:
                x += 50 * (np.sin(5 * X / 2) - np.sin(3 * X / 2))
            x_coords.append(x)
        return x_coords

    x_coords = calculate_x_coords()

    fig, ax = plt.subplots()

    for j in range(n):
        for i in range(n):
            x = x_coords[i] * gap_H
            y = j * D
            points.append((x, y))

    df = pd.DataFrame(points, columns=['X', 'Y'])
    return df


def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Euler angles (degrees).

    Args:
        R: (3, 3) rotation matrix

    Returns:
        angles_deg: array [roll, pitch, yaw] in degrees
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees(np.array([x, y, z]))


def find_best_lines(points, top_n=18, line_points=1):
    """
    (Deprecated by the redefined function below.)
    Find the best-fit line among combinations of points (by R^2).

    Args:
        points: (N, 2) numpy array
        top_n: number of smallest-Y points to consider
        line_points: number of points per candidate line

    Returns:
        best: tuple(score, comb)
    """
    sorted_points = points[points[:, 1].argsort()][:top_n]
    visualize_points(sorted_points)

    combs = list(combinations(sorted_points, line_points))

    best_fit_lines = []
    for comb in combs:
        X = np.array([p[0] for p in comb]).reshape(-1, 1)
        y = np.array([p[1] for p in comb])
        model = LinearRegression().fit(X, y)
        score = model.score(X, y)
        best_fit_lines.append((score, comb))

    best_fit_lines.sort(reverse=True, key=lambda x: x[0])
    return best_fit_lines[0]


def find_best_lines(points):
    """
    Sort points into a 6x6 grid by iteratively selecting 6 points that form the "best" row.

    Args:
        points: (N, 2) numpy array

    Returns:
        ordered_points: (36, 2) numpy array
    """
    ordered_points = np.zeros((36, 2))
    current_index = 0

    for step in range(6):
        n_points = len(points)
        top_n = min(18, n_points)

        sorted_points = points[np.argsort(points[:, 1])][:top_n]
        candidate_lines = list(combinations(sorted_points, 6))

        y_values = np.array([p[1] for p in candidate_lines[0]])
        if np.all(y_values == y_values[0]):
            best_line = sorted(candidate_lines[0], key=lambda p: p[0])
        else:
            best_line = None
            best_r_value = -np.inf
            lowest_y_mean = np.inf

            for line in candidate_lines:
                X = np.array([p[0] for p in line]).reshape(-1, 1)
                y = np.array([p[1] for p in line])
                model = LinearRegression().fit(X, y)
                score = model.score(X, y)
                y_mean = np.mean(y)

                if score >= 0.9 and y_mean < lowest_y_mean:
                    best_line = line
                    best_r_value = score
                    lowest_y_mean = y_mean

        if best_line is not None:
            best_line = np.array(sorted(best_line, key=lambda p: p[0]))
            ordered_points[current_index:current_index + 6] = best_line
            current_index += 6

            points = np.array([p for p in points if p.tolist() not in best_line.tolist()])

    return ordered_points


def find_best_lines_2(points):
    """
    Robust 6x6 row extraction by repeatedly selecting a plausible row of 6 points.

    Args:
        points: (N, 2) numpy array

    Returns:
        ordered_points: (36, 2) numpy array
    """
    def is_all_zeros(arr):
        return np.all(arr == 0)

    if is_all_zeros(points):
        return points

    if isinstance(points, str):
        print(f"Error: points should not be a string. Received: {points}")
        return points

    if not isinstance(points, np.ndarray):
        print("Error: points is not a numpy array.")
        return None

    if points is None or points.ndim != 2:
        return points

    ordered_points = np.zeros((36, 2))
    current_index = 0

    if not is_all_zeros(points):
        while len(points) > 3:
            print("Current number of points:", len(points))

            top_n = min(18, len(points))
            sorted_points = points[np.argsort(points[:, 1])][:top_n]
            candidate_lines = list(combinations(sorted_points, 6))

            best_line = None
            for candidate in candidate_lines:
                y_values = np.array([p[1] for p in candidate])
                y_mean = np.mean(y_values)

                # For near-front view: allow points in (almost) same row
                if np.all(y_values == y_values[0]) or np.all(np.abs(y_values - y_mean) <= 100):
                    best_line = sorted(candidate, key=lambda p: p[0])
                else:
                    best_line = None
                    best_r_value = -np.inf
                    lowest_y_mean = np.inf

                    for line in candidate_lines:
                        X = np.array([p[0] for p in line]).reshape(-1, 1)
                        y = np.array([p[1] for p in line])
                        model = LinearRegression().fit(X, y)
                        score = model.score(X, y)
                        y_mean = np.mean(y)

                        if score >= 0.9 and y_mean < lowest_y_mean:
                            best_line = line
                            lowest_y_mean = y_mean

                if best_line is not None:
                    best_line = np.array(sorted(best_line, key=lambda p: p[0]))
                    ordered_points[current_index:current_index + 6] = best_line
                    current_index += 6

                    points = np.array([p for p in points if p.tolist() not in best_line.tolist()])
                    break

        return ordered_points


def find_best_lines_12_9(points):
    """
    Sort points into a 12x9 grid (108 points): by Y, then X within each row.

    Args:
        points: (N, 2) numpy array or DataFrame

    Returns:
        ordered_points: (108, 2) numpy array
    """
    if isinstance(points, str):
        print(f"Error: points should not be a string. Received: {points}")
        return None

    if points is None or len(points) == 0 or points.ndim != 2:
        return None

    points = points.values if hasattr(points, 'values') else np.array(points)

    ordered_points = np.zeros((108, 2))
    sorted_points = points[np.argsort(points[:, 1])]

    for i in range(9):
        start_idx = i * 12
        end_idx = start_idx + 12
        group = sorted_points[start_idx:end_idx]
        ordered_points[start_idx:end_idx] = group[np.argsort(group[:, 0])]

    return ordered_points


def find_best_lines_6_6(points):
    """
    Sort points into a 6x6 grid (36 points): by Y, then X within each row.

    Args:
        points: (N, 2) numpy array or DataFrame

    Returns:
        ordered_points: (36, 2) numpy array
    """
    points = points.values if hasattr(points, 'values') else np.array(points)

    ordered_points = np.zeros((36, 2))
    sorted_points = points[np.argsort(points[:, 1])]

    for i in range(6):
        start_idx = i * 6
        end_idx = start_idx + 6
        group = sorted_points[start_idx:end_idx]
        ordered_points[start_idx:end_idx] = group[np.argsort(group[:, 0])]

    return ordered_points


def find_best_lines_3(points):
    """
    Variant of find_best_lines_2 with a tighter Y tolerance (<= 50).

    Args:
        points: (N, 2) numpy array

    Returns:
        ordered_points: (36, 2) numpy array
    """
    def is_all_zeros(arr):
        return np.all(arr == 0)

    if is_all_zeros(points):
        return points

    if isinstance(points, str):
        print(f"Error: points should not be a string. Received: {points}")
        return points

    if not isinstance(points, np.ndarray):
        print("Error: points is not a numpy array.")
        return None

    if points is None or points.ndim != 2:
        return points

    ordered_points = np.zeros((36, 2))
    current_index = 0

    if not is_all_zeros(points):
        while len(points) > 3:
            print("Current number of points:", len(points))

            top_n = min(18, len(points))
            sorted_points = points[np.argsort(points[:, 1])][:top_n]
            candidate_lines = list(combinations(sorted_points, 6))

            best_line = None
            for candidate in candidate_lines:
                y_values = np.array([p[1] for p in candidate])
                y_mean = np.mean(y_values)

                if np.all(y_values == y_values[0]) or np.all(np.abs(y_values - y_mean) <= 50):
                    best_line = sorted(candidate, key=lambda p: p[0])
                else:
                    best_line = None
                    best_r_value = -np.inf
                    lowest_y_mean = np.inf

                    for line in candidate_lines:
                        X = np.array([p[0] for p in line]).reshape(-1, 1)
                        y = np.array([p[1] for p in line])
                        model = LinearRegression().fit(X, y)
                        score = model.score(X, y)
                        y_mean = np.mean(y)

                        if score >= 0.9 and y_mean < lowest_y_mean:
                            best_line = line
                            lowest_y_mean = y_mean

                if best_line is not None:
                    best_line = np.array(sorted(best_line, key=lambda p: p[0]))
                    ordered_points[current_index:current_index + 6] = best_line
                    current_index += 6

                    points = np.array([p for p in points if p.tolist() not in best_line.tolist()])
                    break

        return ordered_points


def largest_rectangle(points):
    """
    Find the maximum-area quadrilateral from convex hull vertices (brute force).

    Args:
        points: (N, 2) numpy array

    Returns:
        best_rectangle: (4, 2) numpy array
    """
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    max_area = 0
    best_rectangle = None

    for combination in combinations(hull_points, 4):
        rect = np.array(combination)

        area = 0.5 * np.abs(
            rect[0][0] * rect[1][1] + rect[1][0] * rect[2][1] +
            rect[2][0] * rect[3][1] + rect[3][0] * rect[0][1] -
            (rect[1][0] * rect[0][1] + rect[2][0] * rect[1][1] +
             rect[3][0] * rect[2][1] + rect[0][0] * rect[3][1])
        )

        if area > max_area:
            max_area = area
            best_rectangle = rect

    return best_rectangle


def get_points_within_distance(coords, line_start, line_end, distance):
    """
    Select points that lie within a vertical distance from a line segment.

    Args:
        coords: (N, 2) numpy array
        line_start: (2,) array-like
        line_end: (2,) array-like
        distance: tolerance

    Returns:
        selected_points: (M, 2) numpy array
    """
    selected_points = []

    # Vertical line case
    if line_end[0] == line_start[0]:
        for point in coords:
            if abs(point[0] - line_start[0]) <= distance:
                selected_points.append(point)
    else:
        slope = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])
        intercept = line_start[1] - slope * line_start[0]

        for point in coords:
            expected_y = slope * point[0] + intercept
            if abs(point[1] - expected_y) <= distance:
                selected_points.append(point)

    return np.array(selected_points)


def iterative_filtering_12_9(coords_data1, N):
    """
    Iteratively extract 12-point rows from a noisy 12x9 grid.

    Args:
        coords_data1: (N, 2) numpy array
        N: distance tolerance for row selection

    Returns:
        df: DataFrame with columns ['X','Y'] ordered by extracted rows
    """
    all_selected_points = []

    if len(coords_data1) < 2:
        return coords_data1

    while len(coords_data1) > 12:
        largest_points_1 = largest_rectangle(coords_data1)
        largest_points = largest_points_1[largest_points_1[:, 1].argsort()[::-1]]

        line_start, line_end = largest_points[0], largest_points[1]
        filtered_points = get_points_within_distance(coords_data1, line_start, line_end, distance=N)

        if len(filtered_points) < 12:
            line_start, line_end = largest_points[0], largest_points[2]
            filtered_points = get_points_within_distance(coords_data1, line_start, line_end, distance=50)

        sorted_filtered_points = filtered_points[np.argsort(filtered_points[:, 0])]
        all_selected_points.extend(sorted_filtered_points)

        mask = (coords_data1[:, None] == sorted_filtered_points).all(-1).any(1)
        coords_data1 = coords_data1[~mask]

    final_sorted_points = coords_data1[np.argsort(coords_data1[:, 0])]
    all_selected_points.extend(final_sorted_points)

    df = pd.DataFrame(all_selected_points, columns=['X', 'Y'])
    return df


def iterative_filtering_6_6(coords_data1, N):
    """
    Iteratively extract 6-point rows from a noisy 6x6 grid.

    Args:
        coords_data1: (N, 2) numpy array
        N: distance tolerance for row selection

    Returns:
        df: DataFrame with columns ['X','Y'] ordered by extracted rows
    """
    if coords_data1.size == 0:
        return coords_data1

    if coords_data1.ndim == 1:
        coords_data1 = coords_data1.reshape(-1, 2)

    all_selected_points = []

    if coords_data1.size == 1 or np.all(coords_data1.size == 0):
        print("Because of a size issue, iterative_filtering_6_6 cannot proceed.")
        return coords_data1

    if len(coords_data1) < 2:
        print("Because of a size issue, iterative_filtering_6_6 cannot proceed.")
        return coords_data1

    while len(coords_data1) > 6:
        largest_points_1 = largest_rectangle(coords_data1)
        largest_points = largest_points_1[largest_points_1[:, 1].argsort()[::-1]]

        line_start, line_end = largest_points[0], largest_points[1]
        filtered_points = get_points_within_distance(coords_data1, line_start, line_end, distance=N)

        if len(filtered_points) < 6:
            line_start, line_end = largest_points[0], largest_points[2]
            filtered_points = get_points_within_distance(coords_data1, line_start, line_end, distance=N)

        sorted_filtered_points = filtered_points[np.argsort(filtered_points[:, 0])]
        all_selected_points.extend(sorted_filtered_points)

        mask = (coords_data1[:, None] == sorted_filtered_points).all(-1).any(1)
        coords_data1 = coords_data1[~mask]

    if coords_data1.ndim == 1:
        coords_data1 = coords_data1.reshape(-1, 2)

    final_sorted_points = coords_data1[np.argsort(coords_data1[:, 0])]
    all_selected_points.extend(final_sorted_points)

    df = pd.DataFrame(all_selected_points, columns=['X', 'Y'])
    return df


def extract_euler_angles_from_homography(H, camera_matrix):
    """
    Decompose a homography into candidate rotations/translations and return Euler angles
    for a selected rotation hypothesis.

    Args:
        H: (3, 3) homography matrix
        camera_matrix: (3, 3) camera intrinsic matrix

    Returns:
        (X, Y, Z): selected Euler angles (degrees)
    """
    retval, rotations, translations, normals = cv2.decomposeHomographyMat(H, camera_matrix)

    R0 = rotations[0]
    R1 = rotations[1]
    R2 = rotations[2]
    R3 = rotations[3]

    def rotation_matrix_to_euler_angles(R):
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Near gimbal lock: special handling
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        def normalize_angle(angle):
            # Normalize into roughly [-90, 90] with sign conventions used in the original code.
            if -90 < angle < 90:
                angle = -1 * angle
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            return angle

        x, y, z = np.degrees(x), np.degrees(y), np.degrees(z)
        x = normalize_angle(x)
        y = normalize_angle(y)
        z = normalize_angle(z)

        return x, y, z

    X0, Y0, Z0 = rotation_matrix_to_euler_angles(R0)
    X1, Y1, Z1 = rotation_matrix_to_euler_angles(R1)
    X2, Y2, Z2 = rotation_matrix_to_euler_angles(R2)
    X3, Y3, Z3 = rotation_matrix_to_euler_angles(R3)

    R0_index = 0
    R2_index = 2

    norm_R0 = np.sqrt((X0 ** 2) + (Y0 ** 2) + (Z0 ** 2))
    norm_R2 = np.sqrt((X2 ** 2) + (Y2 ** 2) + (Z2 ** 2))

    # Keep original selection rule as-is.
    selected_translation = R0_index if norm_R0 > norm_R2 else R2_index
    print(selected_translation)

    euler_angles = {
        0: (X0, Y0, Z0),
        1: (X1, Y1, Z1),
        2: (X2, Y2, Z2),
        3: (X3, Y3, Z3),
    }

    X4, Y4, Z4 = euler_angles[selected_translation]
    return X4, Y4, Z4


def compute_homography_diff_um(points_src, points_dst, H, S):
    """
    Apply homography and compute mean Euclidean error in micrometers (um).

    Assumption in original code:
      - One grid cell corresponds to 0.6 mm (600 um)
      - S is the measured cell length in the same unit as points

    Args:
        points_src: (N, 2) numpy array
        points_dst: DataFrame with columns ['X','Y']
        H: (3, 3) homography matrix
        S: cell length scale (used to convert to um)

    Returns:
        mean_distance_um: Series or scalar
    """
    ones = np.ones((points_src.shape[0], 1))
    homogenous_points_src = np.hstack([points_src, ones])

    um = 600 / S

    transformed_points = np.dot(H, homogenous_points_src.T).T
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2][:, np.newaxis]

    Error_dst = (points_dst - transformed_points) * um

    df = pd.DataFrame()
    df['Distance'] = np.sqrt(Error_dst['X'] ** 2 + Error_dst['Y'] ** 2)

    df = np.mean(df)
    return df


def align_points_to_grid(data, target_rows=9, target_cols=12, y_threshold=50, x_threshold=30):
    """
    Assign unordered points to a (target_rows x target_cols) grid by clustering rows (Y)
    and then splitting columns (X). Pads missing points with NaN.

    NOTE: Must be applied after homography alignment, per original comment.

    Args:
        data: (N, 2) numpy array
        target_rows: number of rows
        target_cols: number of cols
        y_threshold: row grouping threshold
        x_threshold: column split threshold

    Returns:
        final_data: (target_rows*target_cols, 2) numpy array (with NaNs for missing)
    """
    data_sorted = data[np.argsort(data[:, 1])]
    rows_list = []
    current_row = [data_sorted[0]]

    # Group into rows by Y proximity
    for i in range(1, len(data_sorted)):
        prev_y = current_row[-1][1]
        curr_point = data_sorted[i]
        curr_y = curr_point[1]

        if abs(curr_y - prev_y) > y_threshold or len(current_row) >= target_cols * 2:
            rows_list.append(np.array(current_row))
            current_row = [curr_point]
        else:
            current_row.append(curr_point)

    if current_row:
        rows_list.append(np.array(current_row))

    padded_rows = []
    for row in rows_list:
        row_sorted = row[np.argsort(row[:, 0])]
        cols = []
        current_col = [row_sorted[0]]

        for i in range(1, len(row_sorted)):
            prev_x = current_col[-1][0]
            curr_point = row_sorted[i]
            curr_x = curr_point[0]

            if abs(curr_x - prev_x) > x_threshold or len(current_col) >= target_cols:
                if len(cols) >= target_cols:
                    break
                cols.extend(current_col)
                current_col = [curr_point]
            else:
                current_col.append(curr_point)

        cols.extend(current_col)

        if len(cols) > target_cols:
            cols = cols[:target_cols]
        elif len(cols) < target_cols:
            missing = target_cols - len(cols)

            xs = np.array(cols)[:, 0]
            global_x_mean = np.nanmean(np.array(data)[:, 0])
            tol = 1e-9

            left_count = int(np.sum(xs < global_x_mean - tol))
            right_count = int(np.sum(xs > global_x_mean + tol))
            mid_count = len(xs) - left_count - right_count

            left_target = target_cols // 2
            right_target = target_cols - left_target

            need_left = max(0, left_target - left_count)
            need_right = max(0, right_target - right_count)

            add_left = min(need_left, missing)
            missing -= add_left

            add_right = min(need_right, missing)
            missing -= add_right

            add_left_extra = missing // 2
            add_right_extra = missing - add_left_extra

            padL = np.full((add_left + add_left_extra, 2), np.nan) if (add_left + add_left_extra) > 0 else np.empty((0, 2))
            padR = np.full((add_right + add_right_extra, 2), np.nan) if (add_right + add_right_extra) > 0 else np.empty((0, 2))

            cols = np.vstack([padL, cols, padR])

        padded_rows.append(np.array(cols))

    while len(padded_rows) < target_rows:
        padding_row = np.full((target_cols, 2), np.nan)
        padded_rows.append(padding_row)

    if len(padded_rows) > target_rows:
        padded_rows = padded_rows[:target_rows]

    final_data = np.vstack(padded_rows)

    assigned_points = np.vstack([row[~np.isnan(row[:, 0])] for row in padded_rows])

    if assigned_points.shape[0] < data.shape[0]:
        print("⚠️ Some points were not assigned due to thresholds. Consider adjusting thresholds or reviewing data.")


    return final_data


def align_points_to_grid_Y_minus(data, target_rows=9, target_cols=12, y_threshold=50, x_threshold=30):
    """
    Same as align_points_to_grid, but sorts by descending Y first.

    Args:
        data: (N, 2) numpy array
        target_rows: number of rows
        target_cols: number of cols
        y_threshold: row grouping threshold
        x_threshold: column split threshold

    Returns:
        final_data: (target_rows*target_cols, 2) numpy array (with NaNs for missing)
    """
    data_sorted = data[np.argsort(-data[:, 1])]
    rows_list = []
    current_row = [data_sorted[0]]

    for i in range(1, len(data_sorted)):
        prev_y = current_row[-1][1]
        curr_point = data_sorted[i]
        curr_y = curr_point[1]

        if abs(curr_y - prev_y) > y_threshold or len(current_row) >= target_cols * 2:
            rows_list.append(np.array(current_row))
            current_row = [curr_point]
        else:
            current_row.append(curr_point)

    if current_row:
        rows_list.append(np.array(current_row))

    padded_rows = []
    for row in rows_list:
        row_sorted = row[np.argsort(row[:, 0])]
        cols = []
        current_col = [row_sorted[0]]

        for i in range(1, len(row_sorted)):
            prev_x = current_col[-1][0]
            curr_point = row_sorted[i]
            curr_x = curr_point[0]

            if abs(curr_x - prev_x) > x_threshold or len(current_col) >= target_cols:
                if len(cols) >= target_cols:
                    break
                cols.extend(current_col)
                current_col = [curr_point]
            else:
                current_col.append(curr_point)

        cols.extend(current_col)

        if len(cols) > target_cols:
            cols = cols[:target_cols]
        elif len(cols) < target_cols:
            missing = target_cols - len(cols)

            xs = np.array(cols)[:, 0]
            global_x_mean = np.nanmean(np.array(data)[:, 0])
            tol = 1e-9

            left_count = int(np.sum(xs < global_x_mean - tol))
            right_count = int(np.sum(xs > global_x_mean + tol))
            mid_count = len(xs) - left_count - right_count

            left_target = target_cols // 2
            right_target = target_cols - left_target

            need_left = max(0, left_target - left_count)
            need_right = max(0, right_target - right_count)

            add_left = min(need_left, missing)
            missing -= add_left

            add_right = min(need_right, missing)
            missing -= add_right

            add_left_extra = missing // 2
            add_right_extra = missing - add_left_extra

            padL = np.full((add_left + add_left_extra, 2), np.nan) if (add_left + add_left_extra) > 0 else np.empty((0, 2))
            padR = np.full((add_right + add_right_extra, 2), np.nan) if (add_right + add_right_extra) > 0 else np.empty((0, 2))

            cols = np.vstack([padL, cols, padR])

        padded_rows.append(np.array(cols))

    while len(padded_rows) < target_rows:
        padding_row = np.full((target_cols, 2), np.nan)
        padded_rows.append(padding_row)

    if len(padded_rows) > target_rows:
        padded_rows = padded_rows[:target_rows]

    final_data = np.vstack(padded_rows)

    assigned_points = np.vstack([row[~np.isnan(row[:, 0])] for row in padded_rows])

    if assigned_points.shape[0] < data.shape[0]:
        print("⚠️ Some points were not assigned due to thresholds. Consider adjusting thresholds or reviewing data.")
        print(final_data)

    return final_data
