import numpy as np
from numpy.linalg import svd, inv, norm
from scipy.optimize import least_squares

# ============================================================
# Import dependent utility functions
# ============================================================

from lib.L1_Image_Conversion import *
from lib.L2_Point_Detection_Conversion import *
from lib.L3_Zhang_Camera_Calibration import *
from lib.L4_Pipeline_Utilities import *
from lib.L5_Visualization_Utilities import *

# ============================================================
# 1) Utilities: normalization (Hartley), homography (Normalized DLT)
# ============================================================

def normalize_points_2d(pts):
    """
    Normalize 2D points with Hartley normalization.
    pts: (N,2)
    returns: pts_norm (N,2), T (3,3)
    """
    pts = np.asarray(pts, dtype=np.float64)
    mean = pts.mean(axis=0)
    pts_centered = pts - mean
    d = np.sqrt((pts_centered[:, 0] ** 2 + pts_centered[:, 1] ** 2)).mean()
    if d < 1e-12:
        s = 1.0
    else:
        s = np.sqrt(2.0) / d

    T = np.array([
        [s, 0, -s * mean[0]],
        [0, s, -s * mean[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    pts_h = np.column_stack([pts, np.ones(len(pts))])
    pts_n = (T @ pts_h.T).T
    pts_n = pts_n[:, :2] / pts_n[:, 2:3]
    return pts_n, T


def compute_homography_normalized_dlt(src_pts, dst_pts):
    """
    Compute homography H mapping src_pts -> dst_pts using normalized DLT.
    src_pts: (N,2), dst_pts: (N,2)
    returns: H (3,3) with H[2,2] = 1
    """
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)

    src_n, T_src = normalize_points_2d(src_pts)
    dst_n, T_dst = normalize_points_2d(dst_pts)

    N = src_n.shape[0]
    A = np.zeros((2 * N, 9), dtype=np.float64)

    for i in range(N):
        x, y = src_n[i]
        u, v = dst_n[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, x * u, y * u, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * v, y * v, v] ### by Cross product of A

    _, _, Vt = svd(A)
    Hn = Vt[-1].reshape(3, 3)

    # Denormalize
    H = inv(T_dst) @ Hn @ T_src
    if abs(H[2, 2]) < 1e-12:
        H = H / (np.sign(H[2, 2]) * 1e-12)
    else:
        H = H / H[2, 2]
    return H


# ============================================================
# 2) Zhang closed-form: solve for K from multiple homographies
# ============================================================

def v_ij(H, i, j):
    """
    Construct v_ij vector used in Zhang method for B = K^{-T} K^{-1}.
    H: (3,3)
    i,j: column indices 0..2
    returns: (6,)
    """
    h = H
    return np.array([
        h[0, i] * h[0, j],
        h[0, i] * h[1, j] + h[1, i] * h[0, j],
        h[1, i] * h[1, j],
        h[2, i] * h[0, j] + h[0, i] * h[2, j],
        h[2, i] * h[1, j] + h[1, i] * h[2, j],
        h[2, i] * h[2, j]
    ], dtype=np.float64)


def K_from_B(B):
    """
    Recover intrinsic K from B = K^{-T} K^{-1} (Zhang).
    B is symmetric (3,3).
    returns: K (3,3)
    """
    B11, B12, B13 = B[0, 0], B[0, 1], B[0, 2]
    B22, B23 = B[1, 1], B[1, 2]
    B33 = B[2, 2]

    denom = (B11 * B22 - B12 * B12)

    if abs(denom) < 1e-18: ###
        raise ValueError("Degenerate B (cannot recover K).")

    cy = (B12 * B13 - B11 * B23) / denom
    lam = B33 - (B13 * B13 + cy * (B12 * B13 - B11 * B23)) / B11
    if lam < 0:
        # Numerical issue can make lambda slightly negative
        lam = abs(lam)

    fx = np.sqrt(lam / B11)
    fy = np.sqrt(lam * B11 / denom)

    cx = -B13 * fx * fx / lam


    # Skew is typically assumed 0 in many calibrations
    s = 0.0

    K = np.array([
        [fx, s, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return K

def init_intrinsic_like_opencv(H_list, image_size, aspect_ratio=0.0):
    """
    OpenCV initIntrinsicParams2D-style fallback.
    Parameters
    ----------
    H_list : list of (3,3) homographies
        Homographies from plane -> image (object -> image)
    image_size : (w, h)
        Image size in pixels
    aspect_ratio : float
        fx/fy if fixed, 0 means free

    Returns
    -------
    K_init : (3,3) ndarray
        Initial intrinsic matrix
    """

    w, h = image_size
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5

    nimages = len(H_list)
    if nimages == 0:
        raise ValueError("No homographies for fallback init")

    # Each image gives 2 equations, unknowns are u=1/fx^2, v=1/fy^2
    A = np.zeros((2 * nimages, 2), dtype=np.float64)
    b = np.zeros((2 * nimages, 1), dtype=np.float64)

    for i, H in enumerate(H_list):
        H = H.astype(np.float64).copy()

        # shift principal point to origin
        H[0, :] -= H[2, :] * cx
        H[1, :] -= H[2, :] * cy

        h = H[:, 0]
        v = H[:, 1]
        d1 = 0.5 * (h + v)
        d2 = 0.5 * (h - v)

        # normalize directions (same as OpenCV)
        def normalize(vec):
            return vec / np.linalg.norm(vec)

        h = normalize(h)
        v = normalize(v)
        d1 = normalize(d1)
        d2 = normalize(d2)

        # h ⟂ v
        A[2 * i + 0, 0] = h[0] * v[0]
        A[2 * i + 0, 1] = h[1] * v[1]
        b[2 * i + 0, 0] = -h[2] * v[2]

        # d1 ⟂ d2
        A[2 * i + 1, 0] = d1[0] * d2[0]
        A[2 * i + 1, 1] = d1[1] * d2[1]
        b[2 * i + 1, 0] = -d1[2] * d2[2]

    # Solve [A][u v]^T = b using SVD (least squares)
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    u, v = sol.flatten()

    # Recover focal lengths
    from math import sqrt
    fx = sqrt(abs(1.0 / u))
    fy = sqrt(abs(1.0 / v))

    # Optional aspect ratio constraint
    if aspect_ratio > 0:
        tf = (fx + fy) / (aspect_ratio + 1.0)
        fx = aspect_ratio * tf
        fy = tf

    K_init = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return K_init

def zhang_closed_form(object_pts_xy, image_pts_list):
    """
    Zhang closed-form initialization:
    1) Estimate homography per view (planar target).
    2) Solve for intrinsic K.
    3) Solve extrinsics (R,t) per view.

    object_pts_xy: (M,2) planar points (X,Y)
    image_pts_list: list of (M,2) arrays, one per image
    returns:
      K_init (3,3)
      rvecs_init: (N,3) Rodrigues vectors
      tvecs_init: (N,3)
    """
    object_pts_xy = np.asarray(object_pts_xy, dtype=np.float64)
    N = len(image_pts_list)


    # Homographies H mapping object plane -> image
    H_list = []
    for pts_img in image_pts_list:
        pts_img = np.asarray(pts_img, dtype=np.float64)
        H = compute_homography_normalized_dlt(object_pts_xy, pts_img)
        H_list.append(H)


    # Build V b = 0

    V = []
    for H in H_list:
        V.append(v_ij(H, 0, 1))
        V.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))
    V = np.vstack(V)

    _, _, Vt = svd(V)
    b = Vt[-1]  # smallest singular vector

    # Construct symmetric B
    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ], dtype=np.float64)

    # 1) sign fix: make B33 positive (or make B11 positive)
    if B[2, 2] < 0:
        B = -B

    # 2) scale fix: normalize so that B33 = 1
    B = B / B[2, 2]

    try:
        K_init = K_from_B(B)  # Zhang
    except ValueError:

        print("image position is too samll")
        K_init = init_intrinsic_like_opencv(
            H_list=H_list,
            image_size=(4000, 3000),
            aspect_ratio=0.0
        )
        used_init = "OpenCV-fallback"

    # Extrinsics per view: H = K [r1 r2 t]
    Kinv = inv(K_init)
    rvecs = []
    tvecs = []

    for H in H_list:
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]

        lam = 1.0 / norm(Kinv @ h1)
        r1 = lam * (Kinv @ h1)
        r2 = lam * (Kinv @ h2)
        r3 = np.cross(r1, r2)
        t = lam * (Kinv @ h3)

        # Orthonormalize R with SVD (closest rotation)
        R = np.column_stack([r1, r2, r3])
        U, _, VtR = svd(R) ### here error??
        R = U @ VtR
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ VtR

        rvec = rodrigues_from_R(R)
        rvecs.append(rvec)
        tvecs.append(t)

    return K_init, np.vstack(rvecs), np.vstack(tvecs)


# ============================================================
# 3) Rodrigues helpers
# ============================================================

def R_from_rodrigues(rvec):
    """
    Rodrigues formula: rvec (3,) -> R (3,3)
    """
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = norm(rvec)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)

    k = rvec / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ], dtype=np.float64)

    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R


def rodrigues_from_R(R):
    """
    Convert rotation matrix R -> Rodrigues vector rvec (3,)
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(trace)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)

    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    r = np.array([rx, ry, rz], dtype=np.float64)
    rvec = (theta / (2.0 * np.sin(theta))) * r
    return rvec


# ============================================================
# 4) Projection with Brown distortion (k1,k2,p1,p2,k3)
# ============================================================

def project_points(object_pts_xyz, K, dist, rvec, tvec):
    """
    Project 3D points to 2D with Brown distortion.
    object_pts_xyz: (M,3)
    K: (3,3)
    dist: (5,) -> [k1,k2,p1,p2,k3]
    rvec: (3,)
    tvec: (3,)
    returns: (M,2)
    """
    object_pts_xyz = np.asarray(object_pts_xyz, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(4)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2= dist ##k3

    R = R_from_rodrigues(rvec)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)

    # Camera coordinates
    Xc = (R @ object_pts_xyz.T).T + tvec[None, :]
    X = Xc[:, 0]
    Y = Xc[:, 1]
    Z = Xc[:, 2]

    # Avoid division by zero
    Z = np.where(np.abs(Z) < 1e-12, 1e-12, Z)

    # Normalized coords
    x = X / Z
    y = Y / Z

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    radial = 1.0 + k1 * r2 + k2 * r4 ####+ k3 * r6

    # Tangential
    x_t = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_t = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    x_d = x * radial + x_t
    y_d = y * radial + y_t

    # Pixel coords
    u = fx * x_d + cx
    v = fy * y_d + cy

    return np.column_stack([u, v])


# ============================================================
# 5) LM refinement: bundle adjustment style
# ============================================================

def pack_params(K, dist, rvecs, tvecs):
    """
    Pack parameters into a 1D vector for optimization.
    Intrinsic: fx, fy, cx, cy (4)
    Dist: k1,k2,p1,p2,k3 (5)
    Extrinsics per view: rvec(3) + tvec(3) -> 6N
    """

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p = [fx, fy, cx, cy]
    p.extend(list(dist.reshape(-1)))
    p.extend(list(rvecs.reshape(-1)))
    p.extend(list(tvecs.reshape(-1)))
    return np.array(p, dtype=np.float64)


def unpack_params(p, n_views):
    """
    Unpack 1D parameter vector into K, dist, rvecs, tvecs.
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    dist = p[4:8]  ##if use k3 use [4:9]
    idx = 8
    rvecs = p[idx: idx + 3 * n_views].reshape(n_views, 3)
    idx += 3 * n_views
    tvecs = p[idx: idx + 3 * n_views].reshape(n_views, 3)

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    return K, dist, rvecs, tvecs


def reprojection_residuals(p, object_pts_xyz, image_pts_list):
    """
    Compute stacked reprojection residuals for all views and points.
    Returns (2 * total_points,) residual vector.
    """
    n_views = len(image_pts_list)
    K, dist, rvecs, tvecs = unpack_params(p, n_views)

    residuals = []
    for i in range(n_views):
        proj = project_points(object_pts_xyz, K, dist, rvecs[i], tvecs[i])
        obs = np.asarray(image_pts_list[i], dtype=np.float64)
        residuals.append((proj - obs).reshape(-1))
    return np.concatenate(residuals, axis=0)


def calibrate_zhang_then_lm(image_pts_list, grid_size=(12, 9), spacing=1.0):
    """
    Full pipeline:
    1) Closed-form (Zhang): estimate K and extrinsics (no distortion)
    2) LM refinement: optimize K(4) + dist(5) + extrinsics(6N)

    image_pts_list: list of (M,2) arrays, M = grid_size[0]*grid_size[1]
                    Each element corresponds to one calibration image.
    grid_size: (cols, rows) -> e.g., (9,12)
    spacing: physical spacing between adjacent points (any unit)
    returns:
      K_opt (3,3), dist_opt (5,), rvecs_opt (N,3), tvecs_opt (N,3), rmse_pixels
    """
    cols, rows = grid_size
    M = cols * rows

    # Build ideal planar object points (Z=0)
    # Note: scaling spacing affects translation scale only, not intrinsic/dist in theory.
    obj_xy = np.array([(c * spacing, r * spacing) for r in range(rows) for c in range(cols)], dtype=np.float64)
    obj_xyz = np.column_stack([obj_xy, np.zeros(M, dtype=np.float64)])

    # Zhang closed-form initialization
    K_init, rvecs_init, tvecs_init = zhang_closed_form(obj_xy, image_pts_list)

    # Distortion init = zeros
    dist_init = np.zeros(4, dtype=np.float64)

    # Pack params
    p0 = pack_params(K_init, dist_init, rvecs_init, tvecs_init)

    # LM optimization (Levenberg–Marquardt)
    # Note: method='lm' requires number of residuals >= number of params, typically satisfied.
    ### least squares mean
    res = least_squares(
        fun=reprojection_residuals,
        x0=p0,
        args=(obj_xyz, image_pts_list),
        method="lm"
    )

    K_opt, dist_opt, rvecs_opt, tvecs_opt = unpack_params(res.x, len(image_pts_list))

    # RMSE in pixels
    r = reprojection_residuals(res.x, obj_xyz, image_pts_list)
    rmse = np.sqrt(np.mean(r * r))

    return K_opt, dist_opt, rvecs_opt, tvecs_opt, float(rmse), K_init


# ============================================================
# 6) Example usage (you replace image_pts_list with your detected corners)
# ============================================================

if __name__ == "__main__":
    # Example placeholder:
    # image_pts_list should be a list of arrays, each (108,2) for 9*12 grid.
    # Fill it with your detected points (e.g., from cv2.findCirclesGrid or your detector).
    image_pts_list = []  # <-- put your per-image points here

    # Uncomment after you fill image_pts_list:
    # K, dist, rvecs, tvecs, rmse, K_init = calibrate_zhang_then_lm(
    #     image_pts_list=image_pts_list,
    #     grid_size=(9, 12),
    #     spacing=1.0
    # )
    # print("K:\n", K)
    # print("dist [k1,k2,p1,p2,k3]:", dist)
    # print("RMSE (pixels):", rmse)
    pass
