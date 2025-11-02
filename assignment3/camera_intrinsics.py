"""
Camera intrinsics utilities and reusable stereo/SfM helpers.

Drop this file alongside your assignment notebook and import the functions you
need. The module provides:

1. Chessboard-based calibration for recovering your phone's intrinsic matrix.
2. Convenience loaders that fall back to an approximate FOV-based K.
3. Stereo preprocessing/matching utilities that accept arbitrary intrinsics.
4. A minimal SfM pipeline that works with calibrated or approximate cameras.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

DEFAULT_FOV_DEGREES = 60.0
DEFAULT_CACHE_PATH = Path("assignment3/phone_intrinsics.npz")


@dataclass(frozen=True)
class IntrinsicsBundle:
    """Container for calibration results."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


def estimate_intrinsics_from_fov(
    image_shape: Tuple[int, int],
    fov_degrees: float = DEFAULT_FOV_DEGREES,
) -> IntrinsicsBundle:
    """
    Approximate camera intrinsics by assuming a symmetric field of view.

    Parameters
    ----------
    image_shape:
        Height, width tuple for the input image.
    fov_degrees:
        Horizontal field of view assumption for the camera.
    """
    height, width = image_shape
    fov_radians = np.deg2rad(fov_degrees)
    focal_length = max(width, height) / (2.0 * np.tan(fov_radians / 2.0))

    camera_matrix = np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros(5, dtype=np.float64)
    return IntrinsicsBundle(camera_matrix, dist_coeffs)


def estimate_phone_intrinsics(
    calibration_images: Iterable[Path],
    pattern_size: Tuple[int, int] = (9, 6),
    square_size_m: float = 0.024,
    cache_path: Optional[Path] = DEFAULT_CACHE_PATH,
) -> IntrinsicsBundle:
    """
    Estimate intrinsics from a set of chessboard calibration photos.

    Parameters
    ----------
    calibration_images:
        Iterable of filesystem paths to chessboard views captured with your phone.
    pattern_size:
        Number of inner corners (columns, rows) on the chessboard.
    square_size_m:
        Physical size of one square edge in meters.
    cache_path:
        Optional .npz file location for caching the calibration result.
    """
    calibration_images = list(calibration_images)
    if not calibration_images:
        raise ValueError("Provide at least one chessboard image for calibration.")

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_m

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    gray_ref: Optional[np.ndarray] = None

    for path in calibration_images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            continue

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            40,
            1e-5,
        )
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(refined)
        gray_ref = gray

    if not objpoints:
        raise RuntimeError("Failed to detect the calibration pattern in the images.")

    assert gray_ref is not None
    ret, camera_matrix, dist_coeffs, *_ = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray_ref.shape[::-1],
        None,
        None,
    )
    if not ret:
        raise RuntimeError("Camera calibration did not converge.")

    bundle = IntrinsicsBundle(camera_matrix, dist_coeffs)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            pattern_size=pattern_size,
            square_size=square_size_m,
        )
    return bundle


def load_intrinsics(
    calibration_images: Optional[Iterable[Path]] = None,
    fallback_shape: Optional[Tuple[int, int]] = None,
    cache_path: Path = DEFAULT_CACHE_PATH,
    fov_degrees: float = DEFAULT_FOV_DEGREES,
) -> IntrinsicsBundle:
    """
    Load cached intrinsics, run calibration if images are supplied, or fall back
    to an FOV-based guess.
    """
    if calibration_images:
        return estimate_phone_intrinsics(calibration_images, cache_path=cache_path)

    if cache_path.exists():
        cached = np.load(cache_path)
        return IntrinsicsBundle(cached["camera_matrix"], cached["dist_coeffs"])

    if fallback_shape is None:
        raise ValueError("fallback_shape is required when no calibration data exists.")

    return estimate_intrinsics_from_fov(fallback_shape, fov_degrees)


def make_scaled_intrinsics(
    camera_matrix: np.ndarray,
    fx_scale: float = 1.0,
    fy_scale: float = 1.0,
    cx_offset: float = 0.0,
    cy_offset: float = 0.0,
) -> np.ndarray:
    """Perturb an intrinsic matrix for sensitivity experiments."""
    scaled = camera_matrix.copy()
    scaled[0, 0] *= fx_scale
    scaled[1, 1] *= fy_scale
    scaled[0, 2] += cx_offset
    scaled[1, 2] += cy_offset
    return scaled


# ---------------------------------------------------------------------------
# Stereo helpers
# ---------------------------------------------------------------------------

STEREO_FRAME_SIZE = (756, 567)
DEFAULT_BASELINE_METERS = 5.5 * 0.0254  # convert inches to meters


def preprocess_stereo_pair(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    intrinsics: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    frame_size: Tuple[int, int] = STEREO_FRAME_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Undistort and resize stereo pair, returning grayscale frames."""
    left_gray = cv2.cvtColor(left_rgb, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rgb, cv2.COLOR_BGR2GRAY)

    if intrinsics is not None:
        dist = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        left_gray = cv2.undistort(left_gray, intrinsics, dist)
        right_gray = cv2.undistort(right_gray, intrinsics, dist)

    left_gray = cv2.resize(left_gray, frame_size)
    right_gray = cv2.resize(right_gray, frame_size)
    return left_gray, right_gray


def match_stereo_keypoints(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    max_visualized: int = 50,
):
    """Return ORB keypoints, descriptors, and cross-checked matches."""
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(left_gray, None)
    kp2, des2 = orb.detectAndCompute(right_gray, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)

    match_vis = cv2.drawMatches(
        left_gray,
        kp1,
        right_gray,
        kp2,
        matches[:max_visualized],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return kp1, kp2, matches, match_vis


def estimate_fundamental_and_essential(
    kp1: Sequence[cv2.KeyPoint],
    kp2: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
):
    """
    Estimate the fundamental matrix (and essential matrix when K is supplied).
    """
    if not matches:
        raise ValueError("Need matches to estimate epipolar geometry.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    if camera_matrix is not None:
        dist = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        pts1 = cv2.undistortPoints(
            pts1.reshape(-1, 1, 2), camera_matrix, dist, P=camera_matrix
        ).reshape(-1, 2)
        pts2 = cv2.undistortPoints(
            pts2.reshape(-1, 1, 2), camera_matrix, dist, P=camera_matrix
        ).reshape(-1, 2)

    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.999,
    )
    if F is None or F.shape != (3, 3):
        raise RuntimeError("Fundamental matrix estimation failed.")

    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]
    E = None
    if camera_matrix is not None:
        E = camera_matrix.T @ F @ camera_matrix

    return inliers1, inliers2, F, E


def compute_disparity_and_depth(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    camera_matrix: Optional[np.ndarray] = None,
    baseline_m: float = DEFAULT_BASELINE_METERS,
    num_disparities: int = 256,
    block_size: int = 9,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute a disparity map (and optional depth map) from a stereo pair."""
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

    depth = None
    if camera_matrix is not None and baseline_m > 0:
        with np.errstate(divide="ignore"):
            depth = (camera_matrix[0, 0] * baseline_m) / disparity
        depth[~np.isfinite(depth)] = np.nan

    return disparity, depth


# ---------------------------------------------------------------------------
# Structure from motion helpers
# ---------------------------------------------------------------------------


def extract_and_match_features(
    img1: np.ndarray,
    img2: np.ndarray,
    ratio: float = 0.75,
    flann_trees: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """SIFT feature extraction with ratio-tested FLANN matching."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    index_params = dict(algorithm=1, trees=flann_trees)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    pts1, pts2 = [], []
    for best, second in matcher.knnMatch(des1, des2, k=2):
        if best.distance < ratio * second.distance:
            pts1.append(kp1[best.queryIdx].pt)
            pts2.append(kp2[best.trainIdx].pt)

    return (
        np.array(pts1, dtype=np.float32),
        np.array(pts2, dtype=np.float32),
    )


def undistort_sequence(
    images: Sequence[np.ndarray],
    intrinsics: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray],
) -> list[np.ndarray]:
    """Undistort all images in a sequence (if intrinsics are provided)."""
    if intrinsics is None:
        return list(images)
    dist = dist_coeffs if dist_coeffs is not None else np.zeros(5)
    return [cv2.undistort(img, intrinsics, dist) for img in images]


def run_sfm(
    images: Sequence[np.ndarray],
    intrinsics: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    visualize_epipolar: bool = False,
):
    """
    Minimal sequential SfM pipeline that accepts optional intrinsics.

    Returns
    -------
    trajectory:
        List of 4x4 poses (camera-to-world) for each processed image.
    structure:
        List of 3D point batches reconstructed between consecutive frames.
    """
    if len(images) < 2:
        raise ValueError("Need at least two images to run structure from motion.")

    rectified = undistort_sequence(images, intrinsics, dist_coeffs)
    trajectory = [np.eye(4, dtype=np.float64)]
    structure: list[np.ndarray] = []

    for idx in range(len(rectified) - 1):
        img1, img2 = rectified[idx], rectified[idx + 1]

        pts1, pts2 = extract_and_match_features(img1, img2)
        if len(pts1) < 8 or len(pts2) < 8:
            continue

        F, mask_F = cv2.findFundamentalMat(
            pts1,
            pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=1.0,
            confidence=0.999,
        )
        if F is None or F.shape != (3, 3):
            continue

        inliers1 = pts1[mask_F.ravel() == 1]
        inliers2 = pts2[mask_F.ravel() == 1]
        if len(inliers1) < 8:
            continue

        if intrinsics is not None:
            E = intrinsics.T @ F @ intrinsics
            recover_kwargs = dict(cameraMatrix=intrinsics)
        else:
            E, _ = cv2.findEssentialMat(
                inliers1,
                inliers2,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0,
            )
            if E is None:
                continue
            recover_kwargs = dict(focal=1.0, pp=(0.0, 0.0))

        _, R, t, mask_pose = cv2.recoverPose(
            E,
            inliers1,
            inliers2,
            **recover_kwargs,
        )

        last_pose = trajectory[-1]
        current_pose = np.eye(4, dtype=np.float64)
        current_pose[:3, :3] = R
        current_pose[:3, 3] = t.ravel()
        trajectory.append(last_pose @ current_pose)

        pose_inliers1 = inliers1[mask_pose.ravel() == 1]
        pose_inliers2 = inliers2[mask_pose.ravel() == 1]
        if len(pose_inliers1) < 8:
            continue

        K = intrinsics if intrinsics is not None else np.eye(3)
        proj1 = K @ np.eye(3, 4)
        proj2 = K @ np.hstack((R, t))

        pts4d = cv2.triangulatePoints(proj1, proj2, pose_inliers1.T, pose_inliers2.T)
        pts3d = (pts4d / pts4d[3])[:3].T

        homogeneous_pts = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
        pts_cam2 = (np.hstack((R, t)) @ homogeneous_pts.T).T

        valid = (pts3d[:, 2] > 0) & (pts_cam2[:, 2] > 0)
        if valid.any():
            structure.append(pts3d[valid])

    return trajectory, structure


__all__ = [
    "IntrinsicsBundle",
    "estimate_intrinsics_from_fov",
    "estimate_phone_intrinsics",
    "load_intrinsics",
    "make_scaled_intrinsics",
    "preprocess_stereo_pair",
    "match_stereo_keypoints",
    "estimate_fundamental_and_essential",
    "extract_and_match_features",
    "compute_disparity_and_depth",
    "run_sfm",
]
