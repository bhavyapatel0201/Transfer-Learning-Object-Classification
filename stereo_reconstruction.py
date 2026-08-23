#!/usr/bin/env python3
"""
Two-View Stereo 3D Reconstruction Pipeline
============================================
Calibrates a camera from checkerboard images, decomposes the projection
matrix into intrinsics via RQ factorization, matches SIFT features between
two stereo photos, computes F and E, recovers R/T, triangulates 3D points,
and saves a visualization + point cloud.

USAGE
-----
python stereo_reconstruction.py \
    --calib_dir ./calib_images \
    --board_cols 9 --board_rows 6 --square_size 25 \
    --img1 ./stereo/img1.jpg --img2 ./stereo/img2.jpg \
    --output_dir ./results

Requires: opencv-contrib-python, numpy, matplotlib
    pip install opencv-contrib-python numpy matplotlib
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def calibrate_camera(calib_dir, board_cols, board_rows, square_size):
    pattern_size = (board_cols, board_rows)

    objp = np.zeros((board_cols * board_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []
    img_size = None

    image_paths = sorted(
        glob.glob(os.path.join(calib_dir, "*.jpg"))
        + glob.glob(os.path.join(calib_dir, "*.jpeg"))
        + glob.glob(os.path.join(calib_dir, "*.png"))
    )

    if not image_paths:
        sys.exit(f"[ERROR] No calibration images found in {calib_dir}")

    used = 0
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Could not read {path}, skipping.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            print(f"[WARN] Chessboard not found in {path}, skipping.")
            continue

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners)
        used += 1

    if used < 3:
        sys.exit(
            f"[ERROR] Only {used} usable calibration images found. "
            "Need at least a handful of clear checkerboard shots."
        )

    print(f"[INFO] Using {used} calibration images.")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    print(f"[INFO] Calibration RMS reprojection error: {ret:.3f} px")
    print("[INFO] Intrinsic matrix K:\n", K)

    R0, _ = cv2.Rodrigues(rvecs[0])
    t0 = tvecs[0]
    Rt0 = np.hstack((R0, t0))
    M = K @ Rt0

    return K, dist, M, img_size, ret


def decompose_projection_matrix(M):
    A, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(M)
    A = A / A[2, 2]
    return A


def rq_decomposition_manual(Mblock):
    Q, R = np.linalg.qr(np.flipud(Mblock).T)
    R = np.flipud(R.T)
    Q = Q.T
    R = np.fliplr(R)
    Q = np.flipud(Q)
    R = R / R[2, 2]
    return R, Q


def detect_and_match_sift(img1, img2, output_dir):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    print(f"[INFO] Detected {len(kp1)} keypoints in img1, {len(kp2)} in img2.")

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"[INFO] {len(good_matches)} good matches after ratio test.")

    if len(good_matches) < 20:
        print(
            "[WARN] Very few good matches. Consider retaking photos with more "
            "texture/overlap between views."
        )

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    match_vis = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    match_path = os.path.join(output_dir, "matches.jpg")
    cv2.imwrite(match_path, match_vis)
    print(f"[INFO] Saved match visualization to {match_path}")

    return pts1, pts2


def compute_F_and_E(pts1, pts2, A):
    F, mask = cv2.findFundamentalMat(
        pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99
    )
    if F is None:
        sys.exit("[ERROR] Fundamental matrix estimation failed. Check your matches.")

    mask = mask.ravel().astype(bool)
    pts1_in = pts1[mask]
    pts2_in = pts2[mask]
    print(f"[INFO] Fundamental matrix inliers: {pts1_in.shape[0]} / {pts1.shape[0]}")

    E = A.T @ F @ A

    return F, E, pts1_in, pts2_in


def recover_pose(E, pts1_in, pts2_in, A):
    retval, R, T, pose_mask = cv2.recoverPose(E, pts1_in, pts2_in, A)
    pose_mask = pose_mask.ravel().astype(bool)
    print(f"[INFO] recoverPose kept {pose_mask.sum()} / {len(pose_mask)} points as valid.")
    return R, T, pose_mask


def build_projection_matrices(A, R, T):
    I = np.hstack((np.eye(3), np.zeros((3, 1))))
    M1 = A @ I
    D = np.hstack((R, T))
    M2 = A @ D
    return M1, M2


def triangulate_points(M1, M2, pts1, pts2):
    pts1_2xN = pts1.T
    pts2_2xN = pts2.T
    points_4d_hom = cv2.triangulatePoints(M1, M2, pts1_2xN, pts2_2xN)
    points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T
    return points_3d


def filter_outliers(points_3d, factor=5.0):
    dists = np.linalg.norm(points_3d, axis=1)
    median_dist = np.median(dists)
    keep = dists < factor * median_dist
    return points_3d[keep]


def save_point_cloud_ply(points_3d, path):
    n = points_3d.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in points_3d:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def visualize_and_save(points_3d, output_dir):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=3, c=points_3d[:, 2], cmap="viridis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Reconstructed 3D Point Cloud")
    png_path = os.path.join(output_dir, "reconstruction.png")
    plt.savefig(png_path, dpi=150)
    print(f"[INFO] Saved 3D scatter plot to {png_path}")

    ply_path = os.path.join(output_dir, "reconstruction.ply")
    save_point_cloud_ply(points_3d, ply_path)
    print(f"[INFO] Saved point cloud to {ply_path} (open with Open3D / MeshLab / CloudCompare)")


def main():
    parser = argparse.ArgumentParser(description="Two-view stereo 3D reconstruction pipeline")
    parser.add_argument("--calib_dir", required=True, help="Folder of checkerboard calibration images")
    parser.add_argument("--board_cols", type=int, default=9, help="Inner corners per row")
    parser.add_argument("--board_rows", type=int, default=6, help="Inner corners per column")
    parser.add_argument("--square_size", type=float, default=25.0, help="Checkerboard square size (e.g. mm)")
    parser.add_argument("--img1", required=True, help="First stereo image")
    parser.add_argument("--img2", required=True, help="Second stereo image")
    parser.add_argument("--output_dir", default="./results", help="Where to save all outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== STEP 1: Calibrating camera ===")
    K, dist, M, img_size, reproj_err = calibrate_camera(
        args.calib_dir, args.board_cols, args.board_rows, args.square_size
    )
    np.save(os.path.join(args.output_dir, "M.npy"), M)
    np.save(os.path.join(args.output_dir, "K.npy"), K)

    print("\n=== STEP 2: Extracting intrinsics A from M (RQ decomposition) ===")
    A = decompose_projection_matrix(M)
    print("Recovered A:\n", A)
    np.save(os.path.join(args.output_dir, "A.npy"), A)

    print("\n=== STEP 3: Detecting & matching SIFT features ===")
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    if img1 is None or img2 is None:
        sys.exit("[ERROR] Could not read one of the stereo images. Check paths.")
    pts1, pts2 = detect_and_match_sift(img1, img2, args.output_dir)

    print("\n=== STEP 4: Computing Fundamental and Essential matrices ===")
    F, E, pts1_in, pts2_in = compute_F_and_E(pts1, pts2, A)
    print("F:\n", F)
    print("E:\n", E)
    np.save(os.path.join(args.output_dir, "F.npy"), F)
    np.save(os.path.join(args.output_dir, "E.npy"), E)

    print("\n=== STEP 5: Recovering R, T from E ===")
    R, T, pose_mask = recover_pose(E, pts1_in, pts2_in, A)
    print("R:\n", R)
    print("T:\n", T)
    np.save(os.path.join(args.output_dir, "R.npy"), R)
    np.save(os.path.join(args.output_dir, "T.npy"), T)

    pts1_final = pts1_in[pose_mask]
    pts2_final = pts2_in[pose_mask]

    print("\n=== STEP 6: Building projection matrices M1, M2 ===")
    M1, M2 = build_projection_matrices(A, R, T)
    print("M1:\n", M1)
    print("M2:\n", M2)
    np.save(os.path.join(args.output_dir, "M1.npy"), M1)
    np.save(os.path.join(args.output_dir, "M2.npy"), M2)

    print("\n=== STEP 7: Triangulating 3D points ===")
    points_3d = triangulate_points(M1, M2, pts1_final, pts2_final)
    print(f"[INFO] Triangulated {points_3d.shape[0]} raw 3D points.")
    points_3d_clean = filter_outliers(points_3d)
    print(f"[INFO] {points_3d_clean.shape[0]} points remain after outlier filtering.")

    visualize_and_save(points_3d_clean, args.output_dir)

    print(f"\n[DONE] All results saved in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()