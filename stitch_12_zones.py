"""
Robust stitching for 12-zone plate images where the reference bar appears horizontal.

Dependencies:
    pip install opencv-contrib-python-headless numpy

Author: ChatGPT (2025)
"""

import cv2
import numpy as np
import os

# -----------------------------------------------------------
# 1. DETECT THE HORIZONTAL BAR
# -----------------------------------------------------------
def detect_horizontal_bar(gray):
    """
    Detects the dark horizontal bar used as stitching reference.
    Returns:
        mask  - binary mask of detected bar
        cy    - vertical centroid of the detected bar
    """

    # Horizontal gradient
    grad = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=5)
    grad = cv2.convertScaleAbs(grad)

    # Threshold on dark regions
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 8
    )

    comb = cv2.bitwise_or(thr, grad)

    # Close horizontally
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    closed = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find largest horizontal contour
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1

    for c in cnts:
        x,y,wc,hc = cv2.boundingRect(c)
        aspect = wc / max(hc,1)       # horizontal aspect ratio
        area = cv2.contourArea(c)
        score = area * aspect         # prefer long horizontal regions

        if score > best_score:
            best_score = score
            best = c

    if best is None:
        raise ValueError("Could not detect horizontal bar")

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [best], -1, 255, -1)

    M = cv2.moments(best)
    cy = int(M["m01"] / (M["m00"] + 1e-6))

    return mask, cy


# -----------------------------------------------------------
# 2. ORIENT IMAGE BASED ON CAMERA ARROW
# -----------------------------------------------------------
def orient_image(img, arrow_direction):
    """
    arrow_direction ∈ {"up","down","left","right"} meaning where the camera was facing.
    """

    if arrow_direction == "up":
        return img
    if arrow_direction == "down":
        return cv2.rotate(img, cv2.ROTATE_180)
    if arrow_direction == "left":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if arrow_direction == "right":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    return img


# -----------------------------------------------------------
# 3. SIFT FEATURE MATCH ALIGNMENT
# -----------------------------------------------------------
def align_images(base, moving):
    sift = cv2.SIFT_create()

    g1 = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)

    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d2, d1, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.65 * n.distance:
            good.append(m)

    if len(good) < 8:
        print("[WARN] Not enough good matches, using identity transform")
        return np.eye(3, dtype=np.float32)

    src = np.float32([k2[m.queryIdx].pt for m in good])
    dst = np.float32([k1[m.trainIdx].pt for m in good])

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        H = np.eye(3, dtype=np.float32)

    return H


# -----------------------------------------------------------
# 4. STITCH A SEQUENCE OF ZONES
# -----------------------------------------------------------
def stitch_sequence(images):
    """
    Images must be pre-rotated (camera direction accounted for).
    """

    # Step 1: normalize width
    target_w = max(img.shape[1] for img in images)
    imgs = []
    for img in images:
        scale = target_w / img.shape[1]
        imgs.append(cv2.resize(img, None, fx=scale, fy=scale))

    # Start canvas
    canvas = imgs[0]

    for i in range(1, len(imgs)):
        print(f"Stitching {i}/{len(imgs)-1}...")

        base = canvas
        mov  = imgs[i]

        gray_base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        gray_mov  = cv2.cvtColor(mov,  cv2.COLOR_BGR2GRAY)

        # Horizontal bar-based alignment
        _, cy_base = detect_horizontal_bar(gray_base)
        _, cy_mov  = detect_horizontal_bar(gray_mov)

        dy = cy_base - cy_mov

        T = np.array([
            [1, 0, 0],
            [0, 1, dy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Refine with SIFT
        H_feat = align_images(base, mov)
        H = T @ H_feat

        h,w = base.shape[:2]
        hh,ww = mov.shape[:2]

        big_w = w + ww
        big_h = max(h, hh + abs(dy))

        warped_base = cv2.warpPerspective(base, np.eye(3), (big_w, big_h))
        warped_mov  = cv2.warpPerspective(mov,  H, (big_w, big_h))

        # Merge using overwrite where mov is nonzero
        mask = np.any(warped_mov > 0, axis=2)
        warped_base[mask] = warped_mov[mask]

        canvas = warped_base

    return canvas


# -----------------------------------------------------------
# 5. MAIN PIPELINE
# -----------------------------------------------------------

# Fill these with your real paths + directions
ZONE_PATHS = [
    # ("1.1_back.jpg", "down"),   # example
]


def main():
    imgs = []
    for path, arrow in ZONE_PATHS:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)

        img = orient_image(img, arrow)
        imgs.append(img)

    stitched = stitch_sequence(imgs)
    cv2.imwrite("/mnt/data/stitched_output_horizontal.png", stitched)
    print("Saved → /mnt/data/stitched_output_horizontal.png")


if __name__ == "__main__":
    main()
