import cv2
import numpy as np
import os

# ------------------------------------------------
#  SETTINGS
# ------------------------------------------------

INPUT_DIR = "/mnt/data/zones"       # Folder containing zone images
OUTPUT_FULL = "/mnt/data/final_full.png"
OUTPUT_SMALL = "/mnt/data/final_small.png"

# Downsample factor for preview
PREVIEW_SCALE = 0.25

# Feature detection resolution (scales down input for keypoints only)
FEATURE_SCALE = 0.5


# ------------------------------------------------
#  HELPER FUNCTIONS
# ------------------------------------------------

def load_for_features(path):
    """Loads image at reduced resolution ONLY for feature detection."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise Exception("Failed to load: " + path)
    if FEATURE_SCALE != 1.0:
        img = cv2.resize(img, None, fx=FEATURE_SCALE, fy=FEATURE_SCALE)
    return img


def load_full(path):
    """Loads full-resolution image for final warping."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise Exception("Failed to load full-res: " + path)
    return img


def detect_and_match(imgA, imgB):
    """Detect and match SIFT (or ORB) features."""
    sift = cv2.SIFT_create()

    kA, dA = sift.detectAndCompute(imgA, None)
    kB, dB = sift.detectAndCompute(imgB, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(dA, dB, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    ptsA = np.float32([kA[m.queryIdx].pt for m in good])
    ptsB = np.float32([kB[m.trainIdx].pt for m in good])

    return ptsA, ptsB


def compute_homography(ptsA, ptsB):
    H, _ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5.0)
    return H


# ------------------------------------------------
#  MAIN STITCHING LOGIC
# ------------------------------------------------

def stitch_zones():
    # List all images in order
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    paths = [os.path.join(INPUT_DIR, f) for f in files]

    print("Found", len(paths), "zone images.")

    # Load first image (full-res) and use it as the base
    base_full = load_full(paths[0])

    # Estimate a very large canvas: 3× width × 3× height
    H0, W0 = base_full.shape[:2]
    CANVAS_H = H0 * 3
    CANVAS_W = W0 * 3

    print("Allocating canvas:", CANVAS_W, "×", CANVAS_H)
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    # Place the first image in the center
    offset_x = W0
    offset_y = H0
    canvas[offset_y:offset_y + H0, offset_x:offset_x + W0] = base_full

    # Accumulated global homography
    H_global = np.eye(3)

    prev_features = load_for_features(paths[0])

    for i in range(1, len(paths)):
        print(f"\nProcessing tile {i+1}/{len(paths)}…")

        # Load low-res version for matching
        next_features = load_for_features(paths[i])

        # Detect and match
        ptsA, ptsB = detect_and_match(prev_features, next_features)

        # Compute relative homography (low-res)
        H_rel = compute_homography(ptsB, ptsA)

        # Scale homography back to full resolution coordinates
        s = 1.0 / FEATURE_SCALE
        S = np.diag([s, s, 1.0])
        H_rel_full = S @ H_rel @ np.linalg.inv(S)

        # Update global homography
        H_global = H_global @ H_rel_full

        # Load full resolution image
        full = load_full(paths[i])

        # Warp directly into the big canvas
        cv2.warpPerspective(
            full,
            H_global,
            (CANVAS_W, CANVAS_H),
            dst=canvas,
            borderMode=cv2.BORDER_TRANSPARENT
        )

        prev_features = next_features

    return canvas


# ------------------------------------------------
#  SAVE OUTPUTS
# ------------------------------------------------

def main():
    final_canvas = stitch_zones()

    # Crop empty borders
    gray = cv2.cvtColor(final_canvas, cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    coords = np.argwhere(mask)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    final_crop = final_canvas[y0:y1+1, x0:x1+1]

    print("Saving FULL resolution:", OUTPUT_FULL)
    cv2.imwrite(OUTPUT_FULL, final_crop)

    # Make preview
    small = cv2.resize(final_crop, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
    print("Saving PREVIEW:", OUTPUT_SMALL)
    cv2.imwrite(OUTPUT_SMALL, small)

    print("Done.")


if __name__ == "__main__":
    main()
