"""
Marker-based stitching for 6 tiled images using '+' fiducial markers.

Layout (reference):
    [ 1 | 2 ]
    [ 3 | 4 ]
    [ 5 | 6 ]

Assumptions:
- Each tile contains exactly 2 '+' markers
- Markers overlap with neighboring tiles
- Only rotation + translation (no scale change)
- Images are grayscale or convertible to grayscale

Dependencies:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import os


# ------------------------------------------------------------
# Rotation utilities
# ------------------------------------------------------------

def rotate_image(image, angle_deg):
    """Rotate image by arbitrary angle around its center"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def rotate_90(image, k):
    """
    Rotate image by multiples of 90 degrees.
    k = 0,1,2,3 -> 0°, 90°, 180°, 270°
    """
    return np.rot90(image, k).copy()


# ------------------------------------------------------------
# Marker detection
# ------------------------------------------------------------

def detect_plus_markers(gray):
    """
    Detect '+' fiducial markers.
    Returns list of (x, y) centers.
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        51, 5
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (15, 15))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)

        if 0.7 < aspect < 1.3:
            cx = x + w / 2
            cy = y + h / 2
            centers.append((cx, cy))

    return centers


# ------------------------------------------------------------
# Transform estimation
# ------------------------------------------------------------

def estimate_rigid_transform(src_pts, dst_pts):
    """
    Estimate rotation + translation between two sets of marker points.
    """
    src_pts = np.asarray(src_pts, dtype=np.float32)
    dst_pts = np.asarray(dst_pts, dtype=np.float32)

    if len(src_pts) < 2:
        raise RuntimeError("At least 2 markers required")

    M, _ = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0
    )

    return M


# ------------------------------------------------------------
# Tile loading
# ------------------------------------------------------------

def load_tile(path, rotate_deg=None, rotate_90_k=None):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Failed to load image: {path}")

    if rotate_90_k is not None:
        img = rotate_90(img, rotate_90_k)

    if rotate_deg is not None:
        img = rotate_image(img, rotate_deg)

    return img


# ------------------------------------------------------------
# Stitching pipeline
# ------------------------------------------------------------

def stitch_tiles(tile_paths, rotations=None):
    """
    tile_paths: dict {zone_id: filepath}
    rotations: dict {zone_id: {"deg": float OR "k": int}}
    """

    tiles = {}
    markers = {}

    # Load tiles and detect markers
    for zone, path in tile_paths.items():
        rot = rotations.get(zone, {}) if rotations else {}
        tiles[zone] = load_tile(path, rot.get("deg"), rot.get("k"))
        markers[zone] = detect_plus_markers(tiles[zone])

        if len(markers[zone]) < 2:
            raise RuntimeError(f"Zone {zone}: insufficient markers detected")

    # Base canvas size
    h, w = tiles[1].shape
    canvas = np.zeros((h * 3, w * 2), dtype=np.uint8)

    # Reference transform (zone 1 placed in center-top)
    transforms = {
        1: np.array([[1, 0, w // 2],
                     [0, 1, h // 2]], dtype=np.float32)
    }

    # Order based on diagram adjacency
    order = [2, 3, 4, 5, 6]

    for z in order:
        for ref in transforms:
            if len(markers[z]) == len(markers[ref]):
                M = estimate_rigid_transform(
                    markers[z],
                    markers[ref]
                )
                M[:, 2] += transforms[ref][:, 2]
                transforms[z] = M
                break

        if z not in transforms:
            raise RuntimeError(f"Could not align zone {z}")

    # Warp tiles into canvas
    for z, img in tiles.items():
        warped = cv2.warpAffine(
            img,
            transforms[z],
            (canvas.shape[1], canvas.shape[0]),
            flags=cv2.INTER_LINEAR
        )
        canvas = np.maximum(canvas, warped)

    return canvas


# ------------------------------------------------------------
# Example execution
# ------------------------------------------------------------

if __name__ == "__main__":

    tile_paths = {
        1: "zone1.png",
        2: "zone2.png",
        3: "zone3.png",
        4: "zone4.png",
        5: "zone5.png",
        6: "zone6.png",
    }

    # Optional rotations per tile
    rotations = {
        # 3: {"k": 1},        # rotate 90°
        # 6: {"deg": -1.0},   # small correction
    }

    stitched = stitch_tiles(tile_paths, rotations)

    cv2.imwrite("stitched_result.png", stitched)
    print("Stitching completed -> stitched_result.png")
