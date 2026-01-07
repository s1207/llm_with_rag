"""
Hybrid stitching:
- Vertical alignment (X): '+' markers
- Horizontal alignment (Y): dark horizontal lines

Layout:
    [ 1 | 2 ]
    [ 3 | 4 ]
    [ 5 | 6 ]
"""

import cv2
import numpy as np


# ============================================================
# Rotation utilities
# ============================================================

def rotate_image(img, angle):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=0)


def rotate_90(img, k):
    return np.rot90(img, k).copy()


# ============================================================
# Horizontal dark line detection
# ============================================================

def detect_horizontal_dark_lines(gray):
    """
    Detect strong horizontal dark lines.
    Returns list of Y positions.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # emphasize horizontal structures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
    dark = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)

    # project vertically
    proj = np.mean(dark, axis=1)

    # normalize & invert
    proj = (proj.max() - proj)
    proj /= proj.max() + 1e-6

    # threshold
    line_idx = np.where(proj > 0.5)[0]

    if len(line_idx) == 0:
        return []

    # group contiguous indices
    groups = np.split(line_idx, np.where(np.diff(line_idx) > 2)[0] + 1)
    centers = [int(g.mean()) for g in groups if len(g) > 5]

    return centers


# ============================================================
# Plus marker detection
# ============================================================

def detect_plus_markers(gray):
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        51, 5
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (15, 15))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if 0.7 < w / float(h) < 1.3:
            centers.append((x + w / 2, y + h / 2))

    return centers


# ============================================================
# Tile loader
# ============================================================

def load_tile(path, rot=None):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(path)

    if rot:
        if "k" in rot:
            img = rotate_90(img, rot["k"])
        if "deg" in rot:
            img = rotate_image(img, rot["deg"])

    return img


# ============================================================
# Stitching logic
# ============================================================

def stitch_tiles(tile_paths, rotations=None):
    tiles = {}
    markers = {}
    hlines = {}

    for z, path in tile_paths.items():
        tiles[z] = load_tile(path, rotations.get(z) if rotations else None)
        markers[z] = detect_plus_markers(tiles[z])
        hlines[z] = detect_horizontal_dark_lines(tiles[z])

        if len(markers[z]) < 1:
            raise RuntimeError(f"Zone {z}: no '+' markers detected")
        if len(hlines[z]) < 1:
            raise RuntimeError(f"Zone {z}: no horizontal lines detected")

    h, w = tiles[1].shape
    canvas = np.zeros((h * 3, w * 2), dtype=np.uint8)

    # Placement dictionary
    offsets = {}

    # ---- Reference tile (zone 1)
    offsets[1] = (w // 2, h // 2)

    # ---- Horizontal neighbors (X alignment via markers)
    def align_x(left, right):
        lx = np.mean([m[0] for m in markers[left]])
        rx = np.mean([m[0] for m in markers[right]])
        return int(lx - rx)

    # ---- Vertical neighbors (Y alignment via dark lines)
    def align_y(top, bottom):
        ty = np.median(hlines[top])
        by = np.median(hlines[bottom])
        return int(ty - by)

    # zone 2 (right of 1)
    dx = align_x(1, 2)
    offsets[2] = (offsets[1][0] + w + dx, offsets[1][1])

    # zone 3 (below 1)
    dy = align_y(1, 3)
    offsets[3] = (offsets[1][0], offsets[1][1] + h + dy)

    # zone 4 (right of 3)
    dx = align_x(3, 4)
    offsets[4] = (offsets[3][0] + w + dx, offsets[3][1])

    # zone 5 (below 3)
    dy = align_y(3, 5)
    offsets[5] = (offsets[3][0], offsets[3][1] + h + dy)

    # zone 6 (right of 5)
    dx = align_x(5, 6)
    offsets[6] = (offsets[5][0] + w + dx, offsets[5][1])

    # ---- Paste tiles
    for z, img in tiles.items():
        x, y = offsets[z]
        canvas[y:y+h, x:x+w] = np.maximum(
            canvas[y:y+h, x:x+w], img
        )

    return canvas


# ============================================================
# Example run
# ============================================================

if __name__ == "__main__":

    tile_paths = {
        1: "zone1.png",
        2: "zone2.png",
        3: "zone3.png",
        4: "zone4.png",
        5: "zone5.png",
        6: "zone6.png",
    }

    rotations = {
        # 3: {"k": 1},
        # 6: {"deg": -0.8},
    }

    stitched = stitch_tiles(tile_paths, rotations)
    cv2.imwrite("stitched_hybrid.png", stitched)

    print("Hybrid stitching completed -> stitched_hybrid.png")
