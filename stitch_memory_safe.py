import cv2
import numpy as np
import os

DOWNSCALE = 0.25   # use low-res copy for homography only

# ============================================================
# 1. Detect horizontal reference bar (low memory)
# ============================================================
def detect_horizontal_bar(gray):
    grad = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=5)
    grad = cv2.convertScaleAbs(grad)

    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 31, 8
    )

    comb = cv2.bitwise_or(thr, grad)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,7))
    closed = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, kernel, 2)

    cnts,_ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        aspect = w / max(h,1)
        area = cv2.contourArea(c)
        score = area * aspect
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        raise RuntimeError("Horizontal bar not detected")

    M = cv2.moments(best)
    cy = int(M["m01"] / (M["m00"]+1e-6))
    return cy


# ============================================================
# 2. Compute approximate transform (downsampled)
# ============================================================
def compute_transform(base_full, mov_full):
    base = cv2.resize(base_full, None, fx=DOWNSCALE, fy=DOWNSCALE)
    mov  = cv2.resize(mov_full,  None, fx=DOWNSCALE, fy=DOWNSCALE)

    g1 = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(mov,  cv2.COLOR_BGR2GRAY)

    cy1 = detect_horizontal_bar(g1)
    cy2 = detect_horizontal_bar(g2)

    dy_low = cy1 - cy2
    dy_full = dy_low / DOWNSCALE   # rescale to full resolution

    # Simple vertical translation
    H = np.array([
        [1, 0, 0],
        [0, 1, dy_full],
        [0, 0, 1]
    ], dtype=np.float32)

    return H


# ============================================================
# 3. Memory-mapped canvas stitching (no huge RAM use)
# ============================================================
def stitch_large(images):
    # Start canvas size with first image
    h0, w0 = images[0].shape[:2]

    # Make a memmapped giant canvas on disk
    canvas_w = sum(img.shape[1] for img in images)
    canvas_h = max(img.shape[0] for img in images)

    mmap_path = "/mnt/data/stitch_memmap.npy"
    canvas = np.memmap(mmap_path, dtype=np.uint8, mode='w+',
                       shape=(canvas_h, canvas_w, 3))

    # Clear
    canvas[:] = 0

    # Place first image
    offset_x = 0
    canvas[0:h0, offset_x:offset_x+w0] = images[0]
    offset_x += w0

    # Stitch remaining images one by one
    for i in range(1, len(images)):
        print(f"Stitching image {i+1}/{len(images)} ...")
        base = images[i-1]
        mov  = images[i]

        H = compute_transform(base, mov)

        # Warp mov into local tile to avoid large arrays
        h,w = canvas.shape[:2]

        # Only warp a small width tile for mov
        tile_w = mov.shape[1] + 200
        tile_h = canvas_h

        warped = cv2.warpPerspective(mov, H, (tile_w, tile_h))

        # Paste at offset
        canvas[0:tile_h, offset_x:offset_x+tile_w] = np.where(
            warped>0, warped, canvas[0:tile_h, offset_x:offset_x+tile_w]
        )

        offset_x += mov.shape[1]

    canvas.flush()
    return mmap_path, canvas_w


# ============================================================
# MAIN
# ============================================================
def main():
    # Fill with actual file paths + camera orientations applied beforehand
    IMAGE_PATHS = [
        # "/mnt/data/1.1_back.jpg",
        # ...
    ]

    images = [cv2.imread(p) for p in IMAGE_PATHS]
    mmap_path, final_width = stitch_large(images)

    print("Memory-mapped output saved at:", mmap_path)
    print("To export as PNG, load in chunks or crop.")


if __name__ == "__main__":
    main()
