"""
Stitch four large zones using a central horizontal bar reference.

Dependencies:
    pip install opencv-python-headless numpy

If you want SIFT (better):
    pip install opencv-contrib-python-headless
"""

import cv2
import numpy as np
import math
from collections import defaultdict, deque

# -------------------------
# Helper utilities
# -------------------------

def load_image(path, gray=True):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if gray:
        if img.ndim == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()
        return img, img_gray
    else:
        return img

def enhance_contrast(img_gray):
    # CLAHE for local contrast improvement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    return clahe.apply(img_gray)

# -------------------------
# Bar detection
# -------------------------

def detect_horizontal_bar(img_gray, debug=False):
    """
    Return (bar_mask, bar_center_y (float), bar_bbox (x,y,w,h))
    bar_mask: binary mask of detected bar region
    """
    # 1) Preprocess
    g = enhance_contrast(img_gray)
    # 2) Edge/gradient
    grad = cv2.Sobel(g, cv2.CV_16S, 0, 1, ksize=5)
    grad = cv2.convertScaleAbs(grad)
    # 3) Adaptive threshold to get dark bar streaks
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)
    # 4) Combine gradient + threshold, morphological closing to fill bar
    comb = cv2.bitwise_or(th, grad)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,3))
    closed = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 5) Find long horizontal contours / connected components and choose the longest horizontal blob
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img_gray.shape
    best = None
    best_score = -1
    for c in contours:
        x,y,ww,hh = cv2.boundingRect(c)
        area = ww*hh
        # Score favors width and horizontal aspect
        score = ww - 3*abs(hh - max(6, h//60)) + area/10000.0
        # Prefer near center vertically
        cy = y + hh/2.0
        center_penalty = -abs(cy - h/2.0)/h*50
        score += center_penalty
        if ww < w*0.05:  # skip very narrow components
            continue
        if score > best_score:
            best_score = score
            best = (x,y,ww,hh)
    if best is None:
        # fallback: threshold horizontal projection
        proj = np.mean(closed, axis=1)
        # find long region where projection above small threshold
        thresh = np.max(proj)*0.15
        mask_1d = proj > thresh
        starts = []
        end = None
        for i,val in enumerate(mask_1d):
            if val and end is None:
                start = i
                end = i
            elif not val and end is not None:
                starts.append((start,end))
                end = None
        if end is not None:
            starts.append((start,end))
        # pick largest
        if len(starts)==0:
            # absolute fallback: center row only
            cy = h//2
            x,y,ww,hh = 0,cy-4,w,8
            mask = np.zeros_like(img_gray, dtype=np.uint8)
            mask[y:y+hh, x:x+ww] = 255
            return mask, cy, (x,y,ww,hh)
        start,end = max(starts, key=lambda x: x[1]-x[0])
        y = max(0, start-3)
        hh = min(h, end - start + 6)
        x = 0
        ww = w
        mask = np.zeros_like(img_gray, dtype=np.uint8)
        mask[y:y+hh, x:x+ww] = 255
        return mask, (start+end)/2.0, (x,y,ww,hh)

    x,y,ww,hh = best
    mask = np.zeros_like(img_gray, dtype=np.uint8)
    # expand slightly vertically to ensure full bar captured
    padv = max(4, hh//3)
    ya = max(0, y-padv)
    yb = min(h, y+hh+padv)
    mask[ya:yb, x:x+ww] = 255
    cy = (ya+yb)/2.0
    if debug:
        print("Bar bbox:", (x,ya,ww,yb-ya))
    return mask, cy, (x,ya,ww,yb-ya)

# -------------------------
# Pairwise alignment using bar ROI
# -------------------------

def init_feature_detector():
    # Prefer SIFT if available (better), fallback to ORB.
    try:
        sift = cv2.SIFT_create()
        def detect_and_compute(img):
            kps, des = sift.detectAndCompute(img, None)
            return kps, des
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        return detect_and_compute, matcher
    except Exception:
        orb = cv2.ORB_create(4000)
        def detect_and_compute(img):
            kps, des = orb.detectAndCompute(img, None)
            return kps, des
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        return detect_and_compute, matcher

detect_and_compute, matcher = init_feature_detector()

def match_bar_regions(imgA_gray, imgB_gray, maskA, maskB, debug=False):
    # Restrict to bar regions for matching
    roiA = cv2.bitwise_and(imgA_gray, imgA_gray, mask=maskA)
    roiB = cv2.bitwise_and(imgB_gray, imgB_gray, mask=maskB)
    # Detect features
    kpsA, desA = detect_and_compute(roiA)
    kpsB, desB = detect_and_compute(roiB)
    if desA is None or desB is None or len(kpsA) < 4 or len(kpsB) < 4:
        if debug:
            print("Not enough features in bar ROIs, falling back to whole-image features")
        # fallback to whole image
        kpsA, desA = detect_and_compute(imgA_gray)
        kpsB, desB = detect_and_compute(imgB_gray)
        if desA is None or desB is None:
            return None
    # knn match
    raw_matches = matcher.knnMatch(desA, desB, k=2)
    good = []
    # ratio test
    for m_n in raw_matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 6:
        # relax ratio
        good = [m_n[0] for m_n in raw_matches if len(m_n)>=1]
    if len(good) < 4:
        return None
    ptsA = np.float32([kpsA[m.queryIdx].pt for m in good])
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in good])
    # estimate homography/affine with RANSAC
    M, mask = cv2.estimateAffinePartial2D(ptsA, ptsB, method=cv2.RANSAC, ransacReprojThreshold=6.0)
    if M is None:
        # fallback to homography
        H, maskH = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 6.0)
        if H is None:
            return None
        # convert homography to affine approx for warpAffine if possible
        if abs(H[2,0])<1e-6 and abs(H[2,1])<1e-6:
            M = H[0:2,:]
        else:
            # return full homography
            return {'H': H, 'inliers': int(np.sum(maskH))}
    # refine using ECC on the bar ROI (for subpixel)
    try:
        # warp imgA to imgB initial using M
        rowsB, colsB = imgB_gray.shape
        warp_init = cv2.warpAffine(imgA_gray, M, (colsB, rowsB))
        # compute refinement over bar region intersection
        mask_inter = cv2.bitwise_and(maskA, maskB)
        if np.count_nonzero(mask_inter) > 500:
            # prepare float images
            im1 = warp_init.astype(np.float32)/255.0
            im2 = imgB_gray.astype(np.float32)/255.0
            # ECC requires same size; use mask for weighting via mask cropping
            # Use findTransformECC with motion model AFFINE
            warp_mat = np.eye(2,3,dtype=np.float32)
            # initialize warp_mat from M relative (we used warpAffine earlier)
            warp_mat = M.astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
            try:
                cc, warp_mat = cv2.findTransformECC(im2, cv2.warpAffine(imgA_gray, warp_mat, (colsB, rowsB)).astype(np.float32),
                                                    warp_mat, cv2.MOTION_AFFINE, criteria, mask_inter)
                M = warp_mat
            except cv2.error:
                pass
    except Exception:
        pass
    return {'M': M, 'inliers': int(np.sum(mask)) if isinstance(mask, np.ndarray) else None}

# -------------------------
# Compose global layout from pairwise transforms
# -------------------------

def build_global_layout(image_paths, imgs_gray, masks, pairwise_results, debug=False):
    """
    Build global transforms that map each image into a common canvas.
    We pick one image as root (first), then BFS through best pairwise transforms (M or H).
    Each result in pairwise_results is a dict keyed by (i,j) -> result mapping image i -> image j.
    """
    n = len(image_paths)
    # adjacency list with transform from i to j
    adj = defaultdict(list)
    for (i,j),res in pairwise_results.items():
        if res is None:
            continue
        adj[i].append((j,res))
        # invert transform for j->i
        # handle affine M 2x3
        if 'M' in res and res['M'] is not None:
            M = res['M']
            A = M[:, :2]; t = M[:,2:]
            Ainv = np.linalg.inv(A)
            Minv = np.hstack([Ainv, -Ainv.dot(t)])
            adj[j].append((i, {'M':Minv, 'inliers':res.get('inliers')}))
        elif 'H' in res and res['H'] is not None:
            H = res['H']
            Hin = np.linalg.inv(H)
            adj[j].append((i, {'H':Hin, 'inliers':res.get('inliers')}))
    # BFS from node 0 to compute global transforms to root canvas
    root = 0
    transforms = {root: np.eye(3,3,dtype=np.float32)}  # homogeneous 3x3 transform mapping image coords to canvas coords
    visited = set([root])
    q = deque([root])
    while q:
        i = q.popleft()
        for j,res in adj[i]:
            if j in visited:
                continue
            # transform from j -> i known in adj? We stored res as mapping i->j in adjacency
            # We need transform_j_to_root = transform_i_to_root * transform_j_to_i
            if 'M' in res and res['M'] is not None:
                M_ij = res['M']  # maps i -> j (or j->i depending how stored). We stored earlier adjacency so res maps i->j
                # we need mapping from j to i. But above when building adj we inserted both directions.
                # For safety, if res is mapping j->i, we convert to homography
                A = M_ij[:, :2]; t = M_ij[:,2:]
                H_ij = np.eye(3,3,dtype=np.float32)
                H_ij[:2,:2] = A
                H_ij[:2,2:3] = t
            elif 'H' in res and res['H'] is not None:
                H_ij = res['H'].astype(np.float32)
            else:
                continue
            # Now compute global transform for j.
            transforms[j] = transforms[i] @ np.linalg.inv(H_ij)
            visited.add(j)
            q.append(j)
    return transforms

# -------------------------
# Stitching & blending
# -------------------------

def create_canvas_and_warp(images_color, transforms):
    # compute bounding boxes of all warped images to build canvas size
    corners = []
    for i, img in enumerate(images_color):
        h,w = img.shape[:2]
        H = transforms[i]
        # corners of image in homogeneous coords
        pts = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]], dtype=np.float32).T
        warped = H @ pts
        warped = warped / warped[2,:]
        corners.append(warped[:2,:].T)
    all_pts = np.vstack(corners)
    min_xy = np.min(all_pts, axis=0)
    max_xy = np.max(all_pts, axis=0)
    min_x, min_y = min_xy
    max_x, max_y = max_xy
    # canvas size
    canvas_w = int(math.ceil(max_x - min_x))
    canvas_h = int(math.ceil(max_y - min_y))
    # offset transform to shift coordinates to positive
    offset = np.eye(3,3,dtype=np.float32)
    offset[0,2] = -min_x
    offset[1,2] = -min_y
    # Warp each image into canvas
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    for i, img in enumerate(images_color):
        H = offset @ transforms[i]
        h,w = img.shape[:2]
        # warp color image
        warped = cv2.warpPerspective(img.astype(np.float32), H, (canvas_w, canvas_h))
        # create mask for where image contributes
        mask = np.any(warped>0, axis=2).astype(np.float32)
        # linear feathering on edges: compute distance transform to mask boundary
        kernel = np.ones((15,1), np.uint8)
        gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # create soft mask in source then warp it
        src_mask = np.ones((h,w), dtype=np.float32)
        src_mask = cv2.distanceTransform((src_mask*255).astype(np.uint8), cv2.DIST_L2,5)
        src_mask = src_mask.astype(np.float32)
        # normalize and clip
        src_mask = src_mask / (src_mask.max()+1e-6)
        warped_mask = cv2.warpPerspective(src_mask, H, (canvas_w, canvas_h))
        # Accumulate weighted blending
        for c in range(3):
            canvas[:,:,c] += warped[:,:,c] * warped_mask
        weight += warped_mask
    # normalize
    eps = 1e-6
    res = np.zeros_like(canvas, dtype=np.uint8)
    for c in range(3):
        channel = canvas[:,:,c] / (weight + eps)
        channel = np.nan_to_num(channel)
        channel = np.clip(channel, 0, 255)
        res[:,:,c] = channel.astype(np.uint8)
    return res, (canvas_w, canvas_h), (-min_x, -min_y)

# -------------------------
# Public high-level stitch function
# -------------------------

def stitch_zones(image_paths, debug=False):
    # load images
    images_color = []
    images_gray = []
    masks = []
    bboxes = []
    centers = []
    for p in image_paths:
        color, gray = load_image(p, gray=True)
        images_color.append(color if color.ndim==3 else cv2.cvtColor(color, cv2.COLOR_GRAY2BGR))
        images_gray.append(gray)
        mask, cy, bbox = detect_horizontal_bar(gray, debug=debug)
        masks.append(mask)
        bboxes.append(bbox)
        centers.append(cy)
        if debug:
            print("Detected bar center y (px) for {}: {:.1f}, bbox: {}".format(p, cy, bbox))
    n = len(image_paths)
    # compute pairwise matches for all pairs
    pairwise_results = {}
    for i in range(n):
        for j in range(i+1, n):
            if debug:
                print(f"Matching {i} <-> {j}")
            res = match_bar_regions(images_gray[i], images_gray[j], masks[i], masks[j], debug=debug)
            pairwise_results[(i,j)] = res
            pairwise_results[(j,i)] = res  # we'll invert later when building graph
            if debug:
                print(" ->", "OK" if res else "NO_MATCH")
    # build global layout
    transforms = build_global_layout(image_paths, images_gray, masks, pairwise_results, debug=debug)
    # convert transforms to 3x3 homogeneous matrices
    transforms_h = {}
    for i in range(n):
        if i in transforms:
            transforms_h[i] = transforms[i]
        else:
            transforms_h[i] = np.eye(3,3,dtype=np.float32)  # fallback
    # warp & compose
    stitched, (W,H), offset = create_canvas_and_warp(images_color, transforms_h)
    return stitched, transforms_h, (W,H), offset

# -------------------------
# Measurement utilities
# -------------------------

def measure_dimensions(stitched_img, scale_pixels_per_mm=None):
    h,w = stitched_img.shape[:2]
    dims = {'width_px': w, 'height_px': h}
    if scale_pixels_per_mm is not None:
        dims['width_mm'] = w / scale_pixels_per_mm
        dims['height_mm'] = h / scale_pixels_per_mm
    return dims

# -------------------------
# Example usage with provided sample files
# -------------------------

if __name__ == "__main__":
    # Replace these with your four zone filenames. The two that you uploaded:
    sample_paths = [
        "/mnt/data/IMG_2508.jpeg",   # example bottom/top zone 1
        "/mnt/data/image0000025.png",
        "/mnt/data/image0000026.png",
        # add the 4th zone path here when available
    ]

    # If you only have 2 sample images, the script will still attempt to match them.
    stitched, transforms, (W,H), offset = stitch_zones(sample_paths, debug=True)

    print("Stitched image size (px):", W, H)
    # If you know a physical scale: e.g., you measured a known reference that is X mm long and corresponds to P pixels,
    # you can compute pixels_per_mm = P / X and pass to measurement.
    # Example: pixels_per_mm = 10.0  (you must compute/measure)
    dims = measure_dimensions(stitched, scale_pixels_per_mm=None)
    print("Measured dims:", dims)

    # Save result
    cv2.imwrite("/mnt/data/stitched_result.png", stitched)
    print("Saved /mnt/data/stitched_result.png")

    # Also save a visualization of detected bars per input (debug)
    for idx,p in enumerate(sample_paths):
        col = cv2.imread(p)
        if col is None:
            continue
        # overlay mask (if available)
        try:
            mask, _, bbox = detect_horizontal_bar(cv2.cvtColor(col, cv2.COLOR_BGR2GRAY))
            overlay = col.copy()
            overlay[mask>0] = (0,0,255)
            vis = cv2.addWeighted(col, 0.7, overlay, 0.3, 0)
            cv2.imwrite(f"/mnt/data/detected_bar_{idx}.png", vis)
        except Exception:
            pass
    print("Also saved per-zone bar visualizations: /mnt/data/detected_bar_*.png")
