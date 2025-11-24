"""
Improved stitching using central horizontal bar reference + feature fallback.

Dependencies:
    pip install opencv-contrib-python-headless numpy

Notes:
 - I included the 3 sample files you uploaded as defaults. Replace/add the 4th path.
 - Outputs saved to /mnt/data/: stitched_result.png, debug images.
"""

import cv2
import numpy as np
import math
import os
from collections import defaultdict, deque

# -----------------------
# Input images (replace/add the 4th zone)
# -----------------------
IMAGE_PATHS = [
    "/mnt/data/IMG_2508.jpeg",      # sample zone
    "/mnt/data/image0000025.png",   # sample zone
    "/mnt/data/image0000026.png",   # sample zone
    # "/mnt/data/zone4.png"         # <-- add your 4th path here
]

OUT_DIR = "/mnt/data"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def imread_color_gray(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        gray = img.copy()
    else:
        color = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return color, gray

def save_dbg(name, img):
    cv2.imwrite(os.path.join(OUT_DIR, name), img)

def clahe(img_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    return clahe.apply(img_gray)

# -----------------------
# Bar detection (Hough + morphology + fallback)
# -----------------------
def detect_horizontal_bar_v2(img_gray, debug_prefix=""):
    """
    Returns: bar_mask (uint8), center_y (float), bbox (x,y,w,h)
    Strategy:
      - CLAHE -> Sobel (vertical gradient) -> threshold + morphological close (wide horizontal kernel)
      - HoughLinesP to detect long horizontal segments; choose best near image center
      - Fallback to projection method if Hough fails
    """
    h, w = img_gray.shape
    g = clahe(img_gray)
    # vertical gradient highlights horizontal edges (Sobel dy)
    sob = cv2.Sobel(g, cv2.CV_16S, 0, 1, ksize=5)
    sob = cv2.convertScaleAbs(sob)
    # adaptive threshold to pick dark horizontal bar
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)
    comb = cv2.bitwise_or(th, sob)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15,w//100), 3))
    closed = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Hough for long lines
    lines = cv2.HoughLinesP(closed, rho=1, theta=np.pi/180, threshold=max(100, w//30),
                             minLineLength=int(w*0.4), maxLineGap=50)
    if lines is not None:
        # choose line closest to vertical center and longest
        best = None
        best_score = -1
        for (x1,y1,x2,y2) in lines[:,0,:]:
            length = math.hypot(x2-x1, y2-y1)
            cy = (y1+y2)/2.0
            score = length - abs(cy - h/2.0)*0.5
            if score > best_score:
                best_score = score
                best = (x1,y1,x2,y2,length,cy)
        if best is not None:
            x1,y1,x2,y2,length,cy = best
            # create mask around that line (pad vertically)
            pad_v = max(6, int(h*0.01))
            ymin = max(0, int(min(y1,y2)-pad_v))
            ymax = min(h, int(max(y1,y2)+pad_v))
            xmin = max(0, int(min(x1,x2)-10))
            xmax = min(w, int(max(x1,x2)+10))
            mask = np.zeros_like(img_gray, dtype=np.uint8)
            mask[ymin:ymax, xmin:xmax] = 255
            bbox = (xmin, ymin, xmax-xmin, ymax-ymin)
            if debug_prefix:
                dbg = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(dbg, (xmin,ymin),(xmax,ymax),(0,0,255),2)
                save_dbg(f"{debug_prefix}_bar_hough.png", dbg)
            return mask, cy, bbox
    # fallback: horizontal projection of closed image
    proj = np.mean(closed, axis=1)
    thr = np.max(proj)*0.15
    mask1d = proj > thr
    segments = []
    start = None
    for i, val in enumerate(mask1d):
        if val and start is None:
            start = i
        elif not val and start is not None:
            segments.append((start, i-1))
            start = None
    if start is not None:
        segments.append((start, len(mask1d)-1))
    if segments:
        # choose largest segment that is near center
        best_seg = max(segments, key=lambda s: (s[1]-s[0]) - 0.5*abs(((s[0]+s[1])/2.0) - h/2.0))
        s0, s1 = best_seg
        pad_v = max(6, int((s1-s0)*0.2))
        ymin = max(0, s0-pad_v)
        ymax = min(h, s1+pad_v)
        mask = np.zeros_like(img_gray, dtype=np.uint8)
        mask[ymin:ymax, :] = 255
        cy = (ymin+ymax)/2.0
        bbox = (0, ymin, w, ymax-ymin)
        if debug_prefix:
            dbg = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(dbg, (0,ymin),(w,ymax),(0,255,0),2)
            save_dbg(f"{debug_prefix}_bar_proj.png", dbg)
        return mask, cy, bbox
    # absolute fallback
    cy = h//2
    mask = np.zeros_like(img_gray, dtype=np.uint8)
    mask[max(0,cy-8):min(h,cy+8), :] = 255
    return mask, cy, (0, max(0,cy-8), w, 16)

# -----------------------
# Template extraction + matching (robust translation)
# -----------------------
def extract_bar_template(gray, mask, bbox, max_template_width=1200):
    x,y,w,h = bbox
    # crop the bar center region horizontally (avoid edges where zone may be truncated)
    left = x + int(w*0.05)
    right = x + int(w*0.95)
    # reduce template width to manageable size for speed; sample center horizontally
    tw = min(max_template_width, right-left)
    cx = (left + right)//2
    tx0 = max(left, cx - tw//2)
    tx1 = tx0 + tw
    template = gray[y:y+h, tx0:tx1].copy()
    return template, (tx0, y, tw, h)

def match_template_multi_scale(template, target_gray, search_mask=None, scales=(0.5, 0.75, 1.0)):
    """
    Multi-scale normalized cross-correlation template matching.
    Returns best result: dict with keys: 'max_val', 'max_loc', 'scale'
    """
    t_h, t_w = template.shape
    best = None
    for scale in scales:
        # scale target down for speed
        tgt = cv2.resize(target_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if tgt.shape[0] < t_h or tgt.shape[1] < t_w:
            continue
        res = cv2.matchTemplate(tgt, template, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        # map maxLoc back to original coordinates
        max_loc_orig = (int(maxLoc[0]/scale), int(maxLoc[1]/scale))
        if best is None or maxVal > best['max_val']:
            best = {'max_val': float(maxVal), 'max_loc': max_loc_orig, 'scale': scale}
    return best

# -----------------------
# Feature-match fallback
# -----------------------
def init_detector():
    try:
        sift = cv2.SIFT_create()
        def dc(img):
            return sift.detectAndCompute(img, None)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        return dc, matcher
    except Exception:
        orb = cv2.ORB_create(5000)
        def dc(img):
            return orb.detectAndCompute(img, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        return dc, matcher

detect_and_compute, matcher = init_detector()

def feature_match_ransac(imgA, imgB, maskA=None, maskB=None):
    kpsA, desA = detect_and_compute(imgA if maskA is None else cv2.bitwise_and(imgA,imgA,mask=maskA))
    kpsB, desB = detect_and_compute(imgB if maskB is None else cv2.bitwise_and(imgB,imgB,mask=maskB))
    if desA is None or desB is None or len(kpsA)<4 or len(kpsB)<4:
        return None
    knn = matcher.knnMatch(desA, desB, k=2)
    good = []
    for m_n in knn:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 6:
        good = [m_n[0] for m_n in knn if len(m_n)>=1]
    if len(good) < 4:
        return None
    ptsA = np.float32([kpsA[m.queryIdx].pt for m in good])
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold=6.0)
    if H is None:
        return None
    inliers = int(np.sum(mask))
    return {'H': H, 'inliers': inliers}

# -----------------------
# Pairwise matching logic (template first, then features)
# -----------------------
def pairwise_align(imgs_gray, masks, bboxes, debug_prefixes):
    n = len(imgs_gray)
    results = {}
    # for every pair attempt: template match -> if low confidence then feature-match fallback
    for i in range(n):
        tmpl, tmpl_bbox = extract_bar_template(imgs_gray[i], masks[i], bboxes[i])
        debug_template_path = os.path.join(OUT_DIR, f"{debug_prefixes[i]}_template.png")
        cv2.imwrite(debug_template_path, tmpl)
        for j in range(n):
            if i==j:
                continue
            key = (i,j)
            # quick reject: if centers differ by a lot vertically, still try (zones might be offset)
            # Template match into j
            tm = match_template_multi_scale(tmpl, imgs_gray[j], scales=(0.5,0.75,1.0))
            if tm and tm['max_val'] > 0.72:
                # Good translation result. Build affine transform (translation only)
                tx_source = tmpl_bbox[0]
                ty_source = tmpl_bbox[1]
                src_x = tx_source
                src_y = ty_source
                # matched top-left location in target:
                tgt_x, tgt_y = tm['max_loc']
                # compute translation: map coordinates in image i -> j
                # we'll produce a 3x3 homography for consistency (translation)
                dx = tgt_x - src_x
                dy = tgt_y - src_y
                H = np.array([[1,0,dx],[0,1,dy],[0,0,1]], dtype=np.float64)
                results[key] = {'H': H, 'method': 'template', 'score': tm['max_val']}
                # store debug viz
                dbg = cv2.cvtColor(imgs_gray[j], cv2.COLOR_GRAY2BGR)
                th, tw = tmpl.shape
                cv2.rectangle(dbg, (tgt_x,tgt_y), (tgt_x+tw, tgt_y+th), (0,255,0), 2)
                save_dbg(f"match_{i}_to_{j}_template.png", dbg)
                continue
            # else attempt feature-match on bar ROIs (or whole images)
            res_feat = feature_match_ransac(imgs_gray[i], imgs_gray[j], maskA=masks[i], maskB=masks[j])
            if res_feat is not None and res_feat['inliers'] >= 8:
                results[key] = {'H': res_feat['H'], 'method': 'feature', 'score': res_feat['inliers']}
                # store debug visualization of inliers not done here to keep code compact
                continue
            # otherwise mark as no reliable match
            results[key] = None
    return results

# -----------------------
# Build global coordinate system and warp to canvas
# -----------------------
def compose_global_transforms(n, pairwise):
    # Build adjacency and edge weights based on method/score
    adj = defaultdict(list)
    for (i,j),res in pairwise.items():
        if res is None:
            continue
        H = res['H']
        # store i->j as H
        adj[i].append((j, H, res.get('score',1.0)))
        # store inverted j->i
        try:
            Hinv = np.linalg.inv(H)
            adj[j].append((i, Hinv, res.get('score',1.0)))
        except np.linalg.LinAlgError:
            continue
    # BFS from node 0 to compute transform to canvas root
    root = 0
    transforms = {root: np.eye(3,3, dtype=np.float64)}
    visited = set([root])
    q = deque([root])
    while q:
        i = q.popleft()
        for (j, H_ij, score) in adj[i]:
            if j in visited:
                continue
            # H_ij maps i -> j (as stored), so to get j->root:
            # transforms[j] = transforms[i] * inv(H_ij)
            try:
                Hinv = np.linalg.inv(H_ij)
            except np.linalg.LinAlgError:
                continue
            transforms[j] = transforms[i] @ Hinv
            visited.add(j)
            q.append(j)
    # any unvisited nodes get identity (fallback)
    for k in range(n):
        if k not in transforms:
            transforms[k] = np.eye(3,3,dtype=np.float64)
    return transforms

def warp_and_blend(images_color, transforms):
    # compute corners to get canvas extents
    all_pts = []
    for i, img in enumerate(images_color):
        h,w = img.shape[:2]
        H = transforms[i]
        pts = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]], dtype=np.float64).T
        warped = H @ pts
        warped = warped / warped[2,:]
        all_pts.append(warped[:2,:].T)
    all_pts = np.vstack(all_pts)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    min_x, min_y = min_xy
    max_x, max_y = max_xy
    canvas_w = int(math.ceil(max_x - min_x))
    canvas_h = int(math.ceil(max_y - min_y))
    offset = np.eye(3, dtype=np.float64)
    offset[0,2] = -min_x
    offset[1,2] = -min_y
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    for i, img in enumerate(images_color):
        H = offset @ transforms[i]
        warped = cv2.warpPerspective(img.astype(np.float32), H, (canvas_w, canvas_h))
        mask = (warped.sum(axis=2) > 0).astype(np.float32)
        # soft feather: distance transform on mask (in source), then warp
        h0,w0 = img.shape[:2]
        src_mask = np.ones((h0,w0), dtype=np.uint8)*255
        dist = cv2.distanceTransform(src_mask, cv2.DIST_L2, 5)
        dist_norm = dist / (dist.max()+1e-8)
        warped_mask = cv2.warpPerspective(dist_norm.astype(np.float32), H, (canvas_w, canvas_h))
        for c in range(3):
            canvas[:,:,c] += warped[:,:,c] * warped_mask
        weight += warped_mask
    eps = 1e-8
    out = np.zeros_like(canvas, dtype=np.uint8)
    for c in range(3):
        ch = canvas[:,:,c] / (weight + eps)
        ch = np.nan_to_num(ch)
        ch = np.clip(ch, 0, 255)
        out[:,:,c] = ch.astype(np.uint8)
    return out, (canvas_w, canvas_h), (-min_x, -min_y)

# -----------------------
# High-level pipeline
# -----------------------
def stitch_pipeline(paths, debug=True):
    # load
    colors = []
    grays = []
    for p in paths:
        if not os.path.exists(p):
            print("Warning: path not found:", p)
        try:
            c,g = imread_color_gray(p)
            colors.append(c)
            grays.append(g)
        except FileNotFoundError:
            colors.append(None)
            grays.append(None)
    # detect bars
    masks = []
    bboxes = []
    centers = []
    prefixes = []
    for idx,g in enumerate(grays):
        prefix = f"zone{idx}"
        prefixes.append(prefix)
        if g is None:
            masks.append(None); bboxes.append(None); centers.append(None); continue
        mask, cy, bbox = detect_horizontal_bar_v2(g, debug_prefix=prefix if debug else "")
        masks.append(mask)
        bboxes.append(bbox)
        centers.append(cy)
        if debug:
            # save overlay
            dbg = colors[idx].copy()
            y0 = int(bbox[1]); y1 = int(bbox[1]+bbox[3])
            cv2.rectangle(dbg, (int(bbox[0]), y0), (int(bbox[0]+bbox[2]), y1), (0,0,255), 2)
            save_dbg(f"{prefix}_bar_overlay.png", dbg)
    # pairwise align
    pairwise = pairwise_align(grays, masks, bboxes, prefixes)
    # compose transforms
    n = len(paths)
    transforms = compose_global_transforms(n, pairwise)
    # warp and blend
    stitched, (W,H), offset = warp_and_blend(colors, transforms)
    # save
    save_dbg("stitched_result.png", stitched)
    if debug:
        print("Saved stitched_result.png to", OUT_DIR)
        print("Size (px):", W, H)
    return stitched, transforms, (W,H), offset, pairwise

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    stitched, transforms, (W,H), offset, pairwise = stitch_pipeline(IMAGE_PATHS, debug=True)
    # print pairwise summary
    print("Pairwise summary (i->j):")
    for k,v in pairwise.items():
        print(k, "->", None if v is None else f"{v['method']} score={v['score']}")
    print("Transforms (homographies) keys:")
    for i,H in transforms.items():
        print(i, "H shape:", H.shape)
    print("Stitched size (px):", W, H)
    # Example measurement: if you know px_per_mm supply it to compute mm:
    # px_per_mm = ...
    # print("Measured mm dims:", W/px_per_mm, H/px_per_mm)
