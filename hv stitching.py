import cv2
import numpy as np
from skimage.transform import resize
from skimage.feature import canny
from skimage.measure import LineModelND, ransac


def load_and_downsample(path, target_max_dim=2500):
    """Load image and downsample so max dimension = target_max_dim."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    scale = target_max_dim / max(h, w)
    if scale < 1:
        img_small = cv2.resize(img, (int(w*scale), int(h*scale)),
                               interpolation=cv2.INTER_AREA)
        return img_small, scale
    return img, 1.0


def detect_dark_bars(img, orientation="horizontal"):
    """
    Detect dark horizontal or vertical bars using:
    - grayscale + blur
    - black-hat morphology
    - projection profile
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    if orientation == "horizontal":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[1]//2, 15))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, img.shape[0]//2))

    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)

    # Normalize
    norm = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

    # Projection profile
    if orientation == "horizontal":
        profile = np.sum(norm, axis=1)
    else:
        profile = np.sum(norm, axis=0)

    # Use peak detection
    thresh = np.mean(profile) + 1.5*np.std(profile)
    peaks = np.where(profile > thresh)[0]

    if len(peaks) == 0:
        return None

    # Return median location as the bar position
    return int(np.median(peaks))


def estimate_translation(imgA, imgB):
    """
    Robust image alignment by:
    - Grayscale
    - Canny edges
    - Cross-correlation (phase correlation)
    - RANSAC to filter out repeated or ambiguous patterns
    """
    A = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    B = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    edgesA = canny(A, sigma=2)
    edgesB = canny(B, sigma=2)

    # Phase correlation gives initial guess
    shift = cv2.phaseCorrelate(np.float32(edgesA), np.float32(edgesB))[0]
    shift = np.array(shift)  # (dy, dx)

    # Build point correspondences for RANSAC
    ys, xs = np.nonzero(edgesA)
    ptsA = np.vstack([xs, ys]).T
    ptsB = ptsA + shift[::-1]  # reverse to (dx, dy)

    # RANSAC model to refine
    model = LineModelND()
    try:
        model_robust, inliers = ransac(
            (ptsA, ptsB),
            LineModelND,
            min_samples=2,
            residual_threshold=3,
            max_trials=200
        )
        t = model_robust.params[1]  # translation vector
        return np.array([t[1], t[0]])  # return as (dy, dx)
    except Exception:
        # fallback to phase correlation only
        return shift


def stitch_two_zones(zoneA, zoneB, barA, barB):
    """
    Aligns two zones vertically based on bar positions + translation.
    """
    dy_bar = barA - barB

    # robust XY shift
    dy_est, dx_est = estimate_translation(zoneA, zoneB)

    dy = int(round(dy_bar + dy_est))
    dx = int(round(dx_est))

    # create output
    h = max(zoneA.shape[0], zoneB.shape[0] + abs(dy))
    w = max(zoneA.shape[1], zoneB.shape[1] + abs(dx))

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:zoneA.shape[0], :zoneA.shape[1]] = zoneA

    y = dy if dy >= 0 else 0
    x = dx if dx >= 0 else 0

    y2 = 0 if dy >= 0 else -dy
    x2 = 0 if dx >= 0 else -dx

    h2, w2 = zoneB.shape[:2]
    canvas[y:y+h2, x:x+w2] = zoneB

    return canvas


def measure_dimensions(img, pixel_size_mm=None):
    h, w = img.shape[:2]
    if pixel_size_mm:
        return h * pixel_size_mm, w * pixel_size_mm
    return h, w


top, s1 = load_and_downsample("/mnt/data/image0000025.png")
bottom, s2 = load_and_downsample("/mnt/data/image0000026.png")

# detect horizontal bars
bar_top = detect_dark_bars(top, "horizontal")
bar_bottom = detect_dark_bars(bottom, "horizontal")

stitched = stitch_two_zones(top, bottom, bar_top, bar_bottom)

# measure
height, width = measure_dimensions(stitched)
print("Stitched size:", height, width)

cv2.imwrite("stitched.png", stitched)
