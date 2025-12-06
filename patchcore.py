"""
patchcore_tiled_faiss.py

- Builds a memory bank from normal images using tiling (keeps high-res detail).
- Builds a FAISS CPU index and saves both features (.npy) and index (.index).
- Runs tiled inference on large images and stitches an anomaly heatmap back to full resolution.

Author: ChatGPT
"""

import os
import math
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from sklearn.decomposition import IncrementalPCA
import faiss

# -----------------------
# CONFIG (edit these)
# -----------------------
DATA_ROOT = Path("./datasets/custom")        # should contain 'good' (normal) and 'defect' (abnormal) subfolders
NORMAL_DIRNAME = "good"
ABNORMAL_DIRNAME = "defect"

# Tiling / image handling
TILE_SIZE = 1024           # tile size in px (recommended 1024)
OVERLAP = 256              # overlap in px between tiles
MAX_SIDE = None            # not used (we do tiling), set if you prefer global resize

# Sampling & memory bank sizing
SAMPLE_PER_TILE = 500      # randomly sample up to this many patch vectors per tile
MAX_MEMORY_VECTORS = 120000  # cap total sampled vectors kept for FAISS (tune for memory)
USE_PCA = True
PCA_DIM = 128              # reduced dim for FAISS / storage (smaller => faster)
PCA_BATCH = 4096

# FAISS / search
FAISS_NLIST = 0            # 0 -> use IndexFlatL2; if >0 can build IVF (faster for very large banks)
TOP_K = 1                  # nearest neighbors for anomaly score
FAISS_METRIC = faiss.METRIC_L2

# Model backbone settings
BACKBONE = "wide_resnet50_2"
USE_PRETRAINED = True
LAYERS = ["layer2", "layer3"]   # as requested

# Device
DEVICE = "cpu"  # you said CPU only

# Output files
OUT_DIR = Path("./patchcore_outputs")
OUT_DIR.mkdir(exist_ok=True)
FEATURES_PATH = OUT_DIR / "memory_bank_vectors.npy"
FAISS_INDEX_PATH = OUT_DIR / "faiss_index.index"
PCA_PATH = OUT_DIR / "ipca.npy"

# -----------------------
# Utilities: tiling, stitching
# -----------------------
def make_tiles(img_w, img_h, tile_size=TILE_SIZE, overlap=OVERLAP):
    """Return list of (x0, y0, x1, y1) tile boxes that cover the image with given overlap."""
    stride = tile_size - overlap
    xs = list(range(0, max(1, img_w - tile_size + 1), stride))
    ys = list(range(0, max(1, img_h - tile_size + 1), stride))
    # ensure right/bottom edges included
    if xs == []:
        xs = [0]
    if ys == []:
        ys = [0]
    tiles = []
    for y in ys:
        for x in xs:
            x1 = min(x + tile_size, img_w)
            y1 = min(y + tile_size, img_h)
            x0 = max(0, x1 - tile_size)  # shift if near edge so tile size stays consistent
            y0 = max(0, y1 - tile_size)
            tiles.append((x0, y0, x1, y1))
    # remove duplicates
    tiles = list(dict.fromkeys(tiles))
    return tiles

def stitch_maps(score_tiles, img_w, img_h, tile_boxes, tile_size=TILE_SIZE):
    """
    Given list of score arrays (tile_h, tile_w) and their boxes, stitch them together by averaging overlaps.
    Returns full-size score map (img_h, img_w) float32 normalized to 0..1.
    """
    acc = np.zeros((img_h, img_w), dtype=np.float32)
    count = np.zeros((img_h, img_w), dtype=np.float32)
    for score, box in zip(score_tiles, tile_boxes):
        x0,y0,x1,y1 = box
        H = y1 - y0
        W = x1 - x0
        # resize score to tile shape (in case feature-grid differs)
        score_resized = cv2.resize(score.astype("float32"), (W, H), interpolation=cv2.INTER_CUBIC)
        acc[y0:y1, x0:x1] += score_resized
        count[y0:y1, x0:x1] += 1.0
    # avoid division by zero
    mask = count > 0
    out = np.zeros_like(acc)
    out[mask] = acc[mask] / count[mask]
    # normalize to 0..1
    out = out - out.min()
    if out.max() > 0:
        out = out / out.max()
    return out

# -----------------------
# Backbone with hooks to capture layer2 & layer3
# -----------------------
class BackBoneFeatureExtractor:
    def __init__(self, device="cpu", pretrained=True, layers=("layer2","layer3")):
        # load torchvision wide_resnet50_2 (weights pre-trained on ImageNet)
        # we use torchvision backbone because anomalib's Patchcore wrapper internals differ across versions.
        model = models.wide_resnet50_2(pretrained=pretrained)
        model.eval()
        model.to(device)
        self.device = device
        self.model = model
        self._layers = layers
        self._saved = {}
        # register hooks
        def get_hook(name):
            def hook(module, inp, out):
                # out is tensor shape (B,C,H,W)
                self._saved[name] = out.detach().cpu()
            return hook
        # attach hooks
        for name in layers:
            layer = dict(model.named_children())[name]
            layer.register_forward_hook(get_hook(name))

        # preproc transform
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def forward(self, pil_tile: Image.Image):
        """
        Input: PIL tile
        Returns: dict{name: feature_tensor_cpu} where each feature tensor is (C,H,W) numpy (float32)
        """
        img_t = self.transform(pil_tile).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _ = self.model(img_t)
        # get saved features and clear them
        out = {}
        for k,v in self._saved.items():
            # v is (1,C,H,W) tensor on CPU
            out[k] = v.squeeze(0).numpy().astype(np.float32)
        self._saved.clear()
        return out

# -----------------------
# Feature -> per-patch vector logic
# -----------------------
def fmaps_to_patch_vectors(fmaps_dict, layers_order=("layer2","layer3")):
    """
    Given a dict of {layer_name: (C,H,W)}, upsample deeper-layer maps to the spatial size of the first (layer2),
    and concatenate along channels. Return a (Npatches, C_total) numpy array.
    """
    base_layer = layers_order[0]
    base = fmaps_dict[base_layer]
    C0, H0, W0 = base.shape
    parts = [base.reshape(C0, -1)]  # (C0, H0*W0)
    for lname in layers_order[1:]:
        fmap = fmaps_dict[lname]  # (C,H,W)
        # upsample to (H0, W0) using cv2.resize on each channel
        C, H, W = fmap.shape
        up = np.zeros((C, H0, W0), dtype=np.float32)
        for c in range(C):
            up[c] = cv2.resize(fmap[c], (W0, H0), interpolation=cv2.INTER_CUBIC)
        parts.append(up.reshape(C, -1))
    # concatenate channels
    concat = np.vstack(parts)  # (C_total, H0*W0)
    # transpose to (Npatches, C_total)
    patches = concat.T.copy()  # (H0*W0, C_total)
    return patches, (H0, W0)

# -----------------------
# Stream-build memory bank
# -----------------------
def build_memory_bank_from_folder(
    normal_dir,
    backbone_extractor: BackBoneFeatureExtractor,
    tile_size=TILE_SIZE,
    overlap=OVERLAP,
    sample_per_tile=SAMPLE_PER_TILE,
    max_vectors=MAX_MEMORY_VECTORS,
    use_pca=USE_PCA,
    pca_dim=PCA_DIM,
    pca_batch=PCA_BATCH,
    layers_order=tuple(LAYERS)
):
    normal_paths = sorted([p for p in Path(normal_dir).glob("*") if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".tif",".tiff")])
    if len(normal_paths) == 0:
        raise ValueError(f"No images found in {normal_dir}")

    sampled_vectors = []
    total_collected = 0

    # We'll also fit IncrementalPCA in streaming if requested
    ipca = None
    if use_pca:
        ipca = IncrementalPCA(n_components=pca_dim)

    print("Sampling patch vectors from normal images (streaming)...")
    for p in tqdm(normal_paths, desc="images"):
        im = Image.open(p).convert("RGB")
        W,H = im.size
        tiles = make_tiles(W, H, tile_size=tile_size, overlap=overlap)
        for box in tiles:
            x0,y0,x1,y1 = box
            tile = im.crop((x0,y0,x1,y1))
            fmaps = backbone_extractor.forward(tile)  # dict: layer->(C,H,W)
            patches, (Hf, Wf) = fmaps_to_patch_vectors(fmaps, layers_order=layers_order)  # (Npatch, dim)
            n_patches = patches.shape[0]
            if n_patches == 0:
                continue
            # sample subset per tile
            nsel = min(sample_per_tile, n_patches)
            if n_patches <= nsel:
                sel = patches
            else:
                idx = np.random.choice(n_patches, size=nsel, replace=False)
                sel = patches[idx]
            # optionally fit PCA incrementally on sel (fit on the fly)
            if use_pca:
                # ipca.partial_fit accepts (n_samples, n_features); call in chunks
                chunk = pca_batch
                for i in range(0, sel.shape[0], chunk):
                    ipca.partial_fit(sel[i:i+chunk])
            sampled_vectors.append(sel)
            total_collected += sel.shape[0]
            # cap memory vectors
            if total_collected >= max_vectors:
                break
        if total_collected >= max_vectors:
            break

    # concatenate
    print(f"Collected ~{total_collected} patch vectors. Concatenating...")
    all_vecs = np.vstack(sampled_vectors).astype(np.float32)
    # trim to max_vectors if slightly over
    if all_vecs.shape[0] > max_vectors:
        all_vecs = all_vecs[:max_vectors]
    print("All vectors shape:", all_vecs.shape)

    # project with ipca if requested
    if use_pca:
        print("Transforming with IncrementalPCA...")
        # ipca has been partial_fit in streaming; now transform in chunks to avoid mem blowup
        X = []
        chunk = pca_batch
        for i in range(0, all_vecs.shape[0], chunk):
            X.append(ipca.transform(all_vecs[i:i+chunk]))
        X = np.vstack(X).astype(np.float32)
        # save ipca components for reuse
        np.save(PCA_PATH, {"components": ipca.components_, "mean": ipca.mean_})
        memory_vectors = X
    else:
        memory_vectors = all_vecs

    # Save raw vectors for debugging
    np.save(FEATURES_PATH, memory_vectors)
    print(f"Saved memory bank vectors to {FEATURES_PATH} (shape {memory_vectors.shape})")
    return memory_vectors, ipca

# -----------------------
# Build FAISS index (CPU)
# -----------------------
def build_faiss_index(memory_vectors, nlist=FAISS_NLIST, save_path=FAISS_INDEX_PATH):
    d = memory_vectors.shape[1]
    if nlist and nlist > 0:
        # IVF index (requires training)
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.nprobe = max(1, nlist//10)
        print("Training IVF index...")
        index.train(memory_vectors)
        index.add(memory_vectors)
    else:
        # exact index (flat)
        index = faiss.IndexFlatL2(d)
        index.add(memory_vectors)
    # Save index
    faiss.write_index(index, str(save_path))
    print(f"FAISS index built and saved to {save_path}; total vectors in index = {index.ntotal}")
    return index

# -----------------------
# Inference: tile, extract, query FAISS, stitch
# -----------------------
def infer_image_with_faiss(
    image_path,
    backbone_extractor,
    index,
    ipca=None,
    layers_order=tuple(LAYERS),
    tile_size=TILE_SIZE,
    overlap=OVERLAP,
    top_k=TOP_K
):
    im = Image.open(image_path).convert("RGB")
    W,H = im.size
    tiles = make_tiles(W,H, tile_size=tile_size, overlap=overlap)
    score_tiles = []
    boxes = []
    for box in tqdm(tiles, desc="tiles"):
        x0,y0,x1,y1 = box
        tile = im.crop((x0,y0,x1,y1))
        fmaps = backbone_extractor.forward(tile)
        patches, (Hf, Wf) = fmaps_to_patch_vectors(fmaps, layers_order=layers_order)  # (Npatch, dim)
        if patches.shape[0] == 0:
            # empty tile
            score_map = np.zeros((Hf, Wf), dtype=np.float32)
            score_tiles.append(score_map)
            boxes.append(box)
            continue
        feats = patches.astype(np.float32)
        # PCA transform if ipca provided (ipca here is object or saved dict)
        if ipca is not None:
            # ipca may be sklearn object or saved dict; handle both
            if hasattr(ipca, "transform"):
                feats_proj = ipca.transform(feats)
            else:
                # ipca is dict with components and mean
                comps = ipca["components"]
                mean = ipca["mean"]
                feats_proj = np.dot(feats - mean, comps.T)
        else:
            feats_proj = feats
        # query faiss
        D, I = index.search(feats_proj.astype(np.float32), top_k)  # D: distances
        # distance per patch: take mean across k neighbors (here k=1)
        dists = D.mean(axis=1)  # (Npatch,)
        # reshape to (Hf, Wf)
        score_map = dists.reshape(Hf, Wf)
        # normalize tile-level to 0..1
        sc = score_map
        sc = sc - sc.min()
        if sc.max() > 0:
            sc = sc / sc.max()
        score_tiles.append(sc.astype(np.float32))
        boxes.append(box)
    # stitch tiles to full image
    full_score = stitch_maps(score_tiles, W, H, boxes, tile_size=tile_size)
    return full_score

# -----------------------
# Visualization helper
# -----------------------
def save_heatmap_overlay(orig_path, score_map, out_path):
    orig = cv2.cvtColor(np.array(Image.open(orig_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    H,W,_ = orig.shape
    heat = (255 * score_map).astype(np.uint8)
    heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_CUBIC)
    heatc = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.6, heatc, 0.4, 0)
    cv2.imwrite(str(out_path), overlay)

# -----------------------
# Main runnable flow
# -----------------------
def main():
    # create extractor
    extractor = BackBoneFeatureExtractor(device=DEVICE, pretrained=USE_PRETRAINED, layers=tuple(LAYERS))

    # Build memory bank (streaming)
    normal_dir = DATA_ROOT / NORMAL_DIRNAME
    memory_vectors, ipca_model = build_memory_bank_from_folder(
        normal_dir=normal_dir,
        backbone_extractor=extractor,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        sample_per_tile=SAMPLE_PER_TILE,
        max_vectors=MAX_MEMORY_VECTORS,
        use_pca=USE_PCA,
        pca_dim=PCA_DIM,
        pca_batch=PCA_BATCH,
        layers_order=tuple(LAYERS)
    )

    # If USE_PCA, we saved PCA components to PCA_PATH as dict; for FAISS we need transformed vectors
    if USE_PCA:
        # Transform memory_vectors with saved ipca components (ipca_model is sklearn IncrementalPCA object)
        print("Applying final PCA transform to build FAISS vectors...")
        # If ipca_model is None, try loading saved PCA
        if ipca_model is None:
            pca_saved = np.load(PCA_PATH, allow_pickle=True).item()
            comps = pca_saved["components"]
            mean = pca_saved["mean"]
            memory_proj = np.dot(memory_vectors - mean, comps.T).astype(np.float32)
            ipca_for_infer = {"components": comps, "mean": mean}
        else:
            # sklearn ipca object present
            X_chunks = []
            chunk = PCA_BATCH
            for i in range(0, memory_vectors.shape[0], chunk):
                X_chunks.append(ipca_model.transform(memory_vectors[i:i+chunk]))
            memory_proj = np.vstack(X_chunks).astype(np.float32)
            ipca_for_infer = ipca_model
    else:
        memory_proj = memory_vectors.astype(np.float32)
        ipca_for_infer = None

    # build faiss index
    index = build_faiss_index(memory_proj, nlist=FAISS_NLIST, save_path=FAISS_INDEX_PATH)

    # run inference on a few abnormal images and save overlays
    abnormal_dir = DATA_ROOT / ABNORMAL_DIRNAME
    abnormal_paths = sorted([p for p in abnormal_dir.glob("*") if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".tif",".tiff")])
    out_vis_dir = OUT_DIR / "visuals"
    out_vis_dir.mkdir(exist_ok=True)

    for p in abnormal_paths:
        print("Inferring:", p)
        score_map = infer_image_with_faiss(
            image_path=str(p),
            backbone_extractor=extractor,
            index=index,
            ipca=ipca_for_infer,
            layers_order=tuple(LAYERS),
            tile_size=TILE_SIZE,
            overlap=OVERLAP,
            top_k=TOP_K
        )
        out_path = out_vis_dir / f"{p.stem}_overlay.png"
        save_heatmap_overlay(str(p), score_map, out_path)
        print("Saved overlay to", out_path)

if __name__ == "__main__":
    main()
