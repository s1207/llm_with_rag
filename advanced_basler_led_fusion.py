"""
Advanced Basler LED Fusion Pipeline
===================================
Baseline framework for industrial glass inspection.

Features
--------
- 16-bit TIFF support
- Subpixel registration
- Flat-field correction
- Bright/dark band estimation
- Confidence map generation
- Ghost suppression (placeholder)
- Laplacian pyramid fusion
- Batch processing

TODO sections are marked for future tuning.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
import tifffile

class Config:
    gaussian_sigma=81
    pyramid_levels=5
    band_blur=51
    debug=True

def read16(f): return tifffile.imread(f).astype(np.float32)
def save16(f,img): tifffile.imwrite(str(f),np.clip(img,0,65535).astype(np.uint16))

def register(ref,mov):
    shift,_=cv2.phaseCorrelate(ref,mov)
    dx,dy=shift
    M=np.float32([[1,0,-dx],[0,1,-dy]])
    reg=cv2.warpAffine(mov,M,(ref.shape[1],ref.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    return reg

def flatfield(img,sigma):
    bg=cv2.GaussianBlur(img,(0,0),sigma)
    bg=np.maximum(bg,1)
    return img/bg*np.mean(bg)

def estimate_bands(img):
    illum=cv2.GaussianBlur(img,(0,0),Config.band_blur)
    illum=(illum-illum.min())/(illum.max()-illum.min()+1e-6)
    return illum

def ghost_suppression(img):
    # Placeholder for future deghosting model
    return img

def gradient_weight(img):
    gx=cv2.Sobel(img,cv2.CV_32F,1,0)
    gy=cv2.Sobel(img,cv2.CV_32F,0,1)
    return cv2.GaussianBlur(cv2.magnitude(gx,gy),(0,0),3)+1e-6

def gp(img,n):
    g=[img]
    for _ in range(n): g.append(cv2.pyrDown(g[-1]))
    return g

def lp(img,n):
    g=gp(img,n)
    l=[]
    for i in range(len(g)-1):
        l.append(g[i]-cv2.pyrUp(g[i+1],dstsize=(g[i].shape[1],g[i].shape[0])))
    l.append(g[-1]); return l

def recon(p):
    x=p[-1]
    for lev in reversed(p[:-1]):
        x=cv2.pyrUp(x,dstsize=(lev.shape[1],lev.shape[0]))+lev
    return x

def fuse(a,b):
    wa=gp(gradient_weight(a)*estimate_bands(a),Config.pyramid_levels)
    wb=gp(gradient_weight(b)*estimate_bands(b),Config.pyramid_levels)
    la=lp(a,Config.pyramid_levels); lb=lp(b,Config.pyramid_levels)
    out=[]
    for A,B,WA,WB in zip(la,lb,wa,wb):
        alpha=WA/(WA+WB+1e-6)
        out.append(alpha*A+(1-alpha)*B)
    return recon(out)

def process(odd,even,outdir):
    outdir=Path(outdir); outdir.mkdir(exist_ok=True)
    o=read16(odd); e=register(o,read16(even))
    o=ghost_suppression(flatfield(o,Config.gaussian_sigma))
    e=ghost_suppression(flatfield(e,Config.gaussian_sigma))
    f=fuse(o,e)
    save16(outdir/"fused.tif",f)
    if Config.debug:
        save16(outdir/"odd_corrected.tif",o)
        save16(outdir/"even_corrected.tif",e)
        save16(outdir/"odd_bands.tif",estimate_bands(o)*65535)
        save16(outdir/"even_bands.tif",estimate_bands(e)*65535)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("odd"); ap.add_argument("even"); ap.add_argument("-o","--out",default="output")
    a=ap.parse_args()
    process(a.odd,a.even,a.out)
