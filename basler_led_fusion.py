
import argparse
from pathlib import Path
import cv2
import numpy as np
import tifffile

GAUSSIAN_SIGMA=81
LEVELS=5

def read16(p):
    return tifffile.imread(p).astype(np.float32)

def save16(p,img):
    img=np.clip(img,0,65535).astype(np.uint16)
    tifffile.imwrite(str(p),img)

def register(ref,mov):
    shift,_=cv2.phaseCorrelate(ref, mov)
    dx,dy=shift
    M=np.float32([[1,0,-dx],[0,1,-dy]])
    reg=cv2.warpAffine(mov,M,(ref.shape[1],ref.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    return reg,(dx,dy)

def flat(img):
    bg=cv2.GaussianBlur(img,(0,0),GAUSSIAN_SIGMA)
    bg=np.maximum(bg,1)
    out=img/bg*np.mean(bg)
    return out

def weight(img):
    gx=cv2.Sobel(img,cv2.CV_32F,1,0,3)
    gy=cv2.Sobel(img,cv2.CV_32F,0,1,3)
    w=cv2.magnitude(gx,gy)+1e-6
    return cv2.GaussianBlur(w,(0,0),3)

def gp(img):
    g=[img]
    for _ in range(LEVELS):
        g.append(cv2.pyrDown(g[-1]))
    return g

def lp(img):
    g=gp(img)
    l=[]
    for i in range(len(g)-1):
        up=cv2.pyrUp(g[i+1],dstsize=(g[i].shape[1],g[i].shape[0]))
        l.append(g[i]-up)
    l.append(g[-1])
    return l

def reconstruct(l):
    img=l[-1]
    for lev in reversed(l[:-1]):
        img=cv2.pyrUp(img,dstsize=(lev.shape[1],lev.shape[0]))+lev
    return img

def fuse(a,b):
    wa=gp(weight(a))
    wb=gp(weight(b))
    la=lp(a); lb=lp(b)
    lf=[]
    for i in range(len(la)):
        sa=wa[i]/(wa[i]+wb[i]+1e-6)
        sb=1-sa
        lf.append(la[i]*sa+lb[i]*sb)
    return reconstruct(lf)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("odd")
    ap.add_argument("even")
    ap.add_argument("-o","--out",default="output")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(exist_ok=True)
    odd=read16(args.odd); even=read16(args.even)
    even_reg,shift=register(odd,even)
    oddf=flat(odd); evenf=flat(even_reg)
    fused=fuse(oddf,evenf)
    save16(out/"odd_registered.tif",odd)
    save16(out/"even_registered.tif",even_reg)
    save16(out/"odd_flat.tif",oddf)
    save16(out/"even_flat.tif",evenf)
    save16(out/"fused.tif",fused)
    print("Shift:",shift)
if __name__=="__main__":
    main()
