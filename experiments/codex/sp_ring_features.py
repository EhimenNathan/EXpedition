"""CPU-only scale-specific circularity features for held-out satellite candidates.

No labels are used in extraction. Only existing geographic OOF candidates and
source imagery are read. Candidate labels are joined later for outer-fold tests.
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import Affine

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

R = 48
yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
rr = np.hypot(xx, yy)
unitx, unity = xx / np.maximum(rr, 1), yy / np.maximum(rr, 1)
angles = np.arctan2(yy, xx)
radii = (4, 6, 8, 12, 16, 24, 32)

def ring_features(arr, x, y):
    _, h, w = arr.shape
    if x < R or y < R or x >= w-R or y >= h-R: return None
    a = np.asarray(arr[:3, y-R:y+R+1, x-R:x+R+1], np.float32)
    if a.shape != (3, 2*R+1, 2*R+1): return None
    gray = .299*a[0] + .587*a[1] + .114*a[2]
    gray = gray - cv2.GaussianBlur(gray, (0,0), 24)
    gray /= max(float(np.std(gray)), 1.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    rad = gx*unitx + gy*unity
    tan = -gx*unity + gy*unitx
    out = {}
    for r in radii:
        m = (np.abs(rr-r) <= 1.5)
        v, g, t = gray[m], rad[m], tan[m]
        # Mean radial gradient is signed ring evidence; strength/symmetry
        # distinguishes circular boundaries from directional field stripes.
        out[f"ring_grad_{r}"] = float(np.mean(g))
        out[f"ring_sym_{r}"] = float(abs(np.mean(g)) / (np.mean(abs(g)) + 1e-4))
        out[f"ring_rad_tan_{r}"] = float(np.mean(abs(g)) / (np.mean(abs(t)) + 1e-4))
        out[f"ring_angular_{r}"] = float(np.std(v))
        out[f"ring_mean_{r}"] = float(np.mean(v))
        # m=1/2/4 angular Fourier energies capture elongated boundaries
        # and crop-row texture versus a rotationally symmetric mound.
        theta = angles[m]
        for k in (1,2,4):
            amp = abs(np.mean((v-v.mean()) * np.exp(-1j*k*theta)))
            out[f"ring_fourier{k}_{r}"] = float(amp)
    for sigma in (2,4,8,12):
        smooth = cv2.GaussianBlur(gray, (0,0), sigma)
        lap = cv2.Laplacian(smooth, cv2.CV_32F)
        out[f"log_center_{sigma}"] = float(-sigma*sigma*lap[R,R])
    return out

def main():
    F = pd.read_parquet(HERE / "sp_morph_features.parquet")
    SC = pd.read_csv(ROOT / "kaggle_import/code/work/sp/scenes.csv")
    rows = []
    for n, (region, q) in enumerate(F.groupby("region")):
        q=q.copy(); q["scene"] = None; q["px"]=-1; q["py"]=-1
        margin = np.full(len(q), -1)
        for sc in SC[SC.site==region].itertuples():
            T=Affine(*json.loads(sc.T))
            tr=Transformer.from_crs(3857,sc.crs,always_xy=True)
            X,Y=tr.transform(q.x.to_numpy(),q.y.to_numpy())
            ix,iy=(~T)*(X,Y)
            ix=np.asarray(ix).round().astype(int); iy=np.asarray(iy).round().astype(int)
            m=np.minimum.reduce([ix,iy,sc.W-1-ix,sc.H-1-iy])
            use=(m>R)&(m>margin)
            q.iloc[np.where(use)[0],q.columns.get_loc("scene")]=sc.scene
            q.iloc[np.where(use)[0],q.columns.get_loc("px")]=ix[use]
            q.iloc[np.where(use)[0],q.columns.get_loc("py")]=iy[use]
            margin[use]=m[use]
        for scene, g in q.groupby("scene",dropna=True):
            arr=np.load(ROOT / "kaggle_import/code/work/sp/img" / f"{scene}.npy",mmap_mode="r")
            for p in g.itertuples():
                d=ring_features(arr,int(p.px),int(p.py))
                if d is None:continue
                d.update({k:getattr(p,k) for k in ("region","fold","cls","score","x","y","y_obj")})
                rows.append(d)
        if (n+1)%20==0: print("regions",n+1,"features",len(rows),flush=True)
    out=pd.DataFrame(rows)
    out.to_parquet(HERE / "sp_ring_features.parquet",index=False)
    print("saved",len(out),"positives",int(out.y_obj.sum()),"folds",out.fold.value_counts().to_dict(),flush=True)

if __name__=="__main__": main()
