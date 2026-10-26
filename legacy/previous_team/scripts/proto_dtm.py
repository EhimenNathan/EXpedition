import sys, json, glob, time, numpy as np, geopandas as gpd, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"."); from koz3 import dtm as D
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
site="180_Гнёздово_Лесная_2"; t0=time.time()
x,y,z,it,gfrac=D.read_las("work/180.las")
area=(x.max()-x.min())*(y.max()-y.min()); dens=len(x)/area; r=D.grid_res_for_density(dens)
print(f"pts={len(x)} ground_frac={gfrac:.2f} dens={dens:.1f}/m2 grid_res={r} x[{x.min():.0f},{x.max():.0f}] y[{y.min():.0f},{y.max():.0f}] z[{z.min():.1f},{z.max():.1f}]")
b=(np.floor(x.min()),np.floor(y.min()),np.ceil(x.max()),np.ceil(y.max()))
dtm,inten,cnt,hole,tr=D.grid_dtm(x,y,z,it,b,r)
dtm=D.resample_to(dtm,r); inten=D.resample_to(inten,r)
print("dtm",dtm.shape,"empty cells %.1f%%"%(100*(cnt==0).mean()),"time %.1fs"%(time.time()-t0))
t1=time.time(); ch=D.relief_stack(dtm); print("relief %.1fs"%(time.time()-t1))
utm=json.load(open(f"data_meta/{site}/UTM.json",encoding="utf-8"))["crs"].replace("urn:ogc:def:crs:EPSG::","EPSG:")
L=[]
for f in glob.glob(f"data_meta/{site}/**/*.geojson",recursive=True):
    g=gpd.read_file(f).to_crs(utm); g["name"]=f.split("_Li_")[-1].replace(".geojson","") if "_Li_" in f else f; L.append(g)
L=pd.concat(L); L=L[~L.geometry.is_empty & L.geometry.notna()]
print(L.name.value_counts().to_dict())
# per-object morphometrics from the DTM
from rasterio import features, transform as rt
T=rt.Affine(D.MODEL_GSD,0,b[0],0,-D.MODEL_GSD,b[3])
lab=features.rasterize([(g,i+1) for i,g in enumerate(L.geometry)],out_shape=dtm.shape,transform=T)
rows=[]
for i,(g,nm) in enumerate(zip(L.geometry,L.name)):
    m=lab==i+1
    if m.sum()<4: continue
    ring=(np.logical_xor(np.isin(lab,[0,i+1]) & (np.hypot(*np.meshgrid(np.arange(dtm.shape[1]),np.arange(dtm.shape[0]),indexing='xy'))>=0),False))
    # local base = median DTM in a 3-6 m annulus outside the polygon
    dil=__import__("scipy.ndimage",fromlist=["x"]).binary_dilation
    a=dil(m,iterations=int(6/D.MODEL_GSD)) & ~dil(m,iterations=int(3/D.MODEL_GSD)) & (lab==0)
    base=np.median(dtm[a]) if a.sum()>10 else np.nan
    zz=dtm[m]; rr,cc=np.nonzero(m); k=np.argmax(zz)
    cx,cy=g.centroid.x,g.centroid.y; px=b[0]+(cc[k]+.5)*D.MODEL_GSD; py=b[3]-(rr[k]+.5)*D.MODEL_GSD
    h=zz.max()-base
    iso=(zz>=base+0.1*h).sum()*D.MODEL_GSD**2 if h>0 else np.nan
    rows.append(dict(cls=nm,area=g.area,height=h,diam=2*np.sqrt(g.area/np.pi),lrm8max=ch["lrm15"][m].max(),log8=ch["log8"][m].max(),
                     peak_off=np.hypot(px-cx,py-cy),iso10_area_ratio=iso/g.area if h>0 else np.nan,inten_contrast=inten[m].mean()-(inten[a].mean() if a.sum() else np.nan)))
M=pd.DataFrame(rows); M.to_csv("work/morph_180.csv",index=False)
print(M.groupby("cls").agg(n=("area","size"),diam=("diam","median"),h_med=("height","median"),h_p10=("height",lambda s:s.quantile(.1)),h_p90=("height",lambda s:s.quantile(.9)),
     peak_off=("peak_off","median"),iso10=("iso10_area_ratio","median"),log8=("log8","median"),inten=("inten_contrast","median")).round(2).to_string())
# figure
cx,cy=L.geometry.centroid.x.median(),L.geometry.centroid.y.median(); half=60
c0=int((cx-half-b[0])/D.MODEL_GSD); r0=int((b[3]-(cy+half))/D.MODEL_GSD); n=int(2*half/D.MODEL_GSD)
names=["hs315","lrm5","lrm15","log8","svf","oneg","slope"]
fig,axs=plt.subplots(2,4,figsize=(20,10))
for ax,nm in zip(axs.flat,names+["inten"]):
    a=(ch[nm] if nm!="inten" else inten)[r0:r0+n,c0:c0+n]; lo,hi=np.percentile(a,[2,98])
    ax.imshow(a,cmap="gray" if nm in("hs315","svf","oneg","inten") else "RdBu_r",vmin=lo,vmax=hi,extent=(cx-half,cx+half,cy-half,cy+half))
    L.boundary.plot(ax=ax,color="lime",lw=0.6); ax.set_xlim(cx-half,cx+half); ax.set_ylim(cy-half,cy+half); ax.set_title(nm); ax.axis("off")
plt.tight_layout(); plt.savefig("viz/relief_180.png",dpi=55)
