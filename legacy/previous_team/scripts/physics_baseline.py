import sys, json, glob, time, numpy as np, geopandas as gpd, pandas as pd, torch, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"."); from koz3 import dtm as D, relief_torch as RT, metric as M
from skimage.feature import peak_local_max
from shapely.geometry import Point
site="180_Гнёздово_Лесная_2"
x,y,z,it,_=D.read_las("work/180.las"); b=(np.floor(x.min()),np.floor(y.min()),np.ceil(x.max()),np.ceil(y.max()))
dtm,inten,cnt,hole,tr=D.grid_dtm(x,y,z,it,b,0.5); dtm=D.resample_to(dtm,0.5)
t0=time.time(); R=RT.relief(torch.from_numpy(dtm)[None,None],0.25)[0].numpy(); print("relief_torch", R.shape, "%.1fs"%(time.time()-t0))
for i,c in enumerate(RT.CHANNELS): print(f"  {c:6s} p1={np.percentile(R[i],1):6.2f} p50={np.percentile(R[i],50):6.2f} p99={np.percentile(R[i],99):6.2f}")
utm=json.load(open(f"data_meta/{site}/UTM.json",encoding="utf-8"))["crs"].replace("urn:ogc:def:crs:EPSG::","EPSG:")
G=[]
for f in glob.glob(f"data_meta/{site}/**/*_Li_*.geojson",recursive=True):
    g=gpd.read_file(f).to_crs(utm); g["cls"]="kurgan"; G.append(g)
G=pd.concat(G); G=G[G.geometry.notna()&~G.geometry.is_empty]; G["group"]=site
# physics detector: multi-scale LoG peaks, score = LoG response; polygon = disc of the matched radius
rows=[]
for r_m,ch in ((2.0,"log2"),(5.0,"log5")):
    resp=R[RT.CHANNELS.index(ch)]
    pk=peak_local_max(resp,min_distance=int(r_m/0.25),threshold_abs=0.02)
    for rr,cc in pk:
        px=b[0]+(cc+.5)*0.25; py=b[3]-(rr+.5)*0.25
        rows.append(dict(group=site,cls="kurgan",geometry=Point(px,py).buffer(r_m),score=float(resp[rr,cc]),r=r_m))
P=gpd.GeoDataFrame(rows,crs=utm)
# NMS across scales: keep the higher-scored disc when centres are closer than the smaller radius
P=P.sort_values("score",ascending=False).reset_index(drop=True); keep=[]
for i,g in enumerate(P.geometry):
    if all(g.centroid.distance(P.geometry[j].centroid)>min(P.r[i],P.r[j]) for j in keep): keep.append(i)
P=P.loc[keep]
s=np.sort(P.score.values)
for q in (0.3,0.5,0.7,0.85):
    thr=float(np.quantile(s,q)); T=M.evaluate(P,G,["kurgan"],thresholds={"kurgan":thr})
    print(f"thr q{q} {thr:.3f}:", T.iloc[0][["n_gt","n_pred","TP","FP","FN","P","R","F1","F2","AP","AR","Fw"]].round(3).to_dict())
# --- chance baseline: same number of discs, same radii, uniform random centres inside the valid DTM area
rng=np.random.default_rng(0); thr=float(np.quantile(s,0.3)); PP=P[P.score>=thr]
H,W=dtm.shape; res=[]
for k in range(5):
    cc=rng.uniform(0,W,len(PP)); rr=rng.uniform(0,H,len(PP))
    Q=gpd.GeoDataFrame(dict(group=site,cls="kurgan",score=1.0,geometry=[Point(b[0]+c*0.25,b[3]-r*0.25).buffer(rad) for c,r,rad in zip(cc,rr,PP.r)]),crs=utm)
    T=M.evaluate(Q,G,["kurgan"],thresholds={"kurgan":0.5}); res.append(T.iloc[0][["TP","FP","P","R","F1","Fw"]].astype(float))
print("RANDOM discs:",pd.DataFrame(res).mean().round(3).to_dict())
print("GT area fraction of site: %.3f"%(G.geometry.area.sum()/(H*W*0.0625)))
