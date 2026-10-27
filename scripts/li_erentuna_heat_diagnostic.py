"""Diagnose whether Erentuna LiDAR misses are heat, position, or class failures."""
import pickle, subprocess, gc, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
reg='011_Эрентуна'; acc=None
for name in ('oof_ens.pkl.zst','oof_aug.pkl.zst','oof_augx.pkl.zst'):
 p=ROOT/'work/li'/name; print('load',name,flush=True)
 pr=subprocess.Popen(['zstd','-dc','-T4',str(p)],stdout=subprocess.PIPE)
 d=pickle.load(pr.stdout); pr.wait(); a=d[reg].astype(np.float32); del d; gc.collect()
 acc=a if acc is None else acc+a
P=acc/3
from koz3.li_train import labels,site_items,decode_site_v2,N_HM,CLS
from rasterio.features import rasterize
from affine import Affine
from shapely.geometry import shape,Point
from scipy.ndimage import maximum_filter
L=labels(); it={x['region']:x for x in site_items(L)}[reg]; meta=dict(it['meta'],region=reg)
F=decode_site_v2(P,it['valid'],meta,thr=.15,dual=.05,comp_thr=.5,prior={'R':200,'K':3,'snb':.2})
from pyproj import Transformer
tr=Transformer.from_crs(3857,meta['crs'],always_xy=True)
pts=[Point(*tr.transform(*shape(f['geometry']).centroid.coords[0])) for f in F if f['properties']['class_name'].startswith('kurg')]
T=Affine(meta['res'],0,meta['x0'],0,-meta['res'],meta['ytop'])
h=P[0]; mx=(h==maximum_filter(h,size=23)) & it['valid'] & (h>=.03); yy,xx=np.where(mx)
rows=[]
for q in L[(L.region==reg)&L.class_name.str.startswith('kurg')].to_crs(meta['crs']).itertuples():
 g=q.geometry; mask=rasterize([(g,1)],out_shape=h.shape,transform=T).astype(bool)
 hit=any(g.covers(p) for p in pts)
 cy,cx=(~T)*(g.centroid.x,g.centroid.y); cy,cx=float(cx),float(cy)
 dist=np.hypot((xx-cx)*meta['res'],(yy-cy)*meta['res']); j=int(np.argmin(dist))
 correct=CLS.index(q.class_name); wrong=1-correct
 rows.append((q.class_name,hit,float(h[mask].max()),float(dist[j]),float(h[yy[j],xx[j]]),float(P[N_HM+correct][mask].mean()),float(P[N_HM+wrong][mask].mean())))
import pandas as pd
D=pd.DataFrame(rows,columns=['cls','hit','heat_in','nearest_peak_m','nearest_peak_heat','correct_mask','wrong_mask'])
print(D.groupby('hit').agg(n=('hit','size'),heat_in_med=('heat_in','median'),peak_dist_med=('nearest_peak_m','median'),peak_dist_p90=('nearest_peak_m',lambda x:np.quantile(x,.9)),correct_mask=('correct_mask','median'),wrong_mask=('wrong_mask','median')).to_string())
for k in (5,10,20): print('miss nearest <=',k,'m',float((D.loc[~D.hit,'nearest_peak_m']<=k).mean()))
print('miss heat_in >=.15',float((D.loc[~D.hit,'heat_in']>=.15).mean()))
print('miss wrong>correct',float((D.loc[~D.hit,'wrong_mask']>D.loc[~D.hit,'correct_mask']).mean()))
D.to_parquet(ROOT/'work/li/erentuna_heat_diagnostic.parquet')
