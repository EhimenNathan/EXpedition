import json
from pathlib import Path
import cv2,geopandas as gpd,numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
R=Path('/home/nathan/Expedition Competition');a=json.loads((R/'data/ae_prep/ann.json').read_text());by={}
for r in a:by.setdefault(r['region'],[]).append(r)
g=gpd.read_parquet(R/'data/labels_3857.parquet');all=[]
for site,chs in by.items():
 li=g[(g.region==site)&(g['mod']=='Li')&g.class_name.isin(['kurgany_povrezhdennye','kurgany_tselye'])]
 ae=g[(g.region==site)&(g['mod']=='Ae')&g.class_name.isin(['kurgany_povrezhdennye','kurgany_tselye'])]
 if li.empty:continue
 tr=Transformer.from_crs(3857,chs[0]['crs'],always_xy=True)
 for cls,z in li.groupby('class_name'):
  aa=ae[ae.class_name==cls];v=np.array([(q.centroid.x,q.centroid.y) for q in aa.geometry]);tree=cKDTree(v) if len(v) else None
  for q in z.geometry:
   c=q.centroid;ux,uy=tr.transform(c.x,c.y)
   if tree is not None:
    dist=tree.query([c.x,c.y])[0];lat=2*np.arctan(np.exp(c.y/6378137.))-np.pi/2;dist*=np.cos(lat)
   else:dist=float('inf')
   if dist<=10:continue
   best=None
   for ch in chs:
    aa0,b,x0,d,e,y0=ch['transform'];px=(ux-x0)/aa0;py=(uy-y0)/e
    if 0<=px<ch['w'] and 0<=py<ch['h']:
     m=min(px,py,ch['w']-px,ch['h']-py)
     if best is None or m>best[0]:best=(m,ch,px,py,abs(aa0))
   if best is None or best[0]<60:continue
   all.append((site,cls,*best))
keep=[]
for site in ['179_Гнёздово_Лесная_1','180_Гнёздово_Лесная_2','181_Гнёздово_Лесная_3','017_Юшково','102_Гнёздово_Заольшанская']:
 z=[q for q in all if q[0]==site]
 if not z:continue
 rng=np.random.default_rng(20261024)
 for q in [z[i] for i in rng.choice(len(z),min(4,len(z)),replace=False)]:keep.append(q)
canvas=np.full((len(keep)//5*190,5*190,3),255,np.uint8)
for i,(site,cls,m,ch,px,py,gsd) in enumerate(keep):
 im=cv2.imread(str(R/'data/ae_prep/chips'/ch['file']))
 rad=min(100,max(30,int(45/gsd)));crop=cv2.getRectSubPix(im,(2*rad,2*rad),(float(px),float(py)))
 crop=cv2.resize(crop,(180,160));cv2.circle(crop,(90,80),4,(0,0,255),1)
 text=f'{site.split("_")[0]} {"D" if cls.endswith("povrezhdennye") else "I"}'
 y=(i//5)*190;x=(i%5)*190;canvas[y:y+160,x:x+180]=crop
 cv2.putText(canvas,text,(x+2,y+178),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
out=R/'Codex Work/ae_yolo_pilot/li_transfer_sheet.jpg';cv2.imwrite(str(out),canvas)
print('saved',out,'candidates',len(all),'shown',len(keep))
