import sys,os,json,glob,warnings,numpy as np; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import sp_dinov3_probe as P
import rasterio
from shapely.geometry import shape
from shapely.ops import transform as stf
from pyproj import Transformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageDraw, ImageFont
P.D=os.environ.get('KOZ3_RAW','data/train-expds-3'); S=os.environ.get('KOZ3_OUT','work/dinov3_probe')
z=np.load(f'{S}/emb.npz'); X,B,y,sid=z['X'],z['B'],z['y'],z['sid']
# leave-one-scene-out probabilities for both models
Xs=StandardScaler().fit_transform(X); pd_=np.zeros(len(y)); pb=np.zeros(len(y))
for s in np.unique(sid):
    tr=sid!=s
    pd_[~tr]=LogisticRegression(max_iter=2000,C=0.5,class_weight='balanced').fit(Xs[tr],y[tr]).predict_proba(Xs[~tr])[:,1]
    pb[~tr]=LogisticRegression(max_iter=2000,C=0.5,class_weight='balanced').fit(B[tr],y[tr]).predict_proba(B[~tr])[:,1]
# rebuild the scene list and candidate coordinates exactly as the probe did
scenes=[]
for reg in sorted(os.listdir(P.D)):
    tifs=glob.glob(f'{P.D}/{reg}/04_*/*.tif'); labs=glob.glob(f'{P.D}/{reg}/*/*_SpOr_курганы_*.geojson')
    if len(tifs)!=1 or not labs: continue
    polys=[]
    for f in labs:
        d=json.load(open(f)); n=(d.get('crs') or {}).get('properties',{}).get('name','EPSG:3857')
        tr=Transformer.from_crs('EPSG:'+n.split(':')[-1],3857,always_xy=True).transform
        polys+=[stf(tr,shape(ft['geometry'])) for ft in d['features'] if ft['geometry'] and not shape(ft['geometry']).is_empty]
    if len(polys)>=15: scenes.append((os.path.getsize(tifs[0]),reg,tifs[0],polys))
scenes=sorted(scenes)[:20]
want={'142_Курганы_13','139_Курганы_10','096_Иссык_Куль_курганы_и_постройки_2'}
font=ImageFont.load_default()
rows=[]
for k,(_,reg,tif,polys) in enumerate(scenes):
    if reg not in want: continue
    a,T,crs,res,cands=P.scene_candidates(tif,polys)
    xy=[(x,yy) for x,yy,*_ in cands if P.chip(a,T,res,x,yy) is not None]
    m=sid==k; assert m.sum()==len(xy),(reg,m.sum(),len(xy))
    yy_=y[m]; scores={'Blob appearance (baseline)':pb[m],'DINOv3-SAT (frozen + linear)':pd_[m]}
    img=np.clip(a.transpose(1,2,0),0,255).astype(np.uint8)
    if img.shape[2]==1: img=np.repeat(img,3,2)
    to=Transformer.from_crs(3857,crs,always_xy=True).transform; inv=~T
    N=len(polys); panels=[]
    for name,sc in scores.items():
        im=Image.fromarray(img).convert('RGB'); dr=ImageDraw.Draw(im)
        for g in polys:
            g2=stf(to,g); ring=g2.exterior if g2.geom_type=='Polygon' else max(g2.geoms,key=lambda q:q.area).exterior
            dr.line([inv*(px,py) for px,py in ring.coords],fill=(255,215,0),width=2)
        top=np.argsort(-sc)[:N]; tp=int(yy_[top].sum())
        for i in top:
            c,r=inv*xy[i]; col=(40,220,90) if yy_[i] else (235,50,50)
            dr.ellipse([c-9,r-9,c+9,r+9],outline=col,width=3)
        W=900; im=im.resize((W,int(im.height*W/im.width)))
        hdr=Image.new('RGB',(W,34),(20,24,22)); ImageDraw.Draw(hdr).text((8,10),f'{reg}  |  {name}: top-{N} candidates → {tp} in kurgans ({tp/N:.0%})',fill=(235,235,235),font=font)
        pan=Image.new('RGB',(W,im.height+34)); pan.paste(hdr,(0,0)); pan.paste(im,(0,34)); panels.append(pan)
        print(reg,name,f'precision@{N}: {tp}/{N}')
    row=Image.new('RGB',(sum(p.width for p in panels)+10,max(p.height for p in panels)),(255,255,255))
    x=0
    for p in panels: row.paste(p,(x,0)); x+=p.width+10
    rows.append(row)
for i,r in enumerate(rows): r.save(f'{S}/dinov3_vs_baseline_{i+1}.png')
