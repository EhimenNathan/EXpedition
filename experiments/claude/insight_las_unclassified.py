import sys,glob,json,os,warnings,numpy as np,laspy; warnings.filterwarnings('ignore')
sys.path.insert(0,'.'); sys.path.insert(0,'experiments/claude')
from koz3 import dtm
from scipy import ndimage as ndi
from shapely.geometry import shape, Point
from shapely.ops import transform as stf
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
src=glob.glob('data/train-expds-3/021_*/01_*/*.las')[0]; dst=src.replace('.las','_unclassified.las')
if not os.path.exists(dst):
    with laspy.open(src) as r, laspy.open(dst,mode='w',header=r.header) as w:
        for pts in r.chunk_iterator(5_000_000):
            pts.classification[:]=1; w.write_points(pts)
print('ground_fraction delivered %.3f stripped %.3f | stripped treated as ground-only: %s'%(dtm.ground_fraction(src),dtm.ground_fraction(dst),dtm.is_ground_only(dst)))
R=0.5
A,va,(x0,y1,_)=dtm.build(src,res=R); Bm,vb,_=dtm.build(dst,res=R)
utm='EPSG:'+json.load(open('data/train-expds-3/021_Новенькое/UTM.json'))['crs'].split(':')[-1]
polys=[]
for f in glob.glob('data/train-expds-3/021_*/*/*_Li_курганы_*.geojson'):
    d=json.load(open(f)); nm=(d.get('crs') or {}).get('properties',{}).get('name','EPSG:3857')
    tr=Transformer.from_crs('EPSG:'+nm.split(':')[-1],utm,always_xy=True).transform
    polys+=[stf(tr,shape(ft['geometry'])) for ft in d['features'] if ft['geometry'] and not shape(ft['geometry']).is_empty]
T=from_origin(x0,y1,R,R); lab=rasterize([(g,i+1) for i,g in enumerate(polys)],A.shape,transform=T); idx=np.arange(1,len(polys)+1)
v=va&vb; bg=v&(lab==0)
diff=Bm-A
print(f'terrain difference (unclassified − delivered): median {np.median(diff[v]):+.3f} m, p95 |diff| {np.percentile(np.abs(diff[v]),95):.3f} m, share |diff|>0.3 m {np.mean(np.abs(diff[v])>0.3):.3f}')
for s in (2,5):
    La=ndi.gaussian_filter(A-ndi.gaussian_filter(A,s/R),1); Lb=ndi.gaussian_filter(Bm-ndi.gaussian_filter(Bm,s/R),1)
    pa=ndi.maximum(La,lab,idx); pb=ndi.maximum(Lb,lab,idx)
    ca=np.std(La[bg]); cb=np.std(Lb[bg])
    print(f'σ={s} m: mound peak relief delivered {np.median(pa):.3f} m vs unclassified {np.median(pb):.3f} m (corr {np.corrcoef(pa,pb)[0,1]:.3f}); background clutter {ca:.3f} → {cb:.3f} m; SNR {np.median(pa)/ca:.2f} → {np.median(pb)/cb:.2f}')
import li_dome_detector as LD
for name,Dm,vv in (('delivered',A,va),('unclassified',Bm,vb)):
    C,rough,area=LD.candidates(Dm,vv)
    X=x0+(C[:,0]+0.5)*R; Y=y1-(C[:,1]+0.5)*R
    hit=set(); 
    for x,y in zip(X,Y):
        for i,g in enumerate(polys):
            if g.contains(Point(x,y)): hit.add(i); break
    print(f'{name:12s}: {len(C)} dome candidates, {len(hit)}/{len(polys)} kurgans contain one, site roughness {rough:.3f} m')
os.remove(dst)
