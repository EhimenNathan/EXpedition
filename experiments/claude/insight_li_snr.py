import sys,glob,json,warnings,numpy as np; warnings.filterwarnings('ignore')
sys.path.insert(0,'.')
from koz3 import dtm
from scipy import ndimage as ndi
from shapely.geometry import shape
from shapely.ops import transform as stf
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from multiprocessing import Pool
D='data'; r=0.5
def crs_of(f):
    nm=(json.load(open(f)).get('crs') or {}).get('properties',{}).get('name','EPSG:3857'); return 'EPSG:4326' if 'CRS84' in nm else 'EPSG:'+nm.split(':')[-1]
def site(reg):
    las=glob.glob(f'{D}/{reg}/01_*/*.las')[0]; utm='EPSG:'+json.load(open(f'{D}/{reg}/UTM.json'))['crs'].split(':')[-1]
    d,v,(x0,y1,_)=dtm.build(las,res=r)
    polys=[]
    for f in glob.glob(f'{D}/{reg}/*/*_Li_*.geojson'):
        tr=Transformer.from_crs(crs_of(f),utm,always_xy=True).transform
        polys+=[(stf(tr,shape(ft['geometry'])),'курган' in f) for ft in json.load(open(f))['features'] if ft['geometry'] and not shape(ft['geometry']).is_empty]
    T=from_origin(x0,y1,r,r); H,W=d.shape
    kur=[g for g,k in polys if k]
    lab=rasterize([(g,i+1) for i,g in enumerate(kur)],(H,W),transform=T)
    anyobj=rasterize([(g.buffer(10),1) for g,_ in polys],(H,W),transform=T)>0
    bg=v&~anyobj
    out=[reg,len(kur)]
    for s in (2.0,5.0,10.0):
        L=d-ndi.gaussian_filter(d,s/r)
        sig=np.median(ndi.maximum(L,lab,np.arange(1,len(kur)+1)))
        clutter=np.std(L[bg])
        # natural look-alikes: background local maxima at least half the median mound signal, per km² of background
        mx=(L==ndi.maximum_filter(L,size=max(3,int(2*s/r))))&bg&(L>=0.5*sig)
        out+= [sig, clutter, sig/clutter, mx.sum()/(bg.sum()*r*r/1e6)]
    slope=np.degrees(np.arctan(np.hypot(*np.gradient(ndi.gaussian_filter(d,2),r))))
    out+=[np.median(slope[lab>0]), np.median(slope[bg])]
    return out
if __name__=='__main__':
    regs=sys.argv[1:] or sorted({p.split('/')[-3] for p in glob.glob(f'{D}/*/01_*/*.las')})
    print(f"{'site':26s} {'n':>4s} | {'σ=2m: signal clutter SNR lookalikes/km²':^42s} | {'σ=5m':^30s} | {'σ=10m':^30s} | slope° mound/bg")
    with Pool(3) as p:
        for o in p.imap(site,regs):
            s=" | ".join(f"{o[2+4*i]:.2f} {o[3+4*i]:.3f} {o[4+4*i]:5.1f} {o[5+4*i]:6.0f}" for i in range(3))
            print(f"{o[0][:26]:26s} {o[1]:4d} | {s} | {o[14]:.1f}/{o[15]:.1f}",flush=True)
