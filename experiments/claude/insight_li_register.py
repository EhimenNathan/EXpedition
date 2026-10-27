import sys,glob,json,warnings,numpy as np; warnings.filterwarnings('ignore')
sys.path.insert(0,'.')
from koz3 import dtm
from scipy import ndimage as ndi
from shapely.geometry import shape
from pyproj import Transformer
from multiprocessing import Pool
D='data'
def crs_of(f):
    nm=(json.load(open(f)).get('crs') or {}).get('properties',{}).get('name','EPSG:3857')
    return 'EPSG:4326' if 'CRS84' in nm else 'EPSG:'+nm.split(':')[-1]
def site(reg):
    las=glob.glob(f'{D}/{reg}/01_*/*.las')[0]
    utm='EPSG:'+json.load(open(f'{D}/{reg}/UTM.json'))['crs'].split(':')[-1]
    res=0.5
    d,v,(x0,y1,r)=dtm.build(las,res=res)
    lrm=ndi.gaussian_filter(d-ndi.uniform_filter(d,int(20/r)),1)
    pts=[]
    for f in glob.glob(f'{D}/{reg}/*/*_Li_курганы_*.geojson'):
        tr=Transformer.from_crs(crs_of(f),utm,always_xy=True)
        for ft in json.load(open(f))['features']:
            if ft['geometry'] and not shape(ft['geometry']).is_empty:
                p=shape(ft['geometry']).representative_point(); pts.append(tr.transform(p.x,p.y))
    P=np.array(pts); H,W=d.shape
    col=(P[:,0]-x0)/r; row=(y1-P[:,1])/r
    sh=np.arange(-15,15.01,0.5); best=(-9,0,0); grid=np.zeros((len(sh),len(sh)))
    for i,dy in enumerate(sh):
        for j,dx in enumerate(sh):
            cc=np.round(col+dx/r).astype(int); rr=np.round(row-dy/r).astype(int)
            ok=(cc>=0)&(cc<W)&(rr>=0)&(rr<H)
            val=lrm[rr[ok],cc[ok]].mean() if ok.any() else np.nan
            grid[i,j]=val
    i,j=np.unravel_index(np.nanargmax(grid),grid.shape)
    z=len(sh)//2
    inside=((col>=0)&(col<W)&(row>=0)&(row<H)).mean()
    vv=v[np.clip(np.round(row).astype(int),0,H-1),np.clip(np.round(col).astype(int),0,W-1)].mean()
    return reg,len(P),r,grid[z,z],grid[i,j],sh[j],sh[i],np.nanmedian(grid),inside,vv
if __name__=='__main__':
    regs=sys.argv[1:] or sorted({p.split('/')[-3] for p in glob.glob(f'{D}/*/01_*/*.las')})
    with Pool(3) as pool:
        for out in pool.imap_unordered(site,regs):
            reg,n,r,s0,sb,dx,dy,med,inside,vv=out
            print(f"{reg[:28]:28s} n={n:4d} relief@labels {s0:+.3f} m | best shift dx={dx:+5.1f} dy={dy:+5.1f} m → {sb:+.3f} m | bg {med:+.3f} | labels inside cloud {inside:.2f} valid {vv:.2f}",flush=True)
