import sys,glob,json,warnings,numpy as np; warnings.filterwarnings('ignore')
sys.path.insert(0,'.')
from koz3 import dtm
from scipy import ndimage as ndi
from shapely.geometry import shape
from pyproj import Transformer
from PIL import Image
D='data'; OUT='work/insights'
def crs_of(f):
    nm=(json.load(open(f)).get('crs') or {}).get('properties',{}).get('name','EPSG:3857'); return 'EPSG:'+nm.split(':')[-1]
tiles=[]
for reg in sys.argv[1:]:
    las=glob.glob(f'{D}/{reg}/01_*/*.las')[0]; utm='EPSG:'+json.load(open(f'{D}/{reg}/UTM.json'))['crs'].split(':')[-1]
    r=0.25; d,v,(x0,y1,r)=dtm.build(las,res=r)
    lrm=d-ndi.uniform_filter(d,int(30/r))
    pts=[];diam=[]
    for f in glob.glob(f'{D}/{reg}/*/*_Li_курганы_*.geojson'):
        tr=Transformer.from_crs(crs_of(f),utm,always_xy=True)
        for ft in json.load(open(f))['features']:
            if ft['geometry'] and not shape(ft['geometry']).is_empty:
                g=shape(ft['geometry']); p=g.centroid; pts.append(tr.transform(p.x,p.y))
    R=int(15/r); S=np.zeros((2*R+1,2*R+1)); n=0
    for x,y in pts:
        c=int((x-x0)/r); rr=int((y1-y)/r)
        if R<=rr<d.shape[0]-R and R<=c<d.shape[1]-R:
            S+=lrm[rr-R:rr+R+1,c-R:c+R+1]; n+=1
    S/=n
    yy,xx=np.mgrid[-R:R+1,-R:R+1]*r
    w=np.clip(S,0,None); cx=(w*xx).sum()/w.sum(); cy=-(w*yy).sum()/w.sum()
    iy,ix=np.unravel_index(S.argmax(),S.shape)
    rad=np.hypot(xx,yy); prof=[S[(rad>=a)&(rad<a+1)].mean() for a in range(0,12)]
    print(f"{reg[:28]:28s} n={n:4d} stacked relief: centre {S[R,R]:+.3f} m, max {S.max():+.3f} m at E{ix*r-15:+.1f} N{15-iy*r:+.1f} m, "
          f"positive-mass centroid E{cx:+.1f} N{cy:+.1f} m | radial profile 0-11 m: {' '.join(f'{p:+.2f}' for p in prof)}")
    im=(S-S.min())/(S.max()-S.min()+1e-9)*255; tiles.append(np.kron(im,np.ones((3,3))))
Image.fromarray(np.hstack(tiles).astype(np.uint8)).save(f'{OUT}/stack.png')
