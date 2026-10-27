"""Cross-site ring-template matching: template = radial profile stacked at site A; detect at site B (never seen)."""
import sys,glob,json,warnings,numpy as np; warnings.filterwarnings('ignore')
sys.path.insert(0,'.')
from koz3 import dtm
from scipy import ndimage as ndi
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from shapely.ops import transform as stf
from pyproj import Transformer
from multiprocessing import Pool
D='data'; r=0.25
def crs_of(f):
    nm=(json.load(open(f)).get('crs') or {}).get('properties',{}).get('name','EPSG:3857'); return 'EPSG:'+nm.split(':')[-1]
def load(reg):
    las=glob.glob(f'{D}/{reg}/01_*/*.las')[0]; utm='EPSG:'+json.load(open(f'{D}/{reg}/UTM.json'))['crs'].split(':')[-1]
    d,v,(x0,y1,_)=dtm.build(las,res=r); lrm=d-ndi.uniform_filter(d,int(30/r))
    polys=[]
    for f in glob.glob(f'{D}/{reg}/*/*_Li_курганы_*.geojson'):
        tr=Transformer.from_crs(crs_of(f),utm,always_xy=True).transform
        polys+=[stf(tr,shape(ft['geometry'])) for ft in json.load(open(f))['features'] if ft['geometry'] and not shape(ft['geometry']).is_empty]
    return lrm,v,x0,y1,polys
def profile(lrm,x0,y1,polys,R=40):
    S=np.zeros((2*R+1,2*R+1)); n=0
    for g in polys:
        p=g.centroid; c=int((p.x-x0)/r); rr=int((y1-p.y)/r)
        if R<=rr<lrm.shape[0]-R and R<=c<lrm.shape[1]-R: S+=lrm[rr-R:rr+R+1,c-R:c+R+1]; n+=1
    S/=n; yy,xx=np.mgrid[-R:R+1,-R:R+1]; rad=np.hypot(xx,yy)
    prof=np.array([S[(rad>=a)&(rad<a+1)].mean() for a in range(R+1)])
    T=np.interp(rad,np.arange(R+1),prof); T-=T.mean(); return T/np.linalg.norm(T)
def detect(lrm,v,T,thr_list):
    # normalised cross-correlation via FFT-free uniform filters (template zero-mean, unit-norm)
    from scipy.signal import fftconvolve
    num=fftconvolve(lrm,T[::-1,::-1],mode='same')
    k=T.shape[0]; mu=ndi.uniform_filter(lrm,k); sq=ndi.uniform_filter(lrm*lrm,k)
    den=np.sqrt(np.maximum(sq-mu*mu,1e-8)*k*k)
    ncc=np.where(v,num/den,0)
    mx=(ncc==ndi.maximum_filter(ncc,size=int(5/r)))
    return ncc,mx
def evaluate(args):
    src,dst=args
    A=load(src); B=load(dst)
    T=profile(A[0],A[2],A[3],A[4])
    lrm,v,x0,y1,polys=B; ncc,mx=detect(lrm,v,T,None)
    tree=STRtree(polys); out=[]
    # only score inside the labelled area's convex hull + 30 m (labels exist only near the site)
    from shapely.ops import unary_union
    hull=unary_union(polys).convex_hull.buffer(30)
    rr,cc=np.nonzero(mx); X=x0+(cc+0.5)*r; Y=y1-(rr+0.5)*r; s=ncc[rr,cc]
    inh=np.array([hull.contains(Point(a,b)) for a,b in zip(X,Y)])
    X,Y,s=X[inh],Y[inh],s[inh]; o=np.argsort(-s); X,Y,s=X[o],Y[o],s[o]
    used=set(); hit=np.zeros(len(s),bool)
    for i,(a,b) in enumerate(zip(X,Y)):
        for j in tree.query(Point(a,b)):
            if j not in used and polys[j].contains(Point(a,b)): used.add(j); hit[i]=True; break
    G=len(polys); res=[]
    for t in (0.3,0.4,0.5,0.6,0.7):
        k=s>=t; tp=hit[k].sum(); fp=k.sum()-tp; f1=2*tp/max(2*tp+(G-tp)+0.5*fp,1)
        res.append(f"t{t}: TP {tp:3d} FP {fp:4d} R {tp/G:.2f} F1 {f1:.3f}")
    return f"template {src[:12]} → {dst[:26]} (GT {G}): "+" | ".join(res)
if __name__=='__main__':
    pairs=[('036_Тимерево','102_Гнёздово_Заольшанская'),('180_Гнёздово_Лесная_2','102_Гнёздово_Заольшанская'),
           ('102_Гнёздово_Заольшанская','036_Тимерево'),('180_Гнёздово_Лесная_2','036_Тимерево')]
    with Pool(4) as p:
        for line in p.imap(evaluate,pairs): print(line,flush=True)
