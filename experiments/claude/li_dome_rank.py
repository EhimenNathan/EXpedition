import glob,numpy as np,warnings; warnings.filterwarnings('ignore')
from shapely.geometry import Point
from shapely.strtree import STRtree
from shapely.ops import unary_union
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import os; S=os.environ.get('KOZ3_OUT','work/li_dome')
rows=[]
for f in sorted(glob.glob(f'{S}/*.npz')):
    z=np.load(f,allow_pickle=True); C=z['C']; polys=list(z['polys']); rough=float(z['rough']); area=float(z['area'])
    reg=f.split('/')[-1][:-4]
    hull=unary_union(polys).convex_hull.buffer(30); tree=STRtree(polys)
    dens=(C[:,3]>=5).sum()/max(area,1e-6)
    for c in C:
        p=Point(c[0],c[1])
        if not hull.contains(p): continue
        hit=next((int(j) for j in tree.query(p) if polys[j].contains(p)),-1)
        rows.append((reg,hit,c[2],c[3],c[4],c[5],c[6],rough,dens,c[0],c[1]))
import pandas as pd
D=pd.DataFrame(rows,columns=['reg','hit','h','snr','sym','rad','l2','rough','dens','x','y'])
D['y_']=(D.hit>=0).astype(int)
feats=['h','snr','sym','rad','l2','rough','dens']
D['p']=0.0
for r in D.reg.unique():
    tr=D.reg!=r
    m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=15,min_child_samples=40,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=4)
    m.fit(D.loc[tr,feats],D.loc[tr,'y_']); D.loc[~tr,'p']=m.predict_proba(D.loc[~tr,feats])[:,1]
print('leave-site-out AUC: physics ranker %.3f | raw SNR %.3f'%(roc_auc_score(D.y_,D.p),roc_auc_score(D.y_,D.snr)))
nG={r:len(np.load(f'{S}/{r}.npz',allow_pickle=True)['polys']) for r in D.reg.unique()}
f1=lambda tp,fp,fn:2*tp/max(2*tp+fn+.5*fp,1)
def evalsite(G,t,G_n):
    K=G[G.p>=t].sort_values('p',ascending=False); used=set(); tp=0
    for h in K.hit:
        if h>=0 and h not in used: used.add(h); tp+=1
    return tp,len(K)-tp,G_n-tp
ts=np.arange(0.1,0.95,0.05); tot=np.zeros(3)
print(f"{'site':28s} chosen t  TP   FP   FN   F1   (threshold chosen on other sites)")
for r in sorted(D.reg.unique()):
    o=[q for q in D.reg.unique() if q!=r]
    best=max(ts,key=lambda t:f1(*np.sum([evalsite(D[D.reg==q],t,nG[q]) for q in o],0)))
    tp,fp,fn=evalsite(D[D.reg==r],best,nG[r]); tot+=(tp,fp,fn)
    print(f"{r[:28]:28s} {best:.2f} {tp:5d} {fp:5d} {fn:5d} {f1(tp,fp,fn):.3f}")
print('pooled F1 %.3f  TP %d FP %d FN %d'%(f1(*tot),*tot))
E=D[D.reg.str.startswith('011')]
for t in (0.3,0.5,0.7,0.8):
    print('Erentuna at p>=%.1f:'%t, evalsite(E,t,nG[E.reg.iloc[0]]))
D.to_pickle(f'{S}/../li_dome_ranked.pkl')
