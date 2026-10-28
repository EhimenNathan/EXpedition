"""Site-disjoint contextual decoder over frozen DINOv3-SAT token grids.

The existing probe scores every token independently.  This small decoder sees
neighbouring tokens and normalized scene position, so it can distinguish a
coherent hillfort footprint from isolated lookalikes.  Only positive hillfort
scenes are used: scene presence remains the separately validated deployed gate.
"""
import os, sys
import numpy as np

ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..")
sys.path[:0]=[ROOT]

def main(inp="work/hf_dinov3/tokens1024.npz", out="work/hf_dinov3/loc_points_context_oof.csv"):
 import geopandas as gpd, pandas as pd, torch
 import torch.nn as nn
 import torch.nn.functional as F
 from scipy.ndimage import gaussian_filter
 from koz3.sp_prep import site_folds
 torch.manual_seed(123); np.random.seed(123)
 Z=dict(np.load(os.path.join(ROOT,inp),allow_pickle=True)); tok=Z["tok"].astype(np.float32)
 sid=Z["sid"].astype(int); y=Z["y"].astype(np.float32); pos=Z["pos"].astype(np.float32)
 meta=Z["meta"]; ns=len(meta)
 L=gpd.read_parquet(os.path.join(ROOT,"data/labels_3857.parquet")); L=L[L["mod"]=="SpOr"]
 fo=site_folds(L); sf=np.array([fo[m[0]] for m in meta])
 scenes=[]
 for s in range(ns):
  ix=np.flatnonzero(sid==s); h,w=map(int,Z["grid"][s]); assert len(ix)==h*w
  scenes.append((torch.from_numpy(tok[ix].reshape(h,w,-1).transpose(2,0,1)),
                 torch.from_numpy(y[ix].reshape(h,w)),ix))

 class Net(nn.Module):
  def __init__(self,d):
   super().__init__(); self.norm=nn.LayerNorm(d); self.proj=nn.Linear(d,64)
   self.body=nn.Sequential(nn.Conv2d(66,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),
                           nn.Conv2d(64,32,3,padding=1),nn.GroupNorm(8,32),nn.GELU(),nn.Conv2d(32,1,1))
  def forward(self,x):
   # x: D,H,W
   z=self.proj(self.norm(x.permute(1,2,0))).permute(2,0,1)
   h,w=z.shape[-2:]; yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=z.device),torch.linspace(-1,1,w,device=z.device),indexing='ij')
   return self.body(torch.cat([z,xx[None],yy[None]])[None])[0,0]

 dev="cuda" if torch.cuda.is_available() else "cpu"; pred=np.zeros(len(tok),np.float32)
 for fold in sorted(set(sf)):
  tr=[s for s in range(ns) if sf[s]!=fold and bool(scenes[s][1].any())]
  te=[s for s in range(ns) if sf[s]==fold]
  m=Net(tok.shape[1]).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=7e-4,weight_decay=2e-3)
  rng=np.random.default_rng(1000+int(fold)); m.train()
  for it in range(1800):
   s=tr[int(rng.integers(len(tr)))]; x,t,_=scenes[s]; x=x.to(dev);t=t.to(dev)
   k=int(rng.integers(8)); x=torch.rot90(x,k%4,(1,2));t=torch.rot90(t,k%4,(0,1))
   if k>=4:x=x.flip(2);t=t.flip(1)
   q=m(x); pw=min(25.,float((t.numel()-t.sum())/t.sum().clamp(min=1)))
   b=F.binary_cross_entropy_with_logits(q,t,pos_weight=torch.tensor(pw,device=dev))
   p=torch.sigmoid(q); dice=1-(2*(p*t).sum()+1)/((p+t).sum()+1); loss=b+.35*dice
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
   if it in (0,599,1199,1799): print('fold',fold,'it',it,'loss',round(loss.item(),4),flush=True)
  m.eval()
  with torch.inference_mode():
   for s in te:
    x,_,ix=scenes[s]; pred[ix]=torch.sigmoid(m(x.to(dev))).cpu().numpy().ravel()

 rows=[]
 for s in range(ns):
  x,t,ix=scenes[s]; h,w=t.shape; g=gaussian_filter(pred[ix].reshape(h,w),.5,mode="nearest"); a=int(g.argmax())
  rows.append((meta[s][0],meta[s][1],pos[ix][a,2],pos[ix][a,3],float(g.flat[a]),pos[ix][:,2].mean(),pos[ix][:,3].mean(),bool(y[ix].any()),bool(y[ix][a]),int(sf[s])))
 D=pd.DataFrame(rows,columns=["region","tif","x3857","y3857","map_max","cx3857","cy3857","has_hillfort","hit","fold"])
 os.makedirs(os.path.dirname(os.path.join(ROOT,out)),exist_ok=True);D.to_csv(os.path.join(ROOT,out),index=False)
 H=D[D.has_hillfort];print('context hit',round(H.hit.mean(),4),'counts',H.groupby('fold').hit.agg(['sum','count']).to_dict('index'));print('saved',out)

if __name__=="__main__":main(*sys.argv[1:])
