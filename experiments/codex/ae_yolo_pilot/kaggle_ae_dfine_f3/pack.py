"""Build a private, label-safe Kaggle payload for Ae D-FINE fold-3 fine-tune."""
import io
import json
import tarfile
from pathlib import Path

import geopandas as gpd

ROOT=Path('/home/nathan/Expedition Competition')
HERE=ROOT/'Codex Work/ae_yolo_pilot/kaggle_ae_dfine_f3'
HERE.mkdir(parents=True,exist_ok=True)
UPLOAD=HERE/'upload';UPLOAD.mkdir(exist_ok=True)
ANN=json.loads((ROOT/'data/ae_prep/ann.json').read_text())
train={r['region'] for r in ANN if r['fold']!=3}
held={r['region'] for r in ANN if r['fold']==3}
assert train.isdisjoint(held)
safe=[]
for r in ANN:
    r=dict(r)
    if r['fold']==3:
        r.pop('boxes',None);r.pop('classes',None)
        r['boxes']=[];r['classes']=[]
    safe.append(r)
assert all(not r['boxes'] and not r['classes'] for r in safe if r['fold']==3)
labels=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
labels=labels[(labels['mod']=='Ae')&labels.region.isin(train)]
assert not (set(labels.region)&held)
stage=HERE/'stage';stage.mkdir(exist_ok=True)
labels.to_parquet(stage/'labels_3857.parquet')
(stage/'ann.json').write_text(json.dumps(safe,ensure_ascii=False))
config=next((Path('/home/nathan/.cache/huggingface/hub/models--ustc-community--dfine-large-coco/snapshots')).glob('*/config.json'))
tar_path=UPLOAD/'payload.tar'
with tarfile.open(tar_path,'w') as tar:
    tar.add(stage/'ann.json',arcname='data/ae_prep/ann.json')
    tar.add(stage/'labels_3857.parquet',arcname='data/labels_3857.parquet')
    tar.add(config,arcname='hf_dfine_config/config.json')
    tar.add(ROOT/'work/ae/ae_dfj_f3.pt',arcname='work/ae/ae_dfj_f3.pt')
    for p in sorted((ROOT/'koz3').glob('*.py')):
        tar.add(p,arcname=f'koz3/{p.name}')
    for p in sorted((ROOT/'data/ae_prep/chips').glob('*.jpg')):
        tar.add(p,arcname=f'data/ae_prep/chips/{p.name}')
    tar.add(ROOT/'Codex Work/ae_yolo_pilot/ae_dfine_finetune.py',arcname='ae_dfine_finetune.py')
    tar.add(ROOT/'Codex Work/ae_yolo_pilot/ae_dfine_scale.py',arcname='ae_dfine_scale.py')
print('payload',tar_path,tar_path.stat().st_size,'bytes','train_sites',len(train),'held_sites',len(held),
      'train_labels',len(labels),'chips',len(ANN))
(UPLOAD/'dataset-metadata.json').write_text(json.dumps({
    'title':'Codex KOZ3 Ae D-FINE fold3 private',
    'id':'solapeolusola/codex-koz3-ae-dfine-f3',
    'licenses':[{'name':'other'}],
},indent=2))
(UPLOAD/'manifest.json').write_text(json.dumps({
    'train_sites':sorted(train),'heldout_sites':sorted(held),
    'train_labels':len(labels),'heldout_label_count_in_payload':0,
    'training_fold_excluded':3,'source_checkpoint':'ae_dfj_f3.pt',
},ensure_ascii=False,indent=2))
