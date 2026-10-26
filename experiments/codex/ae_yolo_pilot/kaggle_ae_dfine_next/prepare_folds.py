"""Prepare private, label-safe scale15 fine-tune payloads for folds 0 and 2.

The fold-1 joint source has a known 25-km buffer violation and is excluded.
Copies Claude's sources read-only into *private* independent Kaggle payloads;
never changes the competition tree outside Codex Work.
"""
import argparse
import json
import tarfile
from pathlib import Path

import geopandas as gpd

ROOT=Path('/home/nathan/Expedition Competition')
HERE=Path(__file__).resolve().parent
F3_KERNEL=ROOT/'Codex Work/ae_yolo_pilot/kaggle_ae_dfine_f3/kernel/codex_ae_dfine_ft_f3.py'


def prepare(fold):
    assert fold in (0,2)
    out=HERE/f'fold{fold}';upload=out/'upload';kernel_dir=out/'kernel';stage=out/'stage'
    for p in (upload,kernel_dir,stage):p.mkdir(parents=True,exist_ok=True)
    ann=json.loads((ROOT/'data/ae_prep/ann.json').read_text())
    train={r['region'] for r in ann if r['fold']!=fold}
    held={r['region'] for r in ann if r['fold']==fold}
    assert train.isdisjoint(held)
    safe=[]
    for rec in ann:
        r=dict(rec)
        if r['fold']==fold:r['boxes']=[];r['classes']=[]
        safe.append(r)
    assert all(not r['boxes'] and not r['classes'] for r in safe if r['fold']==fold)
    labels=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    labels=labels[(labels['mod']=='Ae')&labels.region.isin(train)]
    assert not (set(labels.region)&held)
    (stage/'ann.json').write_text(json.dumps(safe,ensure_ascii=False))
    labels.to_parquet(stage/'labels_3857.parquet')
    tar_path=upload/'payload.tar'
    with tarfile.open(tar_path,'w') as tar:
        tar.add(stage/'ann.json',arcname='data/ae_prep/ann.json')
        tar.add(stage/'labels_3857.parquet',arcname='data/labels_3857.parquet')
        tar.add(ROOT/f'work/ae/ae_dfj_f{fold}.pt',arcname=f'work/ae/ae_dfj_f{fold}.pt')
        for p in sorted((ROOT/'koz3').glob('*.py')):tar.add(p,arcname=f'koz3/{p.name}')
        for p in sorted((ROOT/'data/ae_prep/chips').glob('*.jpg')):
            tar.add(p,arcname=f'data/ae_prep/chips/{p.name}')
        for name in ('ae_dfine_finetune.py','ae_dfine_scale.py'):
            p=ROOT/'Codex Work/ae_yolo_pilot'/name;tar.add(p,arcname=name)
    ds_slug=f'codex-koz3-ae-dfine-f{fold}-s15'
    (upload/'dataset-metadata.json').write_text(json.dumps({
        'title':f'Codex KOZ3 Ae D-FINE fold{fold} scale15 private',
        'id':f'solapeolusola/{ds_slug}','licenses':[{'name':'other'}]},indent=2))
    (upload/'manifest.json').write_text(json.dumps({
        'train_sites':sorted(train),'heldout_sites':sorted(held),
        'train_labels':len(labels),'heldout_label_count_in_payload':0,
        'training_fold_excluded':fold,'source_checkpoint':f'ae_dfj_f{fold}.pt',
        'known_buffer_violation':False},ensure_ascii=False,indent=2))
    source=F3_KERNEL.read_text()
    source=source.replace('fold-3 Ae-only fine-tunes','fold-'+str(fold)+' Ae-only scale fine-tune')
    source=source.replace("SLUG='codex-koz3-ae-dfine-f3'",f"SLUG='{ds_slug}'")
    source=source.replace("ROOT=Path('/kaggle/working/ae_dfine_f3')",f"ROOT=Path('/kaggle/working/ae_dfine_f{fold}_s15')")
    source=source.replace('ae_dfj_f3.pt',f'ae_dfj_f{fold}.pt')
    source=source.replace("f'work/ae/{name}_f3.pt'",f"f'work/ae/{{name}}_f{fold}.pt'")
    source=source.replace('fold3.pt',f'fold{fold}.pt')
    source=source.replace('FOLD3_SOURCE',f'FOLD{fold}_SOURCE')
    source=source.replace('private fold-3 joint checkpoint',f'private fold-{fold} joint checkpoint')
    source=source.replace("[('ae_dfine_finetune.py','plain'),('ae_dfine_scale.py','scale15')]",
                          "[('ae_dfine_scale.py','scale15')]")
    source=source.replace("'--fold','3'",f"'--fold','{fold}'")
    source=source.replace("'--eval-fold','3'",f"'--eval-fold','{fold}'")
    source=source.replace('fold3_oof',f'fold{fold}_oof')
    source=source.replace('AE_DFINE_F3_DONE',f'AE_DFINE_F{fold}_DONE')
    assert "'--fold','3'" not in source and "'--eval-fold','3'" not in source
    assert "f'work/ae/{name}_f3.pt'" not in source
    assert f"SLUG='{ds_slug}'" in source
    kernel_slug=f'codex-koz3-ae-dfine-ft-f{fold}-s15'
    code_file=f'{kernel_slug}.py'
    (kernel_dir/code_file).write_text(source)
    (kernel_dir/'kernel-metadata.json').write_text(json.dumps({
        'id':f'solapeolusola/{kernel_slug}','title':kernel_slug,
        'code_file':code_file,'language':'python','kernel_type':'script',
        'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':True,
        'keywords':['gpu'],'dataset_sources':[f'solapeolusola/{ds_slug}'],
        'kernel_sources':[],'competition_sources':[],'model_sources':[],
        'machine_shape':'NvidiaTeslaT4'},indent=2))
    print('fold',fold,'tar_bytes',tar_path.stat().st_size,'train_labels',len(labels),
          'heldout_sites',len(held),'kernel',kernel_slug,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True,choices=(0,2));a=p.parse_args()
    prepare(a.fold)
