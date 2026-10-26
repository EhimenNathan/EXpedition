import glob,os,sys
print('python',sys.version,flush=True)
print('input exists',os.path.isdir('/kaggle/input'),flush=True)
print('top',os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else [],flush=True)
for pat in ('/kaggle/input/**/koz3_code.tar.gz','/kaggle/input/**/koz3/sp_io.py','/kaggle/input/**/sp/scenes.csv','/kaggle/input/**/site_groups.csv'):
 print(pat,glob.glob(pat,recursive=True)[:10],flush=True)
