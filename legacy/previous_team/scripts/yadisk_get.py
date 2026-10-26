import requests, sys, time, os
API="https://cloud-api.yandex.net/v1/disk/public/resources/download"
key,path,out=sys.argv[1],sys.argv[2],sys.argv[3]
for t in range(10):
    try:
        p={"public_key":key};
        if path!="/": p["path"]=path
        href=requests.get(API,params=p,timeout=60).json()["href"]
        with requests.get(href,stream=True,timeout=120) as r:
            r.raise_for_status(); os.makedirs(os.path.dirname(out) or ".",exist_ok=True)
            with open(out,"wb") as f:
                for c in r.iter_content(1<<20): f.write(c)
        print("ok",out,os.path.getsize(out)); break
    except Exception as e: print("retry",t,e); time.sleep(3+3*t)
