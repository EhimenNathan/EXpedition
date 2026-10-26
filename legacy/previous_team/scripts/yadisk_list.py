import requests, json, sys, time
API="https://cloud-api.yandex.net/v1/disk/public/resources"
S=requests.Session()
def get(params):
    for t in range(12):
        try:
            r=S.get(API,params=params,timeout=90)
            if r.status_code==200: return r.json()
            print("HTTP",r.status_code,r.text[:200],file=sys.stderr)
            if r.status_code in (404,403): return r.json()
        except Exception as e: print("retry",t,type(e).__name__,file=sys.stderr)
        time.sleep(3+4*t)
    raise SystemExit("gave up "+str(params))
def walk(key, path="/", out=None):
    out=[] if out is None else out; off=0
    while True:
        j=get({"public_key":key,"path":path,"limit":500,"offset":off,"fields":"name,type,size,md5,path,mime_type,_embedded.items.name,_embedded.items.type,_embedded.items.size,_embedded.items.md5,_embedded.items.path,_embedded.items.mime_type,_embedded.total"})
        if "_embedded" not in j:
            out.append(dict(path=j.get("path",path),type=j.get("type"),size=j.get("size"),md5=j.get("md5"),name=j.get("name"),mime=j.get("mime_type"))); return out
        items=j["_embedded"]["items"]
        for it in items:
            if it["type"]=="dir": walk(key,it["path"],out)
            else: out.append(dict(path=it["path"],type="file",size=it.get("size"),md5=it.get("md5"),name=it["name"],mime=it.get("mime_type")))
        off+=len(items)
        if off>=j["_embedded"].get("total",0) or not items: break
    return out
res=walk(sys.argv[1])
json.dump(res,open(sys.argv[2],"w",encoding="utf-8"),ensure_ascii=False,indent=0)
print(sys.argv[2],"entries",len(res),"GB",round(sum((x["size"] or 0) for x in res)/1e9,2))
