"""Check every organiser Yandex.Disk link for re-uploads (md5 / size / modified date) vs the last snapshot.

Usage: python scripts/org_links_check.py [--download DIR]   (download fetches new/changed single files)
"""
import json, os, sys, datetime as dt, urllib.parse, urllib.request

LINKS = {
    "compute_metrics.py": "https://disk.yandex.ru/d/IpeAACooPQ_93A",
    "simple_solution.zip": "https://disk.yandex.ru/d/idZF6YWSJa7_IA",
    "solution_submission.geojson": "https://disk.yandex.ru/d/9bs86z2apphJjw",
    "baseline_solution.zip": "https://disk.yandex.ru/d/mL6Lbj4q_D_FNA",
    "baseline_description": "https://disk.yandex.ru/d/csuxYtjpY3Lr9w",
    "s3_sync_script": "https://disk.yandex.ru/d/9RdM4Xe48osDnQ",
    "val_data_structure": "https://disk.yandex.ru/d/M8w4uWqxAJtxyg",
    "utm_zones_table": "https://disk.yandex.ru/i/740wlXuu5y3HCA",
    "class_descriptions": "https://disk.yandex.ru/i/ZF0Rs1N2Ud8Zzw",
    "archaeologists_qa": "https://disk.yandex.ru/i/hPX4NSor-OsKvQ",
    "sudbishchi_mag_points": "https://disk.yandex.ru/d/vi9MnE7-GMah8g",
    "consultation1_slides": "https://disk.yandex.ru/i/E5E6rVFCVZxAVQ",
    "consultation2_answers": "https://disk.yandex.ru/i/BtHNOwtig6w_0Q",
    "consultation2_slides": "https://disk.yandex.ru/i/g-AVV9YnKyjWRw",
    "readme_template": "https://disk.yandex.ru/d/kWyoJwzgJsRyxA",
    "ai_declaration_template": "https://disk.yandex.ru/d/YpCpLF8WEUsg0w",
    "task_description_pdf": "https://disk.yandex.ru/d/0cgaxJKEQ5X0Ow",
    "useful_link_user": "https://disk.yandex.ru/i/5OuBOqjxXSdQrg",
    "train_dataset_mirror": "https://disk.yandex.ru/d/fLYrcMEDnv9_Mw",
}
API = "https://cloud-api.yandex.net/v1/disk/public/resources"
SNAP = os.path.join(os.path.dirname(__file__), "..", "docs", "org_links_snapshot.json")


def meta(url):
    q = urllib.parse.urlencode({"public_key": url, "limit": 200, "fields": "name,type,size,md5,modified,_embedded.items.name,_embedded.items.type,_embedded.items.size,_embedded.items.md5,_embedded.items.modified,_embedded.items.path,_embedded.total"})
    with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
        d = json.load(r)
    out = {"name": d.get("name"), "type": d.get("type"), "size": d.get("size"), "md5": d.get("md5"), "modified": d.get("modified")}
    if d.get("type") == "dir":
        out["items"] = {i["name"]: [i.get("type"), i.get("size"), i.get("md5"), i.get("modified")] for i in d.get("_embedded", {}).get("items", [])}
    return out


def download(url, dest, path=None):
    q = {"public_key": url}
    if path: q["path"] = path
    with urllib.request.urlopen(f"{API}/download?{urllib.parse.urlencode(q)}", timeout=60) as r:
        href = json.load(r)["href"]
    urllib.request.urlretrieve(href, dest)


def main():
    dl = sys.argv[sys.argv.index("--download") + 1] if "--download" in sys.argv else None
    prev = json.load(open(SNAP, encoding="utf-8")) if os.path.exists(SNAP) else {}
    cur = {}
    for key, url in LINKS.items():
        try:
            cur[key] = meta(url)
        except Exception as e:
            cur[key] = {"error": str(e)}
        m, p = cur[key], prev.get(key)
        status = "NEW" if p is None else ("SAME" if p == m else "CHANGED")
        print(f"{status:8s} {key:28s} {m.get('type','?'):5s} {m.get('modified','')[:19]}  {m.get('name') or m.get('error')}")
        if status == "CHANGED" and "items" in m:
            for n, v in m["items"].items():
                if (p.get("items") or {}).get(n) != v:
                    print(f"           ~ {n}  {v[3][:19] if v[3] else ''}")
        if dl and m.get("type") == "file" and status != "SAME":
            os.makedirs(dl, exist_ok=True)
            download(url, os.path.join(dl, m["name"]))
            print("           downloaded ->", os.path.join(dl, m["name"]))
    json.dump(cur, open(SNAP, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("snapshot", dt.datetime.now().isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()
