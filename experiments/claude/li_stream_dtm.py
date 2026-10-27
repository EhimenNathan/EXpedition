"""Stream every training LAS from the organisers' bucket → 0.5 m DTM cache (OUT/dtm/<region>.npz) → delete the LAS.

Keeps disk use to one or two clouds at a time; LAS files already present locally are used and kept. The cached DTMs
feed the physics experiments (dome candidates, trench detection) without re-downloading.

  KOZ3_S3_ACCESS_KEY=… KOZ3_S3_SECRET_KEY=… KOZ3_RAW=data KOZ3_OUT=work/li_dome python experiments/claude/li_stream_dtm.py [workers=2]
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from koz3 import dtm  # noqa: E402

D = os.environ.get("KOZ3_RAW", "data"); OUT = os.environ.get("KOZ3_OUT", "work/li_dome"); R = 0.5


def one(key_size):
    import s3_changes as s
    key, size = key_size
    reg = key.split("/")[0]; out = f"{OUT}/dtm/{reg}.npz"
    if os.path.exists(out):
        return reg, "cached"
    las = os.path.join(D, key); downloaded = False
    if not (os.path.exists(las) and os.path.getsize(las) == size):
        os.makedirs(os.path.dirname(las), exist_ok=True)
        for a in range(4):
            try:
                s.client().download_file(s.BUCKET, key, las); break
            except Exception as e:
                time.sleep(5 * (a + 1))
        downloaded = True
    t = time.time()
    d, v, (x0, y1, r) = dtm.build(las, res=R)
    os.makedirs(f"{OUT}/dtm", exist_ok=True)
    np.savez_compressed(out, d=d.astype(np.float32), v=v, origin=np.array([x0, y1, r]))
    if downloaded:
        os.remove(las)
    return reg, f"{d.shape} {size / 1e6:.0f} MB {time.time() - t:.0f}s"


def main(workers=2):
    import s3_changes as s
    from concurrent.futures import ProcessPoolExecutor
    rows = s.listing()
    keys = sorted(((k, v[0]) for k, v in rows.items() if k.lower().endswith(".las")), key=lambda kv: kv[1])
    print(len(keys), "LAS files", flush=True)
    with ProcessPoolExecutor(int(workers)) as ex:
        for reg, msg in ex.map(one, keys):
            print("dtm", reg, msg, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
