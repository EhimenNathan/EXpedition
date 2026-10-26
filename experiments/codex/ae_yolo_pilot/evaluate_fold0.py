"""Official-metric diagnostic on all nine held-out Ae fold-0 sites."""
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path("/home/nathan/Expedition Competition")
PILOT = Path(os.environ.get("KOZ3_AE_EVAL_OUT", Path(__file__).resolve().parent / "fold0"))
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc


def evaluate(name, data, gt, threshold):
    selected = data[data.score >= threshold]
    overall, details = score(fc(selected), gt)
    classes = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii"]
    summary = {c: (round(details[c]["f1"], 3), details[c]["tp"], details[c]["fp"], details[c]["fn"])
               for c in classes}
    print(name, "t", threshold, "score", round(overall, 5), "n", len(selected), summary, flush=True)


def main():
    manifest = json.loads((PILOT / "manifest.json").read_text())
    sites = set(manifest["heldout_sites"])
    L = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    L = L[(L["mod"] == "Ae") & L.region.isin(sites)]
    gt = to_gt_geojson(L)
    print("heldout sites", len(sites), "GT", len(L), flush=True)
    deployed = pd.read_parquet(ROOT / "work/ae/oof_spdet_cands.parquet")
    deployed = deployed[(deployed.region.isin(sites)) & (deployed.cls == "kurgany_povrezhdennye")]
    evaluate("deployed", deployed, gt, .7)
    fcos = pd.read_parquet(ROOT / "work/ae/oof_ae_fcos_fast_cands.parquet")
    fcos = fcos[fcos.region.isin(sites)]
    evaluate("fcos_reference", fcos, gt, .4)
    p = PILOT / "oof_merged.parquet"
    if p.exists():
        yolo = pd.read_parquet(p)
        yolo = yolo[yolo.region.isin(sites)]
        # Fixed before looking at fold-0 labels. Any later calibration needs
        # other geographic folds and an untouched final fold.
        evaluate("yolo_primary", yolo, gt, .25)


if __name__ == "__main__":
    main()
