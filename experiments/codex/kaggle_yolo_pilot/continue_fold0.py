"""Fixed-length continuation of the independent YOLO fold-0 pilot."""
import importlib.util
import os
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pilot", HERE / "codex-koz3-yolo-pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

base = HERE / "local_run_f0"
initial = base / "fold0/weights/last.pt"
assert initial.exists()
model = YOLO(str(initial))
model.train(data=str(base / "data.yaml"), epochs=60, imgsz=640, batch=8, device=0, workers=4,
            project=str(base), name="fold0_continue", exist_ok=True, seed=20261023,
            val=False, plots=False, save=True, verbose=True, patience=61)
weights = base / "fold0_continue/weights/last.pt"
assert weights.exists()
sp = pilot.setup()
scenes = pilot.scene_info(sp)
pilot.export_oof(sp, scenes, weights)
