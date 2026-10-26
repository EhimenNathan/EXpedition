"""Re-export predictions from a fixed checkpoint using the audited BGR path."""
import importlib.util
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pilot", here / "codex-koz3-yolo-pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

weights = Path(sys.argv[1]).resolve()
assert weights.exists(), weights
sp = pilot.setup()
scenes = pilot.scene_info(sp)
pilot.export_oof(sp, scenes, weights)
