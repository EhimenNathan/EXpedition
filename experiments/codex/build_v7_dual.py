"""Build Codex's own dual-label variant from Claude's immutable v7 ZIP.

Only inference/koz3_infer/run.py and inference/models/config.json change.
The source archive is read-only and is never extracted into Claude's tree.
"""
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "build/koz3_submission_v7_hillfort_gate_20261023.zip"
TARGET = HERE / "codex_koz3_v7_dual_v1.zip"
OLD = HERE / "candidate_dual/inference/koz3_infer/run.py"


def main():
    old = OLD.read_text()
    start = old.index("    # Held-out geographic folds support a low-confidence damaged-kurgan backup")
    end = old.index("    try:\n        from .models import cfg\n        fc = cfg().get(\"fuse\"", start)
    addition = old[start:end]
    needle = "    try:\n        from .models import cfg\n        fc = cfg().get(\"fuse\""
    assert addition.count("sp_dual") == 1
    with zipfile.ZipFile(SOURCE) as src, zipfile.ZipFile(TARGET, "w") as dst:
        assert "metadata.json" in src.namelist()
        assert src.namelist().count("inference/koz3_infer/run.py") == 1
        assert src.namelist().count("inference/models/config.json") == 1
        for info in src.infolist():
            payload = src.read(info.filename)
            if info.filename == "inference/koz3_infer/run.py":
                text = payload.decode()
                assert text.count(needle) == 1 and "sp_dual" not in text
                payload = text.replace(needle, addition + needle).encode()
            elif info.filename == "inference/models/config.json":
                config = json.loads(payload)
                assert "sp_dual" not in config
                config["sp_dual"] = {"enabled": True, "min_conf": 0.35, "factor": 0.05}
                payload = (json.dumps(config, indent=2) + "\n").encode()
            dst.writestr(info, payload)
    with zipfile.ZipFile(TARGET) as result:
        assert result.testzip() is None
        assert json.loads(result.read("inference/models/config.json"))["sp_dual"]["enabled"]
        assert result.read("inference/koz3_infer/run.py").count(b"selective satellite dual") == 1
        print(f"{TARGET}: {TARGET.stat().st_size:,} bytes, {len(result.namelist())} entries, CRC PASS")


if __name__ == "__main__":
    main()
