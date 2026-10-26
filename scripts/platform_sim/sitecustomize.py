"""Platform simulation for rehearsals: packages the organisers' image (Ultralytics base + baseline requirements) is not
known to contain, and that we do not vendor, behave as NOT INSTALLED — availability probes (importlib.util.find_spec,
importlib.metadata) say absent, and a real import raises ModuleNotFoundError. A hidden dependency fails here, not on
the platform. Enabled with PYTHONPATH=scripts/platform_sim (scripts/rehearse.py: REHEARSE_BLOCK=1)."""
import importlib.abc
import importlib.metadata as _md
import importlib.util as _iu
import sys

# laspy is NOT in the image (the baseline parses LAS itself "so as not to pull laspy into the image"); we vendor it,
# so it may only come from an inference/vendor folder.
VENDOR_ONLY = {"laspy"}
BLOCKED = {"geopandas", "segmentation_models_pytorch", "timm", "lightgbm", "skimage", "sklearn", "matplotlib",
           "pyogrio", "fiona", "fitz", "seaborn", "xgboost", "catboost", "numba", "scikit-learn", "scikit-image"}
_blocked = lambda name: str(name).split(".")[0].replace("-", "_").lower() in {b.replace("-", "_") for b in BLOCKED}


def _vendor_dirs():
    return [p for p in sys.path if p.replace("\\", "/").rstrip("/").endswith("inference/vendor")]


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if _blocked(name):
            raise ModuleNotFoundError(f"[platform-sim] No module named '{name}'", name=name)
        top = name.split(".")[0]
        if top in VENDOR_ONLY and name == top:
            from importlib.machinery import PathFinder
            spec = PathFinder.find_spec(name, _vendor_dirs())
            if spec is None:
                raise ModuleNotFoundError(f"[platform-sim] No module named '{name}' (not in the image, not vendored)", name=name)
            return spec
        return None


_orig_find_spec = _iu.find_spec


def _find_spec(name, package=None):
    if _blocked(name):
        return None
    return _orig_find_spec(name, package)


_orig_version, _orig_distribution = _md.version, _md.distribution


def _version(name):
    if _blocked(name):
        raise _md.PackageNotFoundError(name)
    return _orig_version(name)


def _distribution(name):
    if _blocked(name):
        raise _md.PackageNotFoundError(name)
    return _orig_distribution(name)


_iu.find_spec = _find_spec
_md.version, _md.distribution = _version, _distribution
sys.meta_path.insert(0, _Block())
