"""Where the external data live. Override any of them with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(name, default):
    return Path(os.environ[name]) if os.environ.get(name) else Path(default)


# MaleCNS v1.0 flat connectome files (see data/README.md)
MALECNS_DIR = _env("MALECNS_DIR", ROOT / "data" / "malecns")
# neurons.feather / connectivity.feather made by prepare_connectome.py
DATA_DIR = _env("FLY_DATA_DIR", ROOT / "data" / "processed")
# flybody walking motion capture (small file of datasets_walking-imitation.zip)
MOCAP_FILE = _env("FLYBODY_WALKING_DATA", ROOT / "data" / "flybody" /
                  "walking-dataset-small_female-only_snippets-100_min-len-0.5s_trk-files-0-9.hdf5")


def _flybody_assets():
    if os.environ.get("FLYBODY_ASSETS"):
        return Path(os.environ["FLYBODY_ASSETS"])
    try:
        import flybody                                      # pip install flybody / an editable clone
        return Path(flybody.__file__).resolve().parent / "fruitfly" / "assets"
    except ImportError:
        return ROOT / "third_party" / "flybody" / "flybody" / "fruitfly" / "assets"


# flybody's fruitfly.xml / floor.xml and meshes
FLYBODY_ASSETS = _flybody_assets()
