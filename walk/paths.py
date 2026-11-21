"""Where the external data live. Override any of them with an environment variable."""
import importlib.util
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

# ---- flybody's fruitfly.xml / floor.xml and meshes ----------------------------------------
# Looked up in this order; the first folder that contains fruitfly.xml wins:
#   1. $FLYBODY_ASSETS
#   2. an installed flybody package (pip install / editable clone), found without importing it
#   3. third_party/flybody (what scripts/get_flybody.sh creates)
THIRD_PARTY_ASSETS = ROOT / "third_party" / "flybody" / "flybody" / "fruitfly" / "assets"
FLYBODY_COMMIT = "d015e9bfe441bd90ae431bac24c55cb74bdbce26"   # results were produced with this commit


def _flybody_candidates():
    if os.environ.get("FLYBODY_ASSETS"):
        yield "$FLYBODY_ASSETS", Path(os.environ["FLYBODY_ASSETS"]).expanduser()
    try:
        spec = importlib.util.find_spec("flybody")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            yield "installed flybody package", Path(loc) / "fruitfly" / "assets"
    yield "third_party/flybody", THIRD_PARTY_ASSETS


def _find_flybody_assets():
    for _, p in _flybody_candidates():
        if (p / "fruitfly.xml").is_file():
            return p
    return None


# None if the assets are not found; use flybody_file() to get a file with a helpful error
FLYBODY_ASSETS = _find_flybody_assets()


def flybody_file(name):
    """Path of a file in flybody's fruitfly/assets folder; raises a clear error if it is missing."""
    if FLYBODY_ASSETS is not None and (FLYBODY_ASSETS / name).is_file():
        return FLYBODY_ASSETS / name
    tried = "\n".join(f"  - {why}: {p}" for why, p in _flybody_candidates())
    raise FileNotFoundError(
        f"flybody's fly model ({name}) was not found. Looked in:\n{tried}\n"
        "Fix it with ONE of:\n"
        "  bash scripts/get_flybody.sh                          # fetches the pinned assets into third_party/\n"
        "  export FLYBODY_ASSETS=/path/to/flybody/flybody/fruitfly/assets\n"
        "  pip install --no-deps git+https://github.com/TuragaLab/flybody.git")
