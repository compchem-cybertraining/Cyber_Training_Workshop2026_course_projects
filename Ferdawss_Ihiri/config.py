"""Central settings for the PAH101 cheap-to-expensive ML surrogate project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# PAH101 dataset (edit if your copy lives elsewhere)
PAH101_ROOT = Path("PAH101_data")
PAH101_CSV = PAH101_ROOT / "PAH101_calculation_info_04162024.csv"
PAH101_CIF_DIR = PAH101_ROOT / "cifsH"

GEOM_DIR = PROJECT_ROOT / "geometries"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

for _d in (GEOM_DIR, RESULTS_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MAX_MONOMER_ATOMS = 55

CHEAP_BASIS = "sto-3g"
CHEAP_METHOD = "HF"
CHEAP_NSTATES = 1
CHEAP_MAX_MEMORY_MB = 4000

EV_TO_CM1 = 8065.54429

TARGET_COLUMNS = [
    "GW_QP_BandGap",
    "GWBSE_EsC",
    "Singlet_binding",
]
DEFAULT_TARGET = "Singlet_binding"

CHEAP_TABULATED_FEATURES = [
    "GapC",
    "AtomNumC",
    "MolWtS",
    "RhoC",
    "EpsilonC",
]

RANDOM_SEED = 42
N_CV_FOLDS = 5
