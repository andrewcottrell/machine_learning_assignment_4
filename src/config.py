from pathlib import Path

SRC = Path(__file__).resolve().parent
DATA = SRC / "data"
OUT = SRC / "results"
FIGS = SRC / "figs"
BASELINE = OUT / "baseline"

RAW_PATH = DATA / "networkTraffic.csv"
CAT_MAP_PATH = DATA / "attack_category_map.csv"
FOLDS_PATH = OUT / "folds.npy"

ID_COL = "id"
TARGET_COL = "attack_cat"
FLAG_COLS = ["is_sm_ips_ports", "is_ftp_login"]
NOMINAL_COLS = ["proto", "service", "state"]

RANDOM_STATE = 67
N_SPLITS = 5

SEEDS = [101, 211, 307, 401, 503]

N_CLASSES = 10
MAX_BINS = None
