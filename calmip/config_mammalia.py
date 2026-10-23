import os
from pathlib import Path

DATASET = "inat21"  # small_collemboles, inaturalist19_h or inat21
DATASET_ROOT = Path(os.environ["WORK"]) / "datasets" / "inat21_subset_mini" / "Mammalia"
OUTPUT_ROOT = Path(os.environ["WORK"]) / "results"
INAT_HIERARCHY_FILE = DATASET_ROOT / "metadata" / "inaturalist19_isa.txt" 
INAT_CATEGORIES_FILE = Path.home() / "datasets" / "inaturalist-19" / "categories.json"
EXPERIMENT_NAME = "inat21_mammalia_mini_convnext_mbm_calmip_v1"

# Insecta/Mammalia and mini/full are inferred from the subset metadata.
# Within a subset, kingdom/phylum/class are constant. Use the informative ranks.
INAT21_RANKS = ["order", "family", "genus", "species"]
INAT21_CHECK_FILES = True


MODEL_KEY = "convnext_tiny"  
PRETRAINED = True
IMAGE_SIZE = None          # None = standard timm weights
HEAD_DROPOUT = 0.4
FREEZE_BATCH_NORM = False
AUGMENTATION = "autoaugment_original"  # standard or autoaugment_original


METHOD = "mbm"
HXE_NORMALIZE_WEIGHTS = False
EXPERIMENTS_MBM = [
    ("ce", None),
    ("soft", 15),
    ("soft", 20),
    ("soft", 30),
    ("hxe", 0.2),
    ("hxe", 0.4),
    ("hxe", 0.5),
]

EXPERIMENTS_HAF = [
    ("ce", None),          # Baseline 
    ("haf", None),         # CE fine + SHC + margin + GC
    ("haf_ce", None),    # CE only
    ("haf_shc", None),   # CE + SHC
    ("haf_shc_gc", None), # CE + SHC + GC
    ("haf_shc_margin", None),  # CE + SHC + margin
]

EXPERIMENTS = EXPERIMENTS_MBM

HAF_MARGIN = 3.0           # Authors' value; above ln(2), so the hinge stays active
HAF_MARGIN_MODE = "official"  # official: two KL hinges; paper: one JSD hinge
HAF_MAX_PAIRS = 256        # Maximum valid pairs per batch and coarse rank
HAF_MARGIN_RANKS = None    # iNat: phylum to genus; Small-Collemboles: all coarse ranks
HAF_MONITOR = "val_loss"     # Early stopping: val_ce or val_loss

SEEDS = [42]               # multiseed : [42, 43, 44, 45, 46]
STAGES = ["full"]        # ["frozen", "partial", "deeper"]

# Training parameters
DEVICE = "cuda"            
BATCH_SIZE = 32
NUM_WORKERS = 4            
STAGE_SETTINGS = {
    "full": {"learning_rate": 1e-5, "max_epochs": 80, "patience": 8},
}
#    "frozen":  {"learning_rate": 3e-4, "max_epochs": 50, "patience": 8},
#    "partial": {"learning_rate": 1e-5, "max_epochs": 30, "patience": 6},
#    "deeper":  {"learning_rate": 1e-5, "max_epochs": 30, "patience": 6},

# Switchs
RUN_TRAINING = True
RESUME_TRAINING = True
EVALUATE_VALIDATION = True
EVALUATE_TEST = False       
SAVE_PROBABILITIES = False 
MAKE_PLOTS = True
PREVIEW_AUGMENTATIONS = True
