from pathlib import Path

# File on the remote machine
DATASET = "inaturalist19_h" #small_collemboles or inaturalist19_h
DATASET_ROOT = Path.home() / "datasets" / "inaturalist-19-h"
OUTPUT_ROOT = Path.home() / "results"  # to adapt
INAT_HIERARCHY_FILE = DATASET_ROOT / "metadata" / "inaturalist19_isa.txt" #Used hierarchy
INAT_CATEGORIES_FILE = Path.home() / "datasets" / "inaturalist-19" / "categories.json"
EXPERIMENT_NAME = "inat19h_convnext_hag_v1"


# One backbone and one augmentation by campaign
MODEL_KEY = "convnext_tiny"  # resnet50, convnext_tiny, dinov2_base, dinov3_base
PRETRAINED = True
IMAGE_SIZE = None          # None = standard timm weights
HEAD_DROPOUT = 0.4
FREEZE_BATCH_NORM = False
AUGMENTATION = "autoaugment_original"  # standard or autoaugment_original


METHOD = "mbm"
HXE_NORMALIZE_WEIGHTS = True
EXPERIMENTS_MBM = [
    ("ce", None),          # Baseline
    ("soft", 5),           # Soft Labels : beta
    ("soft", 10),
    ("soft", 15),
    ("soft", 20),
    ("soft", 30),
    ("hxe", 0.1),          # HXE : alpha
    ("hxe", 0.2),  
    ("hxe", 0.4),  
    ("hxe", 0.5),  
    ("hxe", 0.7),  
]

EXPERIMENTS_HAF = [
    ("ce", None),          # Baseline 
    ("haf", None),         # CE fine + SHC + margin + GC
    # ("haf_ce", None),    # CE only
    # ("haf_shc", None),   # CE + SHC
    # ("haf_margin", None),# CE + margin
    # ("haf_gc", None),    # CE + GC
    # ("haf_shc_gc", None),# CE + SHC + GC
    # ("haf_shc_margin", None),  # CE + SHC + margin
]

EXPERIMENTS = EXPERIMENTS_HAF

HAF_MARGIN = 3.0           # Authors' value; above ln(2), so the hinge stays active
HAF_MARGIN_MODE = "official"  # official: two KL hinges; paper: one JSD hinge
HAF_MAX_PAIRS = 256        # Maximum valid pairs per batch and coarse rank
HAF_MARGIN_RANKS = None    # iNat: phylum to genus; Small-Collemboles: all coarse ranks
HAF_MONITOR = "val_loss"     # Early stopping: val_ce or val_loss

SEEDS = [42]               # multiseed : [42, 43, 44, 45, 46]
STAGES = ["full"]        # ["frozen", "partial", "deeper"]

# Training parameters
DEVICE = "auto"            
BATCH_SIZE = 32
NUM_WORKERS = 2            
STAGE_SETTINGS = {
    "full": {"learning_rate": 1e-5, "max_epochs": 40, "patience": 8},
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
