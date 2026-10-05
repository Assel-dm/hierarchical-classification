from pathlib import Path

# File on the remote machine
DATASET = "inaturalist19_h" #small_collemboles or inaturalist19_h
DATASET_ROOT = Path.home() / "datasets" / "inaturalist-19-h
OUTPUT_ROOT = Path.home() / "results"  # to adapt
INAT_HIERARCHY_FILE = DATASET_ROOT / "metadata" / "inaturalist19_isa.txt" #Used hierarchy
INAT_CATEGORIES_FILE = Path.home() / "datasets" / "inaturalist-19" / "categories.json"
EXPERIMENT_NAME = "inat19h_convnext_v1"


# One backbone and one augmentation by campaign
MODEL_KEY = "convnext_tiny"  # resnet50, convnext_tiny, dinov2_base, dinov3_base
PRETRAINED = True
IMAGE_SIZE = None          # None = standard timm weights
HEAD_DROPOUT = 0.4
FREEZE_BATCH_NORM = True
AUGMENTATION = "autoaugment_original"  # standard or autoaugment_original


METHOD = "mbm"
HXE_NORMALIZE_WEIGHTS = True
EXPERIMENTS = [
    ("ce", None),          # Baseline
    ("soft", 5),           # Soft Labels : beta
    ("hxe", 0.4),          # HXE : alpha
]
SEEDS = [42]               # multiseed : [42, 43, 44, 45, 46]
STAGES = ["frozen"]        # ["frozen", "partial", "deeper"]

# Training parameters
DEVICE = "auto"            
BATCH_SIZE = 32
NUM_WORKERS = 2            
STAGE_SETTINGS = {
    "frozen":  {"learning_rate": 3e-4, "max_epochs": 30, "patience": 5},
    "partial": {"learning_rate": 1e-5, "max_epochs": 30, "patience": 5},
    "deeper":  {"learning_rate": 1e-5, "max_epochs": 30, "patience": 5},
}

# Switchs
RUN_TRAINING = True
RESUME_TRAINING = True
EVALUATE_VALIDATION = True
EVALUATE_TEST = False       
SAVE_PROBABILITIES = False 
MAKE_PLOTS = True
PREVIEW_AUGMENTATIONS = True
