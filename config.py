from pathlib import Path

# File on the remote machine
DATASET = "small_collemboles"
DATASET_ROOT = Path("/dataset/small-collomboles")
OUTPUT_ROOT = Path.home() / "results"  # to adapt
EXPERIMENT_NAME = "small_collemboles_convnext_v1"

# One backbone and one augmentation by campaign
MODEL_KEY = "convnext_tiny"  # resnet50, convnext_tiny, dinov2_base, dinov3_base
PRETRAINED = True
IMAGE_SIZE = None          # None = standard timm weights
HEAD_DROPOUT = 0.4
FREEZE_BATCH_NORM = True
AUGMENTATION = "autoaugment_original"  # standard or autoaugment_original


METHOD = "mbm"
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
