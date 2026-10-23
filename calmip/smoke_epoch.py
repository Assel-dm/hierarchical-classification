"""Run one real CE epoch with the normal batch size in a separate results folder"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

cfg.EXPERIMENT_NAME += '_smoke_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
cfg.METHOD = 'mbm'
cfg.EXPERIMENTS = [('ce', None)]
cfg.SEEDS = [42]
cfg.STAGES = ['full']
cfg.STAGE_SETTINGS = {'full': {**cfg.STAGE_SETTINGS['full'], 'max_epochs': 1, 'patience': 1}}
cfg.DEVICE = 'cuda'
cfg.PRETRAINED = True
cfg.RUN_TRAINING = True
cfg.RESUME_TRAINING = False
cfg.EVALUATE_VALIDATION = True
cfg.EVALUATE_TEST = False
cfg.SAVE_PROBABILITIES = False
cfg.MAKE_PLOTS = True

from main import main

print('Isolated smoke campaign:', cfg.EXPERIMENT_NAME)
print('Full train/validation splits, one CE epoch; config.py is not modified.')
main()
