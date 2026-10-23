"""Verify the exact pretrained-model cache used by the campaign"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from common.models import SpeciesClassifier, resolve_model_settings

settings = resolve_model_settings(cfg.MODEL_KEY, cfg.IMAGE_SIZE,
                                  cfg.HEAD_DROPOUT, cfg.FREEZE_BATCH_NORM)
model = SpeciesClassifier(settings, 246, pretrained=True)
print(f"Pretrained weights loaded successfully: {settings['name']}")
