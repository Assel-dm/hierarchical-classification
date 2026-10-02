"""Build timm backbone and control their defreeze""" 
import copy
import torch
from torch import nn
import timm
from timm.data import create_transform

PRESETS = {
    "resnet50": dict(name="resnet50.tv2_in1k", label="ResNet-50", family="resnet", kwargs={},
                     optimizer="adam", weight_decay=0.0, partial=["layer4"], deeper=["layer3", "layer4"]),
    "convnext_tiny": dict(name="convnext_tiny.fb_in1k", label="ConvNeXt Tiny", family="convnext", kwargs={},
                         optimizer="adamw", weight_decay=0.01, partial=["stages.3", "head.norm"],
                         deeper=["stages.2", "stages.3", "head.norm"]),
}


def resolve_model_settings(key, image_size=None, head_dropout=0.4, freeze_batch_norm=True):
    """Resolve the size and normalisation of weights"""
    if timm.__version__ != "1.0.29":
        raise RuntimeError("This protocol requires timm==1.0.29")
    if key not in PRESETS:
        raise ValueError(f"Unknown backbone {key!r}; available: {list(PRESETS)}")
    if image_size is not None and (type(image_size) is not int or image_size <= 0):
        raise ValueError("IMAGE_SIZE must be None or a pisitive integer")
    if not 0 <= head_dropout < 1:
        raise ValueError("HEAD_DROPOUT must be within 0 and 1")
    settings = copy.deepcopy(PRESETS[key])
    cfg = timm.models.get_pretrained_cfg(settings["name"])
    if cfg is None:
        raise ValueError(f"Missing timm pretrained configuration: {settings['name']}")
    data = timm.data.resolve_data_config(pretrained_cfg=cfg.to_dict())
    size = [image_size] * 2 if image_size else list(data["input_size"][-2:])
    if "patch" in settings and any(s % settings["patch"] for s in size):
        raise ValueError(f"Image size {size} must be divisible by patch size {settings['patch']}")
    settings.update(key=key, image_size=size, mean=list(data["mean"]), std=list(data["std"]),
                    interpolation=data["interpolation"], crop_pct=float(data["crop_pct"]),
                    crop_mode=data.get("crop_mode", "center"), head_dropout=head_dropout,
                    freeze_bn=freeze_batch_norm)
    settings["kwargs"].update(drop_rate=0.0, drop_path_rate=0.0)
    if settings["family"] == "vit":
        settings["kwargs"]["img_size"] = size
    return settings


def build_transform(settings, training=False, augmentation="standard"):
    """Apply data augmentation"""
    options = dict(input_size=(3, *settings["image_size"]),
                   mean=tuple(settings["mean"]), std=tuple(settings["std"]),
                   interpolation=settings["interpolation"], use_prefetcher=False)
    if training:
        policies = {"standard": None, "autoaugment_original": "original"}
        if augmentation not in policies:
            raise ValueError(f"Unknown augment : {augmentation}")
        return create_transform(**options, is_training=True, auto_augment=policies[augmentation])
    return create_transform(**options, is_training=False,
                            crop_pct=settings["crop_pct"], crop_mode=settings["crop_mode"])


class SpeciesClassifier(nn.Module):
    def __init__(self, settings, num_leaves, pretrained):
        super().__init__()
        self.settings = settings
        self.backbone = timm.create_model(settings["name"], pretrained=pretrained, num_classes=0, **settings["kwargs"])
        self.head = nn.Sequential(nn.Dropout(settings["head_dropout"]), nn.Linear(self.backbone.num_features, num_leaves))
        self.open_modules = []
        self.set_stage("frozen")

    def forward(self, images):
        """Returns logits of species"""
        features = self.backbone(images)
        if features.ndim != 2 or features.shape[-1] != self.head[1].in_features:
            raise ValueError(f"Unexpected pooled features: {tuple(features.shape)}")
        return self.head(features)

    def set_stage(self, stage):
        """Unfreeze choosen modules """ 
        if stage not in ("frozen", "partial", "deeper"):
            raise ValueError(f"Unknown stage : {stage}")
        self.open_modules = [] if stage == "frozen" else list(self.settings[stage])
        for p in self.backbone.parameters():
            p.requires_grad = False
        for name in self.open_modules:
            for p in self.backbone.get_submodule(name).parameters():
                p.requires_grad = True
        if self.settings["freeze_bn"]:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    for p in module.parameters():
                        p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True
        self.train(self.training)

    def train(self, mode=True):
        """Keep frozen modules in evaluation""" 
        super().train(mode)
        self.backbone.eval()
        if mode:
            for name in self.open_modules:
                self.backbone.get_submodule(name).train(True)
        if self.settings["freeze_bn"]:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self


def make_optimizer(model, learning_rate):
    settings = model.settings
    if settings["optimizer"] == "adam":
        return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    exclusions = set()
    if hasattr(model.backbone, "no_weight_decay"):
        exclusions = {"backbone." + name for name in model.backbone.no_weight_decay()}
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = no_decay if parameter.ndim <= 1 or name.endswith(".bias") or name in exclusions else decay
            group.append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": settings["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=learning_rate)
