"""HAF heads and four losses (Garg et al., ECCV 2022)."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from common.models import SpeciesClassifier


# CE is always active; the tuple enables SHC, margin and GC.
VARIANTS = {
    "haf_ce": (False, False, False),
    "haf_shc": (True, False, False),
    "haf_shc_margin": (True, True, False),
    "haf_shc_gc": (True, False, True),
    "haf": (True, True, True),
}


class HAFClassifier(SpeciesClassifier):
    """Use one normalized linear head per rank on the same timm features"""
    def __init__(self, settings, data, pretrained):
        super().__init__(settings, len(data["species"]), pretrained)
        self.ranks = list(data["ranks"])
        self.coarse_heads = nn.ModuleList([
            nn.Linear(self.backbone.num_features, data["mappings"][rank].shape[1])
            for rank in self.ranks[:-1]
        ])

    def forward(self, images, return_hierarchy=False):
        """Return species logits, or all heads and weights for the HAF loss"""
        features = self.forward_features(images)
        features = self.head[0](features)
        fine_weight = F.normalize(self.head[1].weight, dim=1)
        fine_logits = F.linear(features, fine_weight, self.head[1].bias)
        if not return_hierarchy:
            return fine_logits
        weights = [F.normalize(head.weight, dim=1) for head in self.coarse_heads]
        logits = [F.linear(features, weight, head.bias) for head, weight in zip(self.coarse_heads, weights)]
        return {"logits": fine_logits, "level_logits": logits + [fine_logits],
                "level_weights": weights + [fine_weight]}


def kl_divergence(p, q):
    """Return KL(p || m) and KL(q || m)"""
    # Keep logarithms finite even for very confident predictions.
    tiny = torch.finfo(p.dtype).tiny
    m = (0.5 * (p + q)).clamp_min(tiny).log()
    kl_p = (p * (p.clamp_min(tiny).log() - m)).sum(dim=-1)
    kl_q = (q * (q.clamp_min(tiny).log() - m)).sum(dim=-1)
    return kl_p, kl_q


def jensen_shannon(p, q):
    """Compute JSD"""
    kl_p, kl_q = kl_divergence(p, q)
    return (0.5 * (kl_p + kl_q)).clamp_min(0)


def margin_loss(p, q, margin, mode="official"):
    """Compare different-class pairs"""
    kl_p, kl_q = kl_divergence(p, q)
    if mode == "official":
        return (0.5 * (F.relu(margin - kl_p) + F.relu(margin - kl_q))).mean()
    if mode == "paper":
        jsd = jensen_shannon(p,q)
        return F.relu(margin - jsd).mean()
    raise ValueError("HAF_MARGIN_MODE must be official or paper")


def dissimilar_pairs(labels, max_pairs, training):
    """Select valid unordered pairs"""
    left, right = torch.triu_indices(len(labels), len(labels), offset=1, device=labels.device)
    different = labels[left] != labels[right]
    pairs = torch.stack((left[different], right[different]), dim=1)
    if len(pairs) > max_pairs:
        if training:
            selected = torch.randperm(len(pairs), device=labels.device)[:max_pairs]
        else:
            # Stable validation pairs, spread across the available batch
            selected = torch.linspace(0, len(pairs) - 1, max_pairs, device=labels.device).long()
        pairs = pairs[selected]
    return pairs
