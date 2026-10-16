"""HAF heads and four losses (Garg et al., ECCV 2022)."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from common.models import SpeciesClassifier


# CE is always active; the tuple enables SHC, margin and GC
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
    # Keep logarithms finite even for very confident predictions
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
            selected = torch.linspace(0, len(pairs) - 1, max_pairs, device=labels.device).long()
        pairs = pairs[selected]
    return pairs


class HAFLoss(nn.Module):
    """Add fine CE, soft consistency, margin and geometric consistency"""
    requires_hierarchy = True #Allow head at every levels
    def __init__(self, data, variant="haf", margin=3.0, max_pairs=256,
                 margin_ranks=None, margin_mode="official"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown HAF variant: {variant}; available: {list(VARIANTS)}")
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
            raise ValueError("HAF_MARGIN must be a finite non-negative number")
        if type(max_pairs) is not int or max_pairs < 1:
            raise ValueError("HAF_MAX_PAIRS must be a positive integer")
        if margin_mode not in {"official", "paper"}:
            raise ValueError("HAF_MARGIN_MODE must be official or paper")
        self.ranks = list(data["ranks"])
        if len(self.ranks) < 2 or self.ranks[-1] != "species":
            raise ValueError("HAF requires coarse ranks followed by species")
        if margin_ranks is None:
            # The official iNat model excludes kingdom from the margin loss
            margin_ranks = [rank for rank in self.ranks[:-1] if not (data.get("name") == "inaturalist19_h" and rank == "kingdom")]
        if not isinstance(margin_ranks, list) or len(set(margin_ranks)) != len(margin_ranks):
            raise ValueError("HAF_MARGIN_RANKS must be None or a list of unique coarse ranks")
        if any(rank not in self.ranks[:-1] for rank in margin_ranks):
            raise ValueError(f"HAF_MARGIN_RANKS must be chosen from {self.ranks[:-1]}")
        self.use_shc, self.use_margin, self.use_gc = VARIANTS[variant]
        self.margin, self.max_pairs, self.margin_mode = float(margin), max_pairs, margin_mode
        self.margin_ranks = list(margin_ranks)
        self.last_components = {}

        leaf_ids, sizes = [], []
        for rank in self.ranks[:-1]:
            mapping = torch.as_tensor(data["mappings"][rank], dtype=torch.float32)
            leaf_ids.append(mapping.argmax(dim=1))
            sizes.append(mapping.shape[1])
        leaf_ids.append(torch.arange(len(data["species"])))
        sizes.append(len(data["species"]))
        self.level_sizes = sizes
        for level in range(len(self.ranks) - 1):
            parent_map = torch.zeros(sizes[level + 1], sizes[level])
            parent_map[leaf_ids[level + 1], leaf_ids[level]] = 1
            if not torch.all(parent_map.sum(dim=1) == 1):
                raise ValueError(f"Every {self.ranks[level + 1]} needs exactly one parent")
            self.register_buffer(f"parent_map_{level}", parent_map)
            self.register_buffer(f"leaf_labels_{level}", leaf_ids[level])

    def forward(self, outputs, targets):
        logits, weights = outputs["level_logits"], outputs["level_weights"]
        if len(logits) != len(self.ranks) or len(weights) != len(self.ranks):
            raise ValueError("HAF outputs do not match the taxonomy levels")
        if any(values.shape != (len(targets), size) for values, size in zip(logits, self.level_sizes)):
            raise ValueError("HAF logits do not match the taxonomy class counts")
        ce = F.cross_entropy(logits[-1], targets) #fine CE
        shc, margin, gc = (ce.new_zeros(()) for _ in range(3))
        diagnostics = {}
        active_ranks = pair_count = hinge_count = 0
        active_hinges = ce.new_zeros(())
        probabilities = [values.softmax(dim=1) for values in logits]
        for level, rank in enumerate(self.ranks[:-1]):
            parent_map = getattr(self, f"parent_map_{level}")
            if self.use_shc: #SHC
                soft_targets = probabilities[level + 1] @ parent_map
                shc = shc + jensen_shannon(probabilities[level], soft_targets).mean()
            if self.use_margin and rank in self.margin_ranks: #Margin
                labels = getattr(self, f"leaf_labels_{level}")[targets]
                pairs = dissimilar_pairs(labels, self.max_pairs, self.training)
                n_pairs = len(pairs)
                diagnostics[f"margin_pairs_{rank}"] = (n_pairs, 1)
                diagnostics{f"margin_batch_fraction_{rank}"] = (int(n_pairs > 0), 1)
                rank_active = ce.new_zeros(())
                rank_hinges = 0
                if len(pairs):
                    p, q = probabilities[level][pairs[:, 0]], probabilities[level][pairs[:, 1]]
                    margin = margin + margin_loss(p, q, self.margin, self.margin_mode)
                    # Pairs inspection
                    with torch.no_grad():
                        if self.margin_mode == "official":
                            kl_p, kl_q = kl_divergence(p.detach(), q.detach())
                            rank_active = (kl_p < self.margin).sum() + (kl_q < self.margin).sum()
                            rank_hinges = 2 * n_pairs
                        else :
                            jsd = jensen_shannon(p.detach(), q.detach())
                            rank_active = (jsd < self.margin).sum()
                            rank_hinges = n_pairs
                    active_ranks += 1
                    pair_count += n_pairs
                    active_hinges = active_hinges + rank_active
                    hinge_count += rank_hinges
                diagnostics[f"margin_active_hinge_fraction_{rank}"] = (rank_active, rank_hinges)
            if self.use_gc: #Geometric consistency
                children = F.normalize(weights[level + 1], dim=1)
                parent_target = F.normalize(parent_map.T @ children, dim=1)
                similarity = F.cosine_similarity(weights[level], parent_target, dim=1)
                gc = gc + (1 - similarity).clamp_min(0).mean()
        self.last_components = {"ce": ce.detach(), "shc": shc.detach(),
                                "margin": margin.detach(), "gc": gc.detach()}
        if self.use_margin:
            diagnostics.update({
                "margin_active_ranks": (active_ranks, 1),
                "margin_pairs": (pair_count, 1),
                "margin_active_hinge_fraction": (active_hinges, hinge_count),
                "margin_lower_bound": (len(targets) * active_ranks * max(0., self.margin - math.log(2)), len(targets)),
            })
        self.last_diagnostics = diagnostics
        return ce + shc + margin + gc


def build_loss(variant, parameter, data, **options):
    """Choose a HAF ablation; its parameters are configured explicitly."""
    if parameter is not None:
        raise ValueError("HAF experiments use None; set HAF_MARGIN in config.py")
    return HAFLoss(data, variant=variant, **options)

