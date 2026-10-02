"""The three MBM losses"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_soft_label_matrix(normalized_distances, beta):
    """Build softmax targets"""
    logits = -float(beta) * normalized_distances
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return (values / values.sum(axis=1, keepdims=True)).astype(np.float32)


class HierarchicalSoftLabelLoss(nn.Module):
    """Compute cross-entropy with the targets"""
    def __init__(self, soft_label_matrix):
        super().__init__()
        self.register_buffer("soft_label_matrix", torch.as_tensor(soft_label_matrix, dtype=torch.float32))

    def forward(self, logits, targets):
        soft_targets = self.soft_label_matrix[targets]
        return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


class HierarchicalCrossEntropy(nn.Module):
    """Compute HXE"""
    def __init__(self, mappings, alpha):
        super().__init__()
        for rank, matrix in mappings.items():
            self.register_buffer(rank, torch.as_tensor(matrix, dtype=torch.float32))
        depths = torch.tensor([4., 3., 2., 1.])
        self.register_buffer("weights", torch.exp(-float(alpha) * depths))

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=1)
        log_species = log_p.gather(1, targets[:, None]).squeeze(1)
        ancestor_logs = []
        for mapping in (self.genus, self.family, self.order):
            groups = mapping[targets].argmax(dim=1)
            members = mapping[:, groups].T.bool()
            ancestor_logs.append(torch.logsumexp(log_p.masked_fill(~members, -torch.inf), dim=1))
        log_genus, log_family, log_order = ancestor_logs
        terms = torch.stack([log_species - log_genus, log_genus - log_family,
                             log_family - log_order, log_order], dim=1)
        return -(terms * self.weights).sum(dim=1).mean()


def build_loss(variant, parameter, data):
    """Choose CE, HXE or soft labels"""
    if variant == "ce":
        if parameter is not None:
            raise ValueError("CE ne prend pas de paramètre")
        return nn.CrossEntropyLoss()
    if variant not in {"soft", "hxe"}:
        raise ValueError(f"Variante MBM inconnue : {variant}")
    if isinstance(parameter, bool) or not isinstance(parameter, (int, float)) or not math.isfinite(parameter) or parameter < 0:
        raise ValueError("Alpha/beta doit être un nombre fini positif ou nul")
    if variant == "soft":
        return HierarchicalSoftLabelLoss(build_soft_label_matrix(data["normalized_distances"], parameter))
    return HierarchicalCrossEntropy(data["mappings"], parameter)


def soft_label_statistics(normalized_distances, beta):
    """Prompt mean mass of the true target and the mean entropy"""
    matrix = build_soft_label_matrix(normalized_distances, beta)
    return {"beta": float(beta), "true_class_mass": float(np.diag(matrix).mean()),
            "mean_entropy": float((-(matrix * np.log(matrix + 1e-12)).sum(axis=1)).mean())}


def beta_for_target_mass(normalized_distances, target_mass, iterations=60):
    """estimate beta for a target mass on the true target""" 
    lower, upper = 0., 200.
    if not 1 / len(normalized_distances) <= target_mass <= 1:
        raise ValueError("Masse cible hors de l'intervalle possible")
    for _ in range(iterations):
        middle = (lower + upper) / 2
        if soft_label_statistics(normalized_distances, middle)["true_class_mass"] < target_mass:
            lower = middle
        else:
            upper = middle
    return round((lower + upper) / 2, 3)
