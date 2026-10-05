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
    def __init__(self, mappings, alpha, normalize_weights=False):
        super().__init__()
        self.level_buffers = []

        for index, matrix in enumerate(reversed(list(mappings.values()))):
            name = f"level_{index}"
            self.register_buffer(
                name,
                torch.as_tensor(matrix, dtype=torch.float32),
            )
            self.level_buffers.append(name)

        depths = torch.arange(len(mappings) + 1, 0, -1, dtype=torch.float32)
        weights = torch.exp(-float(alpha) * depths)

        if normalize_weights:
            total = 1 + mappings[next(iter(mappings))].shape[0] * math.exp(
                -float(alpha) * (len(mappings) + 1)
            )
            total += sum(
                matrix.shape[1] * math.exp(-float(alpha) * depth)
                for depth, matrix in enumerate(mappings.values(), start=1)
            )
            weights /= total

        self.register_buffer("weights", weights)

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=1)
        log_nodes = [log_p.gather(1, targets[:, None]).squeeze(1)]

        for name in self.level_buffers:
            mapping = getattr(self, name)
            groups = mapping[targets].argmax(dim=1)
            members = mapping[:, groups].T.bool()
            log_nodes.append(
                torch.logsumexp(log_p.masked_fill(~members, -torch.inf), dim=1)
            )

        log_nodes.append(torch.zeros_like(log_nodes[0]))
        terms = torch.stack(
            [child - parent for child, parent in zip(log_nodes, log_nodes[1:])],
            dim=1,
        )
        return -(terms * self.weights).sum(dim=1).mean()


def build_loss(variant, parameter, data, normalize_hxe=FALSE):
    """Choose CE, HXE or soft labels"""
    if variant == "ce":
        if parameter is not None:
            raise ValueError("CE does not use parameters")
        return nn.CrossEntropyLoss()
    if variant not in {"soft", "hxe"}:
        raise ValueError(f"Unknown method : {variant}")
    if isinstance(parameter, bool) or not isinstance(parameter, (int, float)) or not math.isfinite(parameter) or parameter < 0:
        raise ValueError("Alpha/beta must be a positif or null real number")
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
        raise ValueError("Targeted mass outside the interval")
    for _ in range(iterations):
        middle = (lower + upper) / 2
        if soft_label_statistics(normalized_distances, middle)["true_class_mass"] < target_mass:
            lower = middle
        else:
            upper = middle
    return round((lower + upper) / 2, 3)
