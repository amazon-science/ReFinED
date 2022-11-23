from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor


class EntityDisambiguation(nn.Module):
    def __init__(
        self,
        dropout: float,
        num_classes: int,
        ignore_descriptions: bool = False,
        ignore_types: bool = False,
    ):
        super().__init__()
        self.ignore_descriptions = ignore_descriptions
        self.ignore_types = ignore_types
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_classes + 3, 1)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for all member variables with type nn.Module"""
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            module.bias.data.zero_()
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        class_activations=None,
        candidate_entity_targets=None,
        candidate_pem_values=None,
        candidate_classes=None,
        candidate_description_scores=None,
        current_device: str = None,
    ) -> Tuple[Optional[Tensor], Tensor]:
        """
        Forward pass of ED layer.
        Scores candidate entities.
        :param class_activations: predicted classes for each mention span. shape = (all_ents, num_classes)
        :param candidate_entity_targets: gold entity for each mention span. shape = ?
        :param candidate_pem_values: P(e|m) value for each mention span for each candidate. shape = (all_ents, num_cands)
        :param candidate_classes: classes for each mention span for each candidate. shape = (all_ents, num_cands, num_classes)
        :param candidate_description_scores: candidate_description_scores. shape = (num_ents, num_cands + 1)
        :param current_device: the device (e.g. cuda:0, or cpu) to use (required for multi-gpu to avoid deadlock)
        :return: loss tensor (if candidate_targets is provided), candidate entity (ED) activations tensor
        """
        predicted_classes = class_activations
        predicted_classes_expanded = predicted_classes.unsqueeze(1)
        candidate_diff_classes = candidate_classes - predicted_classes_expanded
        candidate_diff_classes_squared = candidate_diff_classes ** 2
        candidate_dist = torch.sqrt(candidate_diff_classes_squared.sum(dim=2)).unsqueeze(-1)

        candidate_delta_classes = candidate_classes * predicted_classes_expanded

        # disable descriptions see if makes model more general
        # if it does then add significant dropout here instead of all zero
        if self.ignore_descriptions:
            candidate_description_scores = torch.zeros_like(candidate_description_scores)

        if self.ignore_types:
            candidate_delta_classes = torch.zeros_like(candidate_delta_classes)
            candidate_dist = torch.zeros_like(candidate_dist)

        class_and_pem = torch.cat(
            [
                candidate_delta_classes,
                candidate_pem_values.unsqueeze(-1),
                candidate_dist,
                candidate_description_scores[:, :-1].unsqueeze(-1),
            ],  # removes "no_cand" score
            2,
        ).to(current_device)

        class_and_pem = self.dropout(class_and_pem)
        logits = self.classifier(class_and_pem)

        logits_with_none_above = torch.cat(
            (logits, torch.zeros(size=(logits.size(0), 1, 1), device=current_device)), 1
        ).squeeze(-1)

        # candidate_mask is 0 when valid entity, and -10,000 when padding element
        # candidate entities with a pem value of 0 are considered padding elements
        # note that this will not mask out the NOTA entity because NOTA have values set to 1.0 (see torch.ones() below)
        # TODO: use candidate_qcodes tensor instead here
        candidate_mask = (
            torch.cat(
                (
                    candidate_pem_values,
                    torch.ones(size=(candidate_pem_values.size(0), 1), device=current_device),
                ),
                dim=1,
            )
            == 0
        ).int() * -1e8  # assumes pem_value == 0 means padding element

        # candidate_mask_zero 1 when valid element 0 when padding element
        candidate_mask_zero = (candidate_mask == 0).int()

        # consider that gold entity is masked (multiply by 0 prevents this)
        logits_with_none_above = logits_with_none_above * candidate_mask_zero + candidate_mask
        # TODO: go back to above
        # just changed
        # logits_with_none_above = logits_with_none_above + candidate_mask

        if candidate_entity_targets is not None:
            # assumes argmax will select last index if no gold entity
            loss = F.cross_entropy(logits_with_none_above, candidate_entity_targets.argmax(dim=1))
            return loss, logits_with_none_above

        return None, logits_with_none_above
