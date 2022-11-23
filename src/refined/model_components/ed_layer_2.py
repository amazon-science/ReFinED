from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from refined.doc_preprocessing.preprocessor import Preprocessor
from refined.model_components.description_encoder import DescriptionEncoder


class EDLayer(nn.Module):
    def __init__(
        self,
        preprocessor: Preprocessor,
        mention_dim=768,
        output_dim=300,
        hidden_dim=1000,
        dropout=0.1,
        add_hidden=False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(mention_dim, hidden_dim)
        if add_hidden:
            self.mention_projection = nn.Linear(hidden_dim, output_dim)
        else:
            self.mention_projection = nn.Linear(mention_dim, output_dim)
        self.init_weights()
        self.description_encoder = DescriptionEncoder(
            output_dim=output_dim, ff_chunk_size=4, preprocessor=preprocessor
        )
        self.add_hidden = add_hidden

    def get_parameters_to_scale(self) -> List[nn.Parameter]:
        """
        Gets parameters that can be scaled during training. Usually top/last layers.
        :return: list of parameters
        """
        mention_projection_params = list(self.mention_projection.parameters())
        hidden_layer_projection_params = list(self.hidden_layer.parameters())
        description_encoder_projection_params = list(
            self.description_encoder.projection.parameters()
        )
        description_encoder_hidden_params = list(self.description_encoder.hidden_layer.parameters())
        description_encoder_params = list(self.description_encoder.transformer.parameters())
        return (
            mention_projection_params
            + description_encoder_projection_params
            + description_encoder_params
            + hidden_layer_projection_params
            + description_encoder_hidden_params
        )

    def get_parameters_not_to_scale(self) -> List[nn.Parameter]:
        """
        Gets parameters that can be scaled during training. Usually top/last layers.
        :return: list of parameters
        """
        return []

    def forward(
        self,
        mention_embeddings: Tensor,
        candidate_desc: Tensor,
        candidate_entity_targets: Optional[Tensor] = None,
        candidate_desc_emb: Optional[Tensor] = None,
    ):
        if self.add_hidden:
            mention_embeddings = self.mention_projection(
                self.dropout(F.relu(self.hidden_layer(mention_embeddings)))
            )
        else:
            mention_embeddings = self.mention_projection(mention_embeddings)

        if candidate_desc_emb is None:
            candidate_entity_embeddings = self.description_encoder(
                candidate_desc
            )  # (num_ents, num_cands, output_dim)
        else:
            candidate_entity_embeddings = candidate_desc_emb  # (num_ents, num_cands, output_dim)

        scores = (candidate_entity_embeddings @ mention_embeddings.unsqueeze(-1)).squeeze(
            -1
        )  # dot product
        # scores.shape = (num_ents, num_cands)

        # mask out pad candidates (and candidates without descriptions) before calculating loss
        # multiplying by 0 kills the gradients, loss should also ignore it
        mask_value = -100  # very large number may have been overflowing cause -inf loss
        if candidate_desc_emb is not None:
            # assumes candidate_desc_emb is initialised to all 0s
            multiplication_mask = (candidate_desc_emb[:, :, 0] != 0)
            addition_mask = (candidate_desc_emb[:, :, 0] == 0) * mask_value
        else:
            multiplication_mask = (
                candidate_desc[:, :, 0] != self.description_encoder.tokenizer.pad_token_id
            )
            addition_mask = (
                candidate_desc[:, :, 0] == self.description_encoder.tokenizer.pad_token_id
            ) * mask_value
        scores = (scores * multiplication_mask) + addition_mask

        no_cand_score = torch.zeros((scores.size(0), 1), device=scores.device)
        scores = torch.cat([scores, no_cand_score], dim=-1)

        if candidate_entity_targets is not None:
            # handles case when gold entity is masked should set target index to ignore it from loss
            targets = candidate_entity_targets.argmax(dim=1)
            no_cand_index = torch.zeros_like(targets)
            no_cand_index.fill_(
                scores.size(-1) - 1
            )
            # make NOTA (no_cand_score) the correct answer when the correct entity has no description
            targets = torch.where(
                scores[torch.arange(scores.size(0)), targets] != mask_value, targets, no_cand_index
            )
            loss = F.cross_entropy(scores, targets)

            # Changed this loss Nov 17 2022 (have not trained model with this yet)
            # loss = F.cross_entropy(scores, targets, ignore_index=scores.size(-1) - 1)
            # if all targets are ignore_index value then loss is nan in torch 1.11
            # https://github.com/pytorch/pytorch/issues/75181
            # how should loss be calculated when:
            # Q1 - if gold candidate is not in candidates
            # Answer - should make NOTA (no_cand_score) the correct answer
            #          because none of the provided descriptions match the gold entity
            # Q2 - if gold candidate has no description
            # Answer - should make NOTA (no_cand_score) the correct answer
            #          because none of the provided descriptions match the gold entity
            return loss, F.softmax(scores, dim=-1)
        else:
            return None, F.softmax(scores, dim=-1)  # output (num_ents, num_cands + 1)

    def init_weights(self):
        """Initialize weights for all member variables with type nn.Module"""
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
