import logging
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel

from refined.doc_preprocessing.preprocessor import Preprocessor


class DescriptionEncoder(nn.Module):
    def __init__(
        self,
        preprocessor: Preprocessor,
        n_layer: int = 2,
        transformer_name: str = "roberta-base",
        output_dim: int = 300,
        dropout: float = 0.1,
        seq_len: int = 32,
        ff_chunk_size: int = 4,
        hidden_dim: int = 1000,
        add_hidden: bool = False,
    ):
        super().__init__()
        self.add_hidden = add_hidden
        self.dropout = nn.Dropout(dropout)
        config = deepcopy(preprocessor.transformer_config)
        config.output_attentions = False
        config.add_pooling_layer = False
        self.ff_chunk_size = ff_chunk_size
        if ff_chunk_size != 0:
            config.chunk_size_feed_forward = ff_chunk_size
        self.hidden_layer = nn.Linear(config.hidden_size, hidden_dim)
        if add_hidden:
            self.projection = nn.Linear(hidden_dim, output_dim)
        else:
            self.projection = nn.Linear(config.hidden_size, output_dim)
        self.init_weights()

        self.transformer: PreTrainedModel = preprocessor.get_transformer_model()
        self.transformer.encoder.layer = nn.ModuleList(
            self.transformer.encoder.layer[i] for i in range(n_layer)
        )
        if True:
            for param in list(self.transformer.embeddings.parameters()):
                param.requires_grad = False
            logging.debug(f"Froze BERT embedding layers (weights will be fixed during training) 2")

        self.tokenizer = preprocessor.tokenizer
        self.seq_len = seq_len
        self.output_dim = output_dim

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

    def forward(self, input_ids: Tensor):
        """
        Forward pass
        :param input_ids: input ids shape = (num_ents, max_cands, 32)  32 = description length, (descriptions)
        :return: forward pass output tensors
        """
        num_ents, max_cands, _ = input_ids.size()
        input_ids = input_ids.view((-1, input_ids.size(-1)))

        masked_cands_idx = (input_ids[:, 0] == self.tokenizer.pad_token_id).nonzero().squeeze(-1)
        unmasked_cands_idx = (input_ids[:, 0] != self.tokenizer.pad_token_id).nonzero().squeeze(-1)
        unmasked_cands = input_ids[unmasked_cands_idx]

        attention_mask = (unmasked_cands != self.tokenizer.pad_token_id).long()

        pad_candidate = torch.zeros(768, device=input_ids.device)
        embeddings = torch.zeros(input_ids.size(0), 768, device=input_ids.device)
        embeddings[masked_cands_idx] = pad_candidate

        if unmasked_cands.size(0) > 0:
            outputs = self.transformer(input_ids=unmasked_cands, attention_mask=attention_mask)
            embeddings[unmasked_cands_idx] = outputs[0][:, 0, :]

        if self.add_hidden:
            return self.projection(self.dropout(F.relu(self.hidden_layer(embeddings)))).view(
                (num_ents, max_cands, self.output_dim)
            )
        else:
            return self.projection(self.dropout(embeddings)).view(
                (num_ents, max_cands, self.output_dim)
            )
