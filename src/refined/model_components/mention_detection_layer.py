import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss


# This is the label index to ignore when calculating cross entropy loss
# IGNORE_INDEX = -1

IGNORE_INDEX = cel_default_ignore_index = CrossEntropyLoss().ignore_index


class MentionDetection(nn.Module):
    def __init__(self, num_labels: int, dropout: float, hidden_size: int):
        """
        Based on transformer's BertForTokenClassification class. Token
        :param num_labels: number of labels used for token
        :param dropout: dropout for linear layer
        :param hidden_size: embedding size
        """
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_labels)
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

    def forward(self, contextualised_embeddings=None, ner_labels=None):
        """
        Forward pass of Mention Detection layer.
        Treated as token classification problem where each token is assigned a BIO label.
        B = Begin entity (number usually used: 1)
        I = In entity (number usually used: 2)
        O = Not an entity (number usually used: 0)
        IGNORE_INDEX (-1) is used in `bio_label` positions to mark tokens to exclude from the loss function
        :param contextualised_embeddings: encoder's contextualised token embeddings
        :param ner_labels: BIO labels for tokens (length matches sequence_output, including special tokens)
        :return: loss tensor (if ner_labels is provided), token activations tensor
        """
        contextualised_embeddings = self.dropout(contextualised_embeddings)
        logits = self.linear(contextualised_embeddings)

        if ner_labels is not None:
            active_loss = ner_labels.view(-1) != IGNORE_INDEX
            active_logits = logits.view(-1, self.num_labels)
            # ignore loss from tokens where `bio_label` is -1
            active_labels = torch.where(
                active_loss, ner_labels.view(-1), torch.tensor(IGNORE_INDEX).type_as(ner_labels)
            )
            loss = F.cross_entropy(active_logits, active_labels, ignore_index=IGNORE_INDEX)
            return loss, logits

        return None, logits
