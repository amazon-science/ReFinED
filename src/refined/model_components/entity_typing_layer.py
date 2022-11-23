from typing import Optional, Tuple

from torch import nn
from torch import Tensor


class EntityTyping(nn.Module):
    def __init__(self, dropout: float, num_classes: int, encoder_hidden_size: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # not needed
        self.linear = nn.Linear(encoder_hidden_size, num_classes)
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
        self, mention_embeddings: Tensor, span_classes: Optional[Tensor] = None
    ) -> Tuple[Optional[Tensor], Tensor]:
        """
        Forward pass of Entity Typing layer.
        Averages the encoder's contextualised token embeddings for each mention span.
        :param mention_embeddings: mention embeddings
        :param span_classes: class targets
        :return: loss tensor (if span_classes is provided), class activations tensor
        """

        logits = self.linear(mention_embeddings)
        # activations = logits
        if span_classes is not None:
            targets = span_classes
            # loss_function = nn.BCELoss()
            # torch.nn.BCELoss is unsafe to autocast
            loss_function = nn.BCEWithLogitsLoss()
            loss = loss_function(logits, targets)
            return loss, logits.sigmoid()

        return None, logits.sigmoid()
