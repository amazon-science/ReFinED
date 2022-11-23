from dataclasses import dataclass, field
from typing import Optional, NamedTuple, List, NoReturn, Dict

import torch
from torch import Tensor

from refined.data_types.base_types import Span
from refined.utilities.general_utils import get_logger, merge_spans, sort_spans

LOG = get_logger(__name__)


@dataclass
class BatchElementToken:
    """
    This class represents a token used to produce tensors for a forward pass of a nn. The `acc_sum` is used for
    mean pooling contiguous token for mention spans.
    Example of `BatchElementToken` with accumulated sum set
     [CLS] Donald Trump was born in New York [SEP]    (tokens)
       0    1       1    2   2    2  3   3     4      (accumulated sum)
       0    1            0           1         0      (entity mask for reference)
    num entities = 2
    num spans (not same as `Span`) = 5
    start and end are original text character indices.
    """

    acc_sum: Optional[int]
    text: str
    token_id: int
    start: int
    end: int


class BatchElementTns(NamedTuple):
    """Batch element as a tuple of tensors. This tuple is for a single `BatchElement`."""

    token_id_values: Tensor = None  # shape = (seq_len, )
    token_acc_sum_values: Optional[Tensor] = None  # shape = (seq_len, )
    entity_mask_values: Optional[Tensor] = None  # shape = (ent_len, )
    class_target_values: Optional[Tensor] = None  # shape = (all_ent_len, max_num_classes_per_ent)
    attention_mask_values: Tensor = None  # shape = (seq_len, )
    token_type_values: Tensor = None  # shape = (seq_len, )
    candidate_target_values: Optional[Tensor] = None  # shape = (ent_len, max_candidates + 1)
    pem_values: Optional[Tensor] = None  # shape = (ent_len, seq_len)
    candidate_class_values: Optional[Tensor] = None  # shape = (ent_len, max_num_classes_per_ent)
    entity_index_mask_values: Optional[Tensor] = None  # shape = (num_ents)
    batch_element: "BatchElement" = None  # python object used to create this tuple of tensors
    gold_qcode_values: Optional[Tensor] = None  # shape = (all_ent_len, 1)
    candidate_qcode_values: Optional[Tensor] = None  # shape = (ent_len, 1)
    ner_labels: Optional[Tensor] = None  # shape = (seq_len, )
    candidate_desc: Optional[
        Tensor
    ] = None  # shape = (ent_len, max_candidates, 32)  32 = description token length
    candidate_desc_emb: Optional[Tensor] = None  # shape = (ent_len, max_candidates, 300)
    # description embedding (precomputed)


class BatchedElementsTns(NamedTuple):
    """
    Represents the tensors for a list/batch of `BatchElement` that have been collated (vertically stacking in dim=1).
    Equivalent to `BatchElementTns` but tensors have a batch dimension and batch_element is replaced with
    """

    token_id_values: Tensor = None  # shape = (bs, seq_len)
    token_acc_sum_values: Optional[Tensor] = None  # shape = (bs, seq_len)
    entity_mask_values: Optional[Tensor] = None  # shape = (bs, ent_len)
    class_target_values: Optional[Tensor] = None  # shape = (all_ent_len, max_num_classes_per_ent)
    attention_mask_values: Tensor = None  # shape = (bs, seq_len)
    token_type_values: Tensor = None  # shape = (bs, seq_len)
    candidate_target_values: Optional[Tensor] = None  # shape = (bs, ent_len, max_candidates + 1)
    pem_values: Optional[Tensor] = None  # shape = (bs, ent_len, seq_len)
    candidate_class_values: Optional[
        Tensor
    ] = None  # shape = (bs, ent_len, max_num_classes_per_ent)
    entity_index_mask_values: Tensor = None  # shape = (bs, num_ents)
    batch_elements: Optional[
        List["BatchElement"]
    ] = None  # python object used to create this tuple of tensors
    gold_qcode_values: Optional[Tensor] = None  # shape = (all_ent_len, 1)
    candidate_qcode_values: Optional[Tensor] = None  # shape = (bs, ent_len, 1)
    ner_labels: Optional[Tensor] = None  # shape = (bs, seq_len)
    candidate_desc: Optional[
        Tensor
    ] = None  # shape = (bs, ent_len, max_candidates, 32)  32 = description token length
    # description embedding (precomputed)
    candidate_desc_emb: Optional[Tensor] = None  # shape = (bs, ent_len, max_candidates, 300)

    def to(self, device: str) -> 'BatchedElementsTns':
        return BatchedElementsTns(
            *[x.to(device) if isinstance(x, torch.Tensor) else x for x in self]
        )


class ModelReturn(NamedTuple):
    """Result of RefinedModel(nn.Module)."""
    md_loss: Optional[Tensor]
    md_activations: Tensor
    et_loss: Optional[Tensor]
    et_activations: Tensor
    ed_loss: Optional[Tensor]
    ed_activations: Tensor
    entity_spans: List[Span]
    other_spans: Dict[str, List[Span]]
    cand_ids: Tensor
    description_loss: Optional[Tensor]
    candidate_description_scores: Tensor


@dataclass
class BatchElement:
    """
    Represents part of a document (`Doc`) that fits in single pass of a transformer model (< max_seq_len) as a
    singleton batch. If the document is short then this object represents the full document.
    """

    # The document ID for the document where the batch_element was created from.
    doc_id: int

    tokens: List[BatchElementToken]

    # indicates which batch_element_token acc_sums refer to entity-mentions
    entity_mask: Optional[List[int]]

    # spans used for entity typing and entity disambiguation (could be partial labels)
    spans: Optional[List[Span]]

    # text from original document (note: this is the full text of the document)
    text: str

    # spans used for training mention detection (should be complete labels)
    md_spans: Optional[List[Span]] = None

    def __post_init__(self):
        sort_spans(self.spans)
        sort_spans(self.md_spans)

    def add_spans(self, spans: List[Span]) -> NoReturn:
        """
        Adds spans, acc_sum, entity_mask to BatchElement.
        param spans: list of spans to add
        :return: no return modifies self in-place.
        """
        sort_spans(spans)
        self.spans = spans
        spans_queue: List[Span] = self.spans[:]

        acc_sum: int = 0  # keeps track of the accumulated sum of spans
        entity_pos: List[int] = []  # keeps track of which spans represent entities

        # keeps track of the entity we are looking for
        # pop 0 makes the assumptions spans are sorted in ascending order
        # current ent is the last ent discovered - initially set it to the first ent in page
        next_span = spans_queue.pop(0) if len(spans_queue) > 0 else None
        current_span: Optional[Span] = None

        # TODO: reduce duplication with to_batch_elements() method
        for token in self.tokens:
            # process the next token (there are a few different cases to handle here)
            # case 1: a new entity has been entered
            # [2] = is doc_start, and [0] is doc_start
            if next_span is not None and token.start >= next_span.start:
                current_span = next_span
                acc_sum += 1
                # update the next entity to look for
                if len(spans_queue) == 0:
                    next_span = None
                else:
                    # if consecutive entities: skip the current entity to ensure distance between current and next is 1
                    # you have entered the (next) entity so set it to current and move next to next entity
                    if not next_span == current_span:
                        current_span = next_span
                    next_span = spans_queue.pop(0)

                token.acc_sum = acc_sum
                entity_pos.append(acc_sum)
                # token has been processed move on to the next token
                continue

            # case 2: the current entity has finished (and the next token is not an entity)
            if current_span is not None and token.start >= (current_span.start + current_span.ln):
                acc_sum += 1
                current_span = None
                token.acc_sum = acc_sum
                continue

            # case 3: is the general case, the token is normal not part of an entity start nor end
            token.acc_sum = acc_sum

        max_mention = max([x.acc_sum for x in self.tokens]) + 1
        entity_mask = [1 if i in entity_pos else 0 for i in range(max_mention)]
        self.entity_mask = entity_mask

        # check all entities on the page have been consumed
        # if nested entities (links or spans) could cause problems
        if len(spans_queue) > 0:
            LOG.warning(f"Leftover: {spans_queue}")

    def to_ner_labels(
            self, ner_tag_to_ix: Dict[str, int], mask_o: bool = False, add_special_token_labels: bool = False
    ) -> List[int]:
        """
        Convert `BatchElement` to labels for NER.
        param ner_tag_to_ix: dict mapping from ner label (e.g. "B-PERSON") to index in output layer
        :param mask_o: mask non-entity label with -1 (ignore no entity labels during training)
        :param add_special_token_labels: add 2 additional no-entity labels for special tokens (cls and sep)
        :return: a list of int (1 = B, 2 = I, O = 0, mask=-1)
        """
        # assumes that 0 is never followed by 2
        # b = 1
        # i = 2
        # o = 0 if (not mask_o) else -1

        if mask_o:
            ner_tag_to_ix['O'] = -1

        md_spans: List[Span] = self.md_spans[:] if self.md_spans is not None else []
        spans: List[Span] = self.spans[:]
        spans = merge_spans(additional_spans=md_spans, prioritised_spans=spans)
        batch_element_start = self.tokens[0].start if len(self.tokens) > 0 else 10e10
        span_indices = {}
        for ent in spans:
            for char_ix in range(ent.start, ent.start + ent.ln):
                span_indices[char_ix] = ent.coarse_type

        ner_labels = []
        start_indices = [ent.start for ent in spans if ent.start >= batch_element_start]
        next_start = start_indices.pop(0) if len(start_indices) > 0 else None
        for token in self.tokens:
            if token.start in span_indices or token.end - 1 in span_indices:

                coarse_type = span_indices[token.start] if token.start in span_indices else span_indices[token.end - 1]
                ner_label = "-" + coarse_type if coarse_type is not None else ""

                if next_start is not None and token.start >= next_start:
                    next_start = start_indices.pop(0) if len(start_indices) > 0 else None
                    ner_labels.append(ner_tag_to_ix["B" + ner_label])
                else:
                    ner_labels.append(ner_tag_to_ix["I" + ner_label])
            else:
                ner_labels.append(ner_tag_to_ix["O"])
        if add_special_token_labels:
            ner_labels = [ner_tag_to_ix['O']] + ner_labels + [ner_tag_to_ix['O']]

        return ner_labels


@dataclass
class ReFinEDConfig:
    ner_tag_to_ix: Dict[str, int] = field(default_factory=dict)
    data_dir: Optional[str] = None
    transformer_name: str = 'roberta-base'
    max_candidates: int = 30
    debug: bool = False
