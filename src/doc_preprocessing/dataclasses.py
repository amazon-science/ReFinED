import os
import re
import pickle
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from random import sample
from typing import Any, Dict, Iterable, List, NamedTuple, NoReturn, Optional, Set, Tuple

import nltk
import numpy as np
import torch
from dataclasses import dataclass, field
from model_components.config import MAX_SEQ
from torch import Tensor
from transformers import (
    AutoTokenizer,
    BertTokenizerFast,
    PreTrainedTokenizerFast,
    RobertaTokenizerFast,
)
from utilities.general_utils import batch_items, get_logger, get_tokenizer

TRANSFORMER_MAX_SEQ = 512

LOG = get_logger(__name__)


@dataclass
class DocPreprocessor(ABC):
    """
    Abstract class representing a preprocessor that provides methods to use resources such as Wikidata classes, and
    P(e|m) alias table. Implementations could memory-based, disk-based, or network-based depending on
    memory requirements.
    """

    data_dir: str
    transformer_name: str
    max_candidates: int
    ner_tag_to_ix: Dict[str, int]
    num_classes: int = field(init=False)
    max_num_classes_per_ent: int = field(init=False)
    cls_id: int = field(init=False)
    sep_id: int = field(init=False)
    pad_id: int = field(init=False)
    kg_embeddings: torch.Tensor = field(init=False)  # (num_ents, kg_embedding_dim)
    qcode_to_kge_idx_tns: torch.Tensor = field(init=False)  # (num_qcodes, num_ents)
    rel_embeddings: torch.Tensor = field(init=False)  # (num_rels, kg_embedding_dim)

    @abstractmethod
    def get_classes_idx_for_qcode_batch(
        self, qcodes: List[str], shape: Tuple = None
    ) -> torch.LongTensor:
        """
        Retrieves all of the classes indices for the qcodes (from various relations used to construct the lookup).
        :param qcodes: qcodes
        :param shape: shape/size of tensor returned (int)
        :return: long tensor with shape (num_ents, self.max_num_classes) (qcode index 0 is used for padding all classes)
        """
        pass

    @abstractmethod
    def set_precomputed_descriptions(self, file: str):
        pass

    @abstractmethod
    def get_descriptions_for_qcode_batch(
        self, qcodes: List[str], shape: Tuple[int, ...] = None
    ) -> torch.Tensor:
        """
        Retrieves descriptions input_ids for a batch of qcodes and optionally reshapes tensor.
        :param qcodes: qcodes
        :param shape: shape/size of tensor returned
        :return: long tensor with shape (num_ents, self.max_num_classes) (qcode index 0 is used for padding all classes)
        """
        pass

    @abstractmethod
    def get_candidates(
        self,
        surface_form: str,
        person_coref_ref: Dict[str, List[Tuple[str, float]]] = None,
        pruned_candidates: Optional[Set[str]] = None,
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        """
        Given a surface form (e.g. "Donald Trump") the method will return the top k (MAX_CANDIDATES) based on
        P(e|m) lookup dictionary. The method returns List[(qcodes, pem_value, pme_value)], person_coref.
        Person coref dictionary keeps track of humans mentioned in the document and copies the values from full name
        over partial name mentions (e.g. from "Donald Trump" to "Trump").
        :param surface_form: surface form to fetch candidates for
        :param person_coref_ref: a dictionary of previous human mentions with partial names in the dictionary
        :param pruned_candidates: if set it restricts candidate generation to this list
        :return: List[(qcode, pem_value)], person_coref
        """
        pass

    def add_candidates_to_spans(
        self,
        spans: List,
        backward_coref: bool = False,
        candidate_dropout: float = 0.0,
        sample_k_candidates: Optional[int] = None,
    ):
        """
        Adds candidate entities to each span.
        :param spans: list of spans
        :param backward_coref: if true will do 2 passes over spans (first pass collects human candidate entities,
                               and copies candidate over to partial mentions (e.g. surname instead of full name),
                               otherise will build person coreference dictionary sequentially so only forward coref
                               occurs.
        :param candidate_dropout: candidate_dropout
        :param sample_k_candidates: sample k candidates from top 30
        :return: no return (modifies spans in-place)
        """
        pass

    def map_title_to_qcode(self, title: str) -> Optional[str]:
        """
        Convert Wikipedia title to Wikidata ID (qcode).
        :param title: Wikipedia title
        :return: qcode for the Wikipedia title
        """
        pass


@dataclass
class AdditionalEDFeatures:
    """
    This class represents additional features used for Entity Disambiguation (ED).
    """

    edit_dist: float
    label_equals_m: float
    label_contains_m: float
    label_starts_with_m: float
    label_ends_with_m: float

    @classmethod
    def get_pad_element(cls):
        return cls(
            edit_dist=0.0,
            label_contains_m=0.0,
            label_equals_m=0.0,
            label_starts_with_m=0.0,
            label_ends_with_m=0.0,
        )


@dataclass
class Token:
    """
    This class represents a token in tokenized text.
    """

    text: str
    token_id: int
    start: int
    end: int


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


# TODO rename fields so can use ** for forward pass
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
    pme_values: Optional[Tensor] = None  # shape = (ent_len, 1)
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
    candidate_features: Optional[
        Tensor
    ] = None  # shape = (ent_len, max_candidates, 5)  5 = number of string features


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
    pme_values: Optional[Tensor] = None  # shape = (bs, ent_len, 1)
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
    candidate_desc_emb: Optional[Tensor] = None  # shape = (bs, ent_len, max_candidates, 300)
    # description embedding (precomputed)
    candidate_features: Optional[
        Tensor
    ] = None  # shape = (bs, ent_len, max_candidates, 5)  5 = number of features


@dataclass
class Date:
    text: Optional[str] = None
    day: Optional[int] = None
    month: Optional[int] = None
    year: Optional[int] = None

    known_format: bool = True

    def can_identify_format(self):

        if self.day is None or self.month is None:
            return True

        # If the date string contains any letters then assume we can identify format properly
        if bool(re.search(r"[a-zA-Z]", self.text)):
            return True

        # If the day is 12 or under, it is impossible to identify if this is a US format date (month first)
        if self.day <= 12:
            return False

        return True

    def get_doc_format(self) -> Optional[str]:

        if self.day is None or self.month is None:
            return None

        # If the date string contains any letters then does not give away format of number-only dates
        if bool(re.search(r"[a-zA-Z]", self.text)):
            return None

        # If day is 12 or under then this date doesn't reveal more general format
        if self.day <= 12:
            return None

        numbers_only = "".join([l for l in self.text if l.isnumeric()])

        if str(self.day) in numbers_only and numbers_only.index(str(self.day)) == 0:
            return "day_first"
        else:
            return "month_first"


@dataclass
class Span:
    """
    Represents a span (entity-mention) of text in a document, `Doc`.
    """

    text: str  # text of the span (must match text defined by indices/offsets)
    start: int  # start character offset in `Doc`
    ln: int  # length of span (number of characters) in `Doc`

    # (optional) gold standard for training
    mention_gold_classes: Optional[
        List[int]
    ] = None  # the classes for this mention (indices for a class_to_idx dict)
    gold_entity_id: Optional[str] = None  # can be used to retrieve gold classes if not provided

    # (optional) for inference
    candidate_entity_ids: Optional[List[Tuple[str, float]]] = None

    # (optional) results of inference
    pred_entity_id: Optional[Tuple[Dict[str, str], float]] = None  # (id, confidence)
    pred_ranked_entity_ids: Optional[List[Tuple[Iterable[str], float]]] = None  # (id, confidence)
    pred_types: Optional[List[Tuple[str, str, float]]] = None  # (type_id, type_label, confidence)

    coarse_type: Optional[str] = "MENTION"  # High level types such as (MENTION, DATE)

    coarse_mention_type: Optional[str] = None  # OntoNotes/spaCy types for mentions (ORG, LOC, PERSON)

    additional_ed_features: Optional[List[AdditionalEDFeatures]] = None

    # (optional) additional metadata
    metadata: Optional[Dict[Any, Any]] = None
    pruned_candidates: Optional[Set[str]] = None

    date: Optional[Date] = None

    entity_annotation: Optional[bool] = None
    entity_label: Optional[str] = None  # Wikidata label for the linked entity
    failed_class_check: Optional[bool] = None  # Indicates predicted class and actual entity class mismatch


class ModelReturn(NamedTuple):
    """Result of RefinedModel(nn.Module)."""

    md_loss: torch.Tensor
    md_activations: torch.Tensor
    et_loss: torch.Tensor
    et_activations: torch.Tensor
    ed_loss: torch.Tensor
    ed_activations: torch.Tensor
    spans: List[Span]
    cand_ids: torch.Tensor


@dataclass
class BatchElement:
    """
    Represents part of a document (`Doc`) that fits in single pass of a transformer model (< max_seq_len) as a
    singleton batch. If the document is short then the this object represents the full document.
    """

    tokens: List[BatchElementToken]
    entity_mask: Optional[
        List[int]
    ]  # indicates which batch_element_token acc_sums refer to entity-mentions
    spans: Optional[
        List[Span]
    ]  # spans used for entity typing and entity disambiguation (could be partial labels)
    text: str  # text from original document (note: this is the full text of the document)
    md_spans: Optional[
        List[Span]
    ] = None  # spans used for training mention detection (should be complete labels)

    def __post_init__(self):
        sort_spans(self.spans)
        sort_spans(self.md_spans)

    def add_spans(self, spans: List[Span]) -> NoReturn:
        """
        Adds spans, acc_sum, entity_mask to BatchElement.
        :param spans: list of spans to add
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
        :param ner_tag_to_ix: dict mapping from ner label (e.g. "B-PERSON") to index in output layer
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

    def to_ds_relations(self, preprocessor):
        # (num_mentions, num_mentions)
        # Example:
        # [Trump] was born in [New York], [USA]
        # rels: 0=no_relation, 1=birthplace, 2=country
        # rels = torch.tensor(
        #     [
        #     [0, 1, 0],  # trump   (trump birthplace obj)
        #     [0, 0, 2],  # new york  (new york country obj)
        #     [0, 0, 0]   # usa
        #     ]
        # )

        num_mentions = len(self.spans)
        pcode_to_idx = {
            pcode: idx for idx, pcode in enumerate(preprocessor.pcode_to_aliases.keys())
        }
        num_spans = len(self.spans)
        subj_to_obj_rel = np.zeros([num_spans, num_spans])

        for subj_idx, subj_span in enumerate(self.spans):
            subj_start_index = subj_span.start
            subj_end_index = subj_span.start + subj_span.ln
            subj_qcode = subj_span.gold_entity_id
            for obj_idx, obj_span in enumerate(self.spans):
                if obj_idx == subj_idx:
                    continue
                obj_start_index = obj_span.start
                obj_end_index = obj_span.start + obj_span.ln
                if obj_start_index < subj_start_index - 300:
                    continue
                if obj_start_index > subj_start_index + 300:
                    break
                obj_qcode = obj_span.gold_entity_id
                ent_pair = (subj_qcode, obj_qcode)
                if ent_pair in preprocessor.pairs_to_rels:
                    pcodes = preprocessor.pairs_to_rels[ent_pair]
                    for pcode in set(pcodes):
                        start_text_between = min(subj_start_index, obj_start_index)
                        end_text_between = max(subj_end_index, obj_end_index)
                        text_between = self.text[start_text_between:end_text_between]
                        if pcode not in preprocessor.pcode_to_aliases:
                            continue
                        aliases = preprocessor.pcode_to_aliases[pcode]
                        for alias in aliases:
                            if alias.lower() in text_between.lower():
                                label = aliases[-1][1:-1]
                                print(
                                    f"Triple: [{subj_span.text}] [{aliases[-1][1:-1]}] [{obj_span.text}]"
                                )
                                print(f"Matched alias ({alias[1:-1]})")
                                print(f"aliases ({aliases})")
                                print(f"Sentence:\n{text_between}")
                                print("\n\n")
                                print("*" * 50)
                                print("\n\n" * 3)
                                subj_to_obj_rel[subj_idx, obj_idx] = pcode_to_idx[pcode]

        print(subj_to_obj_rel)
        print(num_spans)
        return torch.tensor(subj_to_obj_rel)


@dataclass
class Doc:
    """
    Represents a document of text and spans.
    """

    text: str  # document text (no limit for length)
    tokens: List[Token]  # tokenized text
    spans: Optional[
        List[Span]
    ]  # optional entity-mention spans to process (ET/ED) - can be partial e.g. hyperlinks.
    metadata: Optional[Dict[Any, Any]] = None  # optional field for metadata
    md_spans: Optional[
        List[Span]
    ] = None  # optional entity-mention spans to process (MD only) - so must be complete.

    def __post_init__(self):
        if self.spans is not None:
            self.spans.sort(key=lambda x: x.start)
        if self.md_spans is not None:
            self.md_spans.sort(key=lambda x: x.start)
            # moved the merge here in case spans are sampled for ED and ET (keeps MD (span detection) consistent)
            self.md_spans = merge_spans(
                additional_spans=self.md_spans, prioritised_spans=self.spans
            )

    @classmethod
    def from_text(
        cls,
        text: str,
        transformer_name: str,
        data_dir: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Doc":
        """
        This method is used for end-to-end entity linking from text
        :param text: text
        :param transformer_name: transformer_name
        :param data_dir: data_dir
        :param metadata: metadata
        :return: doc
        """
        return cls(
            text=text,
            spans=None,
            tokens=tokenize(text, transformer_name=transformer_name, data_dir=data_dir),
            metadata=metadata,
        )

    @classmethod
    def from_text_with_spans(
        cls,
        text: str,
        spans: List[Span],
        preprocessor: DocPreprocessor,
        add_candidate: bool = True,
        lower_case_prob: float = 0.0,
        md_spans: Optional[List[Span]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        backward_coref: bool = False,
        candidate_dropout: float = 0.0,
        sample_k_candidates: Optional[int] = None,
    ) -> "Doc":
        """
        Construct `Doc` from text with predetermined spans.
        :param text: document text
        :param spans: document spans (with document character offsets)
        :param preprocessor: doc_prepreocessor used to add candidate entities
        :param add_candidate: if true will add candidate entities to spans
        :param lower_case_prob: probability that text will be lower cased (regularisation for training) (training)
        :param md_spans: mention detection spans, spans that may not have labels but are mentions (training MD layer)
        :param metadata: additional metadata to add to the `Doc`
        :param backward_coref: if true will do 2 passes over spans (first pass collects human candidate entities,
                               and copies candidate over to partial mentions (e.g. surname instead of full name),
                               otherwise will build person coreference dictionary sequentially so only forward coref
                               occurs.
        :param candidate_dropout: candidate_dropout
        :param sample_k_candidates: sample k candidates from top 30
        :return: `Doc`
        """
        text = text.lower() if random.random() < lower_case_prob else text
        tokens = tokenize(
            text, transformer_name=preprocessor.transformer_name, data_dir=preprocessor.data_dir
        )
        if add_candidate:
            preprocessor.add_candidates_to_spans(
                spans,
                backward_coref=backward_coref,
                candidate_dropout=candidate_dropout,
                sample_k_candidates=sample_k_candidates,
            )
        return cls(text=text, spans=spans, tokens=tokens, metadata=metadata, md_spans=md_spans)

    def to_batch_elements(
        self,
        data_dir: str,
        max_mentions: Optional[int] = None,
        override_max_seq: Optional[int] = None,
    ) -> List[BatchElement]:
        """
        Converts `Doc` to list of batch elements.
        When the doc has fewer than max sequence length tokens then the return list is a singleton list.
        If doc has more tokens than the max sequence length it is split into multiple BatchElements.
        Assertion exceptions raised if span is very long > 20 tokens.
        max_mentions: max_mentions
        :return: list of batch elements for the document (no limit on length of list)
        """
        # TODO: simplify this function and _to_batch_elements_e2e()
        if self.spans is None:
            # spans are not provided so end-to-end EL is required (the only input is document text)
            return self._to_batch_elements_e2e(data_dir=data_dir)

        self.spans.sort(key=lambda x: x.start)
        spans_queue: List[Span] = self.spans[:]
        if max_mentions is not None and len(spans_queue) > max_mentions:
            spans_queue = sample(spans_queue, k=max_mentions)
            spans_queue.sort(key=lambda x: x.start)

        # if there are no spans return no training items
        if len(spans_queue) == 0:
            return []

        batch_elements: List[BatchElement] = []

        # keeps track of the entity we are looking for
        # pop 0 makes the assumptions spans are sorted in ascending order
        # current ent is the last ent discovered - initially set it to the first ent in page
        next_span: Optional[Span] = spans_queue.pop(0)
        current_span: Optional[Span] = None
        if override_max_seq is not None:
            max_seq = override_max_seq
        else:
            max_seq: int = MAX_SEQ  # initially set max_seq to constant value (typically 300)
        # keeps track of tokens in the current item ("sentence" even though may not be actual sentence)
        sent_tokens: List[BatchElementToken] = []
        cumm_sum: int = 0  # keeps track of the accumulated sum of spans
        entity_pos: List[int] = []  # keeps track of which spans represent entities
        in_entity: bool = False  # keeps track of whether the for-loop has started but not finished reading an entity
        sent_spans: List[Span] = []  # keeps track of entities

        for idx, token in enumerate(self.tokens):
            # process tokens left-to-right
            # checks whether page length is at risk of exceeding `max_seq`
            # this check ensures that no entity spans across two training items (assumes no entity exceeds length 10)
            # idx > (max_seq - 1) may split entity across batches (only first batch will have span)
            if ((idx > (max_seq - 10)) and not in_entity) or (idx > (max_seq - 1)):
                max_mention = (
                    max([x.acc_sum for x in sent_tokens]) + 1
                )  # determine max number of spans (inc. masks)
                entity_mask = [
                    1 if i in entity_pos else 0 for i in range(max_mention)
                ]  # determine entities
                batch_element = BatchElement(
                    tokens=sent_tokens,
                    entity_mask=entity_mask,
                    spans=sent_spans,
                    text=self.text,
                    md_spans=self.md_spans,
                )
                # add training item to list and set variables for next item
                batch_elements.append(batch_element)
                sent_tokens = []
                sent_spans = []
                cumm_sum = 0
                entity_pos = []
                max_seq = (
                    idx + MAX_SEQ
                )  # update max_seq because offsets for next item are still relative to page text

            # process the next token (there are a few different cases to handle here)

            # case 1: a new entity has been entered
            # [2] = is doc_start, and [0] is doc_start
            if next_span is not None and token.start >= next_span.start:
                current_span = next_span
                cumm_sum += 1
                in_entity = True

                # update the next entity to look for
                if len(spans_queue) == 0:
                    next_span = None
                else:
                    # if consecutive entities: skip the current entity to ensure distance between current and next is 1
                    # you have entered the (next) entity so set it to current and move next to next entity
                    if not next_span == current_span:
                        current_span = next_span
                    next_span = spans_queue.pop(0)

                sent_tokens.append(
                    BatchElementToken(
                        acc_sum=cumm_sum,
                        text=token.text,
                        token_id=token.token_id,
                        start=token.start,
                        end=token.end,
                    )
                )
                entity_pos.append(cumm_sum)
                sent_spans.append(current_span)
                # token has been processed move on to the next token
                continue

            # case 2: the current entity has finished (and the next token is not an entity)
            # [2] is doc_start, [0] is doc start, [1] is doc length
            if current_span is not None and token.start >= (current_span.start + current_span.ln):
                cumm_sum += 1
                in_entity = False
                current_span = None
                sent_tokens.append(
                    BatchElementToken(
                        acc_sum=cumm_sum,
                        text=token.text,
                        token_id=token.token_id,
                        start=token.start,
                        end=token.end,
                    )
                )
                continue

            # case 3: is the general case, the token is normal not part of an entity start nor end
            sent_tokens.append(
                BatchElementToken(
                    acc_sum=cumm_sum,
                    text=token.text,
                    token_id=token.token_id,
                    start=token.start,
                    end=token.end,
                )
            )

        max_mention = max([x.acc_sum for x in sent_tokens]) + 1
        entity_mask = [1 if i in entity_pos else 0 for i in range(max_mention)]
        batch_element = BatchElement(
            tokens=sent_tokens,
            entity_mask=entity_mask,
            spans=sent_spans,
            text=self.text,
            md_spans=self.md_spans,
        )
        batch_elements.append(batch_element)

        # check all entities on the page have been consumed
        # if nested entities (links or spans) could cause problems
        if len(spans_queue) > 0:
            LOG.warning(f"[WARNING] Leftover: {spans_queue}")
        assert (
            len([1 for x in batch_elements if len(x.tokens) > TRANSFORMER_MAX_SEQ]) == 0
        ), "Exceeded max seq length"
        return batch_elements

    def _to_batch_elements_e2e(self, data_dir: str) -> List[BatchElement]:
        """
        E2E version of to_batch_elements().
        Converts `Doc` to list of batch elements.
        When the doc has fewer than max sequence length tokens then the return list is a singleton list.
        If doc has more tokens than the max sequence length it is split into multiple BatchElements.
        Assertion exceptions raised if span is very long > 20 tokens.
        :return: list of batch elements for the document (no limit on length of list)
        """
        sent_tokenizer = get_sent_tokenizer(data_dir=data_dir)
        sentence_boundaries = list(sent_tokenizer.span_tokenize(self.text))
        sent_to_tokens = defaultdict(list)
        current_sent_i = 0
        for temp_idx, token in enumerate(self.tokens):
            token_start, token_end = token.start, token.end
            if (
                token_start >= sentence_boundaries[current_sent_i][1]
                and current_sent_i < len(sentence_boundaries) - 1
            ):
                current_sent_i += 1
            sent_to_tokens[current_sent_i].append(token)

        batch_elements: List[BatchElement] = []
        current_batch_tokens: List[BatchElementToken] = []
        for sent_tokens in sent_to_tokens.values():
            # start new batch if needed (requires there to be a non-empty batch to start with)
            if (
                len(current_batch_tokens) + len(sent_tokens) > MAX_SEQ
                and len(current_batch_tokens) > 0
            ):
                # the addition of this sentence will exceed seq len so start a new batch for it
                batch_elements.append(
                    BatchElement(
                        tokens=current_batch_tokens,
                        entity_mask=None,
                        spans=None,
                        text=self.text,
                        md_spans=None,
                    )
                )
                current_batch_tokens = []

            # add sentence to the current batch (could be the newly created from the statement above)
            if len(sent_tokens) + len(current_batch_tokens) <= MAX_SEQ:
                # there is space in the batch for the addition of this sentence
                current_batch_tokens.extend(
                    [
                        BatchElementToken(
                            acc_sum=None, text=t.text, token_id=t.token_id, start=t.start, end=t.end
                        )
                        for t in sent_tokens
                    ]
                )
            else:
                # len(sent_tokens) + len(current_batch_tokens) > MAX_SEQ and len(current_batch_tokens) = 0
                # the sentence is too long for a single batch so split it up
                # from first branch of if statement clearly first
                for sent_tokens_part in batch_items(sent_tokens, n=MAX_SEQ):
                    current_batch_tokens.extend(
                        [
                            BatchElementToken(
                                acc_sum=None,
                                text=t.text,
                                token_id=t.token_id,
                                start=t.start,
                                end=t.end,
                            )
                            for t in sent_tokens_part
                        ]
                    )
                    batch_elements.append(
                        BatchElement(
                            tokens=current_batch_tokens,
                            entity_mask=None,
                            spans=None,
                            text=self.text,
                            md_spans=None,
                        )
                    )
                    current_batch_tokens = []

        # end last batch if needed
        if len(current_batch_tokens) > 0:
            batch_elements.append(
                BatchElement(
                    tokens=current_batch_tokens,
                    entity_mask=None,
                    spans=None,
                    text=self.text,
                    md_spans=None,
                )
            )
        return batch_elements


sent_tokenizer = None


def get_sent_tokenizer(data_dir: str):
    global sent_tokenizer
    if sent_tokenizer is not None:
        return sent_tokenizer
    try:
        nltk.download("punkt")
        sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    except Exception as err:
        sent_tokenizer = nltk.data.load(
            os.path.join(data_dir, "nltk/tokenizers/punkt/english.pickle")
        )
    return sent_tokenizer


tokenizers = dict()


def get_tokenizer_cached(transformer_name: str, data_dir: str) -> PreTrainedTokenizerFast:
    """
    Get TokenizerFast for the specified transformer name. This differs from AutoTokenizer.from_pretrained because
    the tokenizer has already been instatiated and has specific configuration.
    :param transformer_name: transformer name
    :param data_dir: data dir containing the folder containing the tokenizer
    :return: TokenizerFast
    """
    global tokenizers
    if transformer_name in tokenizers:
        return tokenizers[transformer_name]
    tokenizers[transformer_name] = get_tokenizer(
        transformer_name=transformer_name,
        data_dir=data_dir,
        add_special_tokens=False,
        add_prefix_space=False,
        use_fast=True,
    )
    return tokenizers[transformer_name]


# tokenize is placed here to prevent circular dependency because this module uses it and it uses this module
def tokenize(text: str, transformer_name: str, data_dir: str) -> List[Token]:
    """
    Tokenize text using tokenizer fast.
    :param text: text
    :param transformer_name: transformer name
    :return: list of tokens
    :param data_dir: data dir containing the folder containing the tokenizer
    """
    if len(text) == 0:
        return []
    tokenizer = get_tokenizer_cached(transformer_name, data_dir=data_dir)
    try:
        token_res = tokenizer.encode_plus(
            text,
            return_offsets_mapping=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        word_pieces = tokenizer.convert_ids_to_tokens(
            token_res["input_ids"]
        )  # stores word pieces for debugging
        return [
            Token(word_piece, token_id, start, end)
            for word_piece, token_id, (start, end) in zip(
                word_pieces, token_res["input_ids"], token_res["offset_mapping"]
            )
        ]
    except IndexError:
        LOG.info(f"Skipping article as failed to tokenize: {text}")
        return []


def merge_spans(additional_spans: List[Span], prioritised_spans: List[Span]) -> List[Span]:
    """
    Merge to lists of spans. It ensures the spans do not overlap. When there is overlap prioritised_spans is picked.
    :param additional_spans: the spans to combine with the prioritised_spans (usually predicted)
    :param prioritised_spans: the spans to  prioritise when there is overlap (usually (semi-)supervised)
    :return: a sorted list of merged spans.
    """
    spans = []
    taken_indices = set()
    for prioritised_span in prioritised_spans:
        taken_indices.update(
            range(prioritised_span.start, prioritised_span.start + prioritised_span.ln)
        )
    for prioritised_span in prioritised_spans:
        spans.append(prioritised_span)
    for additional_span in additional_spans:
        if (
            len(
                set(range(additional_span.start, additional_span.start + additional_span.ln))
                & taken_indices
            )
            == 0
        ):
            spans.append(additional_span)

    sort_spans(spans)
    return spans


def sort_spans(spans: Optional[List[Span]]):
    """
    In-place sort spans
    :param spans: spans
    """
    if spans is not None:
        spans.sort(key=lambda x: x.start)
