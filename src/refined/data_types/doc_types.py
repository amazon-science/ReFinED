import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

from refined.data_types.base_types import Token, Span
from refined.data_types.modelling_types import BatchElementToken, BatchElement
from refined.doc_preprocessing.preprocessor import Preprocessor
from refined.utilities.general_utils import batch_items, get_logger, merge_spans

LOG = get_logger(__name__)


@dataclass
class Doc:
    """
    Represents a document of text and spans.
    """
    # This number uniquely identifies this document.
    doc_id: int

    # document text (no limit for length)
    text: str

    # tokenized text
    tokens: List[Token]

    # optional entity-mention spans to process (ET/ED) - can be partial e.g. hyperlinks.
    spans: Optional[List[Span]]

    # optional entity-mention spans to process (MD only) - so must be complete.
    md_spans: Optional[List[Span]] = None

    def __post_init__(self):
        if self.spans is not None:
            self.spans.sort(key=lambda x: x.start)
        if self.md_spans is not None:
            self.md_spans.sort(key=lambda x: x.start)
            # moved the merge here in case spans are sampled for ED and ET (keeps MD (span detection) consistent)
            self.md_spans = merge_spans(
                additional_spans=self.md_spans, prioritised_spans=self.spans
            )
        if self.doc_id is None:
            self.doc_id = random.randint(0, 2 ** 30)

    @classmethod
    def from_text(
            cls,
            text: str,
            preprocessor: Preprocessor,
            doc_id: Optional[int] = None
    ) -> "Doc":
        """
        This method is used for end-to-end entity linking from text
        :param text: text
        :param preprocessor: preprocessor
        :return: doc
        """
        if doc_id is None:
            doc_id = random.randint(0, 2 ** 30)
        return cls(
            text=text,
            spans=None,
            tokens=preprocessor.tokenize(text),
            doc_id=doc_id
        )

    @classmethod
    def from_text_with_spans(
            cls,
            text: str,
            spans: List[Span],
            preprocessor: Preprocessor,
            add_candidate: bool = True,
            lower_case_prob: float = 0.0,
            md_spans: Optional[List[Span]] = None,
            backward_coref: bool = False,
            sample_k_candidates: Optional[int] = None,
            doc_id: Optional[int] = None
    ) -> "Doc":
        """
        Construct `Doc` from text with predetermined spans.
        :param text: document text
        :param spans: document spans (with document character offsets)
        :param preprocessor: doc_preprocessor used to add candidate entities
        :param add_candidate: if true will add candidate entities to spans
        :param lower_case_prob: probability that text will be lower cased (regularisation for training) (training)
        :param md_spans: mention detection spans, spans that may not have labels but are mentions (training MD layer)
        :param backward_coref: if true will do 2 passes over spans (first pass collects human candidate entities,
                               and copies candidate over to partial mentions (e.g. surname instead of full name),
                               otherwise will build person coreference dictionary sequentially so only forward coref
                               occurs.
        :param sample_k_candidates: randomly samples candidates (hard, random, and gold)
        :return: `Doc`
        """
        if doc_id is None:
            doc_id = random.randint(0, 2 ** 30)
        if spans is not None:
            for span in spans:
                span.doc_id = doc_id
        if md_spans is not None:
            for span in md_spans:
                span.doc_id = doc_id
        text = text.lower() if random.random() < lower_case_prob else text
        tokens = preprocessor.tokenize(
            text
        )
        if add_candidate:
            preprocessor.add_candidates_to_spans(
                spans,
                backward_coref=backward_coref,
                sample_k_candidates=sample_k_candidates,
            )
        return cls(text=text, spans=spans, tokens=tokens, md_spans=md_spans, doc_id=doc_id)

    def to_batch_elements(
            self,
            preprocessor: Preprocessor,
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
            return self._to_batch_elements_e2e(preprocessor=preprocessor)

        self.spans.sort(key=lambda x: x.start)
        spans_queue: List[Span] = self.spans[:]

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
            max_seq: int = preprocessor.max_seq  # initially set max_seq to constant value (typically 300)
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
            if ((idx > (max_seq - 10)) and not in_entity) \
                    or (idx > (max_seq - 1)) \
                    or (max_mentions is not None and len(sent_spans) > max_mentions and not in_entity):
                # chunk the page into `max_mentions` mention chunks
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
                    doc_id=self.doc_id
                )
                # add training item to list and set variables for next item
                batch_elements.append(batch_element)
                sent_tokens = []
                sent_spans = []
                cumm_sum = 0
                entity_pos = []
                max_seq = (
                        idx + preprocessor.max_seq
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
            doc_id=self.doc_id
        )
        batch_elements.append(batch_element)

        # check all entities on the page have been consumed
        # if nested entities (links or spans) could cause problems
        if len(spans_queue) > 0:
            LOG.warning(f"[WARNING] Leftover: {spans_queue}")
        assert (
                len([1 for x in batch_elements if len(x.tokens) > preprocessor.max_seq]) == 0
        ), "Exceeded max seq length"
        return batch_elements

    def _to_batch_elements_e2e(self, preprocessor: Preprocessor) -> List[BatchElement]:
        """
        E2E version of to_batch_elements().
        Converts `Doc` to list of batch elements.
        When the doc has fewer than max sequence length tokens then the return list is a singleton list.
        If doc has more tokens than the max sequence length it is split into multiple BatchElements.
        Assertion exceptions raised if span is very long > 20 tokens.
        :return: list of batch elements for the document (no limit on length of list)
        """
        sentence_boundaries = list(preprocessor.split_sentences(self.text))
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
                    len(current_batch_tokens) + len(sent_tokens) > preprocessor.max_seq
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
                        doc_id=self.doc_id
                    )
                )
                current_batch_tokens = []

            # add sentence to the current batch (could be the newly created from the statement above)
            if len(sent_tokens) + len(current_batch_tokens) <= preprocessor.max_seq:
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
                # len(sent_tokens) + len(current_batch_tokens) > preprocessor.max_seq and len(current_batch_tokens) = 0
                # the sentence is too long for a single batch so split it up
                # from first branch of if statement clearly first
                for sent_tokens_part in batch_items(sent_tokens, n=preprocessor.max_seq):
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
                            doc_id=self.doc_id
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
                    doc_id=self.doc_id
                )
            )
        return batch_elements
