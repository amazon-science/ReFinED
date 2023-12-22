import logging
from typing import Any, Iterable, List, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from refined.data_types.doc_types import (
    Doc,
)
from refined.data_types.modelling_types import BatchElementTns, BatchedElementsTns, BatchElement
from refined.doc_preprocessing.preprocessor import Preprocessor
from refined.model_components.mention_detection_layer import IGNORE_INDEX
from refined.utilities.general_utils import batch_items

LOG = logging.getLogger()


def convert_batch_element_to_tensors(
        batch_element: BatchElement, processor: Preprocessor
) -> BatchElementTns:
    """
    Converts batch element to tuple of tensors.
    :param batch_element: batch element to convert
    :param processor: preprocessor used to lookup classes for qcodes
    :return: tuple of tensors for the batch element
    """
    # TODO: this function is too long and short be made simpler
    # it's long because it has to handle different multi-task training setups and inference (without spans)

    token_ln = len(batch_element.tokens) + 2  # special tokens
    token_id_values = torch.zeros((token_ln,), dtype=torch.long)
    token_id_values.fill_(processor.pad_id)
    token_type_values = torch.zeros((token_ln,), dtype=torch.long)
    attention_mask_values = torch.ones((token_ln,), dtype=torch.long)
    token_id_values[1:-1] = torch.from_numpy(np.array([t.token_id for t in batch_element.tokens]))
    token_id_values[0] = processor.cls_id
    token_id_values[-1] = processor.sep_id

    if batch_element.spans is None or len(batch_element.spans) == 0:
        # no spans are provided so this is the only information available
        # spans will be identified as part of the forward pass. This is for inference only.
        return BatchElementTns(
            token_id_values=token_id_values,
            attention_mask_values=attention_mask_values,
            token_type_values=token_type_values,
            batch_element=batch_element,
        )

    num_ents = len(batch_element.spans)
    token_acc_sum_values = torch.zeros((token_ln,), dtype=torch.long)
    # max_candidates should handle sampled candidates
    if len(batch_element.spans) > 0:
        max_candidates = len(batch_element.spans[0].candidate_entities)
    else:
        max_candidates = processor.max_candidates

    entity_mask_values = torch.tensor(
        batch_element.entity_mask, dtype=torch.long
    )  # (max_mentions, )
    pem_values = torch.zeros((num_ents, max_candidates), dtype=torch.float32)
    candidate_class_values = torch.zeros(
        (num_ents, max_candidates, processor.max_num_classes_per_ent), dtype=torch.long
    )

    # 32 is description sequence length (should not be hard-coded)
    # candidate_desc = torch.zeros((num_ents, max_candidates, 32), dtype=torch.long)
    candidate_desc = None
    candidate_desc_emb = None

    entity_index_mask_values = torch.ones((num_ents,), dtype=torch.bool)
    candidate_qcode_values = torch.zeros((num_ents, max_candidates), dtype=torch.long)

    if (
            num_ents > 0
            and (batch_element.spans[0].gold_entity is None or
                 batch_element.spans[0].gold_entity.wikidata_entity_id is None)
    ) or num_ents == 0:
        has_gold_label = False
        candidate_target_values = None
        class_target_values = None
        gold_qcode_values = None
    else:
        has_gold_label = True
        candidate_target_values = torch.zeros((num_ents, max_candidates + 1), dtype=torch.long)
        class_target_values = torch.zeros(
            (num_ents, processor.max_num_classes_per_ent), dtype=torch.long
        )
        if len(batch_element.spans) > 0 and batch_element.spans[0].gold_entity.wikidata_entity_id is not None:
            gold_qcode_values = torch.zeros((num_ents,), dtype=torch.long)
        else:
            gold_qcode_values = None

    if has_gold_label:
        # gold_qcode_values must be set so use the qcode to get the classes (types)
        gold_qcodes = [span.gold_entity.wikidata_entity_id for span in batch_element.spans]
        gold_qcodes = ["Q0" if qcode is None else qcode for qcode in gold_qcodes]
        gold_qcode_values[:num_ents] = torch.from_numpy(
            np.array(list(map(lambda x: int(x.replace("Q", "")), gold_qcodes)))
        )
        # look up the classes from gold qcode
        class_target_values[:num_ents] = processor.get_classes_idx_for_qcode_batch(
            gold_qcodes, shape=(-1, processor.max_num_classes_per_ent)
        )

    if num_ents > 0:
        candidate_qcodes = [str(y[0]) for x in batch_element.spans for y in x.candidate_entities]
        candidate_qcodes_ints = np.array(
            list(map(lambda x: int(x.replace("Q", "")), candidate_qcodes))
        ).reshape(num_ents, max_candidates)
        candidate_qcode_values[:num_ents, :max_candidates] = torch.from_numpy(candidate_qcodes_ints)

        candidate_class_values[:num_ents, :max_candidates] = processor.get_classes_idx_for_qcode_batch(
            candidate_qcodes, shape=(-1, max_candidates, processor.max_num_classes_per_ent)
        )

        if processor.precomputed_descriptions is not None:
            candidate_desc_emb = processor.get_descriptions_emb_for_qcode_batch(
                candidate_qcodes, shape=(num_ents, max_candidates, -1)
            )
            candidate_desc = None
        else:
            candidate_desc = processor.get_descriptions_for_qcode_batch(
                candidate_qcodes, shape=(-1, max_candidates, 32)
            )  # TODO replace constant 32
            candidate_desc_emb = None

        candidate_pem_values = np.array(
            [y[1] for x in batch_element.spans for y in x.candidate_entities]
        ).reshape(num_ents, max_candidates)

        pem_values[:num_ents, :max_candidates] = torch.from_numpy(candidate_pem_values)

    for ent_idx, ent in enumerate(batch_element.spans):
        # must handle None values keep ordering same but with None placeholders
        if has_gold_label:
            gold_qcode = ent.gold_entity.wikidata_entity_id
            gold_ent_in_cands = False

        for cand_idx, (qcode, pem_value) in enumerate(ent.candidate_entities):
            if has_gold_label and str(qcode) == str(gold_qcode):
                gold_ent_in_cands = True
                candidate_target_values[ent_idx, cand_idx] = 1
        if has_gold_label:
            candidate_target_values[ent_idx, -1] = 0 if gold_ent_in_cands else 1  # no entity target

    # careful not to skip 1 (accumulated sum)
    # if start with entity (e.g. [CLS] EU rejects German call... is [0, 1, 2, 3, 4,...] not [0, 2, 3, 4, 5,...]
    token_acc_sum_values[1:-1] = torch.from_numpy(
        np.array([t.acc_sum for t in batch_element.tokens])
    )
    token_acc_sum_values[0] = 0
    token_acc_sum_values[-1] = 0

    ner_labels = torch.tensor(
        batch_element.to_ner_labels(add_special_token_labels=True,
                                    mask_o=False,
                                    ner_tag_to_ix=processor.ner_tag_to_ix),
        dtype=torch.long
    )

    return BatchElementTns(
        token_id_values,
        token_acc_sum_values,
        entity_mask_values,
        class_target_values,
        attention_mask_values,
        token_type_values,
        candidate_target_values,
        pem_values,
        candidate_class_values,
        entity_index_mask_values,
        batch_element,
        gold_qcode_values,
        candidate_qcode_values,
        ner_labels,
        candidate_desc,
        candidate_desc_emb=candidate_desc_emb
    )


def convert_batch_elements_to_batched_tns(
        batch_elements: List[BatchElement],
        preprocessor: Preprocessor,
        max_batch_size: int = 16,
        sort_by_tokens: bool = True,
) -> Iterable[BatchedElementsTns]:
    """
    :param batch_elements: elements to batch.
    :param preprocessor: doc preprocessor.
    :param max_batch_size: max batch size.
    :param sort_by_tokens: whether to sort batch elements by tokens (improves batch efficiency) but means seq for
                           documents might not appear in order next to each other.
    :return: yields generator of batched tensors.
    """
    if sort_by_tokens:
        batch_elements.sort(
            key=lambda x: len(x.tokens), reverse=True
        )  # batch similar length sequences together
    # TODO: think carefully about whether it is useful to train the model on sequences with no entities.
    # If there are no entities in some chunks in the evaluation data this may be useful.
    # For now we are filtering chunks of text that have no spans.
    batch_elements = list(filter(lambda x: x.spans is None or len(x.spans) > 0, batch_elements))
    batched_elements: List[List[BatchElement]] = list(batch_items(batch_elements, n=max_batch_size))
    for batched_element in batched_elements:
        if len(batched_element) == 0:
            continue
        elems_tns = []
        for elem_idx, elem in enumerate(batched_element):
            elems_tns.append(convert_batch_element_to_tensors(elem, preprocessor))
        yield collate_batch_elements_tns(elems_tns, token_pad_value=preprocessor.pad_id)


def convert_doc_to_tensors(
        doc: Doc,
        preprocessor: Preprocessor,
        max_batch_size=16,
        collate=False,
        sort_by_tokens: bool = True,
        max_seq: int = 510,
) -> Union[Iterable[BatchElementTns], Iterable[BatchedElementsTns]]:
    """
    Converts `Doc` into tensors.

    This method is probably not the method you want to use because it returns tensors for the `Doc` only.
    Batching will be inefficient because short documents will be singleton batches. Long documents will contain
    a batch that is less than the max batch size.
    Therefore, this method should only be used when only a single document can be processed at a time.
    For efficient batching, group together `BatchElements` with similar sequence lengths into batches of about
    16 and then convert them to tensors and collate them. Converting `BatchElement` to tensors requires memory
    because all candidate entities for all spans must be binarized (binary encoded) which is a large tensor so
    conversions should not be made unnecessarily.
    :param doc: Doc
    :param preprocessor: preprocessor to use to lookup classes for qcodes
    :param max_batch_size: maximum batch size only used when collate is true
    :param collate: determines whether to collate the tensors into batches or just return a list of tensors
    :param sort_by_tokens: sort batch elements by token length
    :param max_seq: max sequence length for tokens
    :return: a list of batch element tensors (list elements are batches if collate is True)
    """
    if collate:
        return convert_batch_elements_to_batched_tns(
            doc.to_batch_elements(preprocessor=preprocessor, override_max_seq=max_seq),
            preprocessor,
            max_batch_size=max_batch_size,
            sort_by_tokens=sort_by_tokens,
        )
    else:
        return [
            convert_batch_element_to_tensors(elem, preprocessor)
            for elem in doc.to_batch_elements(
                preprocessor=preprocessor, override_max_seq=max_seq
            )
        ]

def collate_batch_elements_tns(
        batch_elements_tns: List[BatchElementTns], token_pad_value: int, ner_pad_value: int = IGNORE_INDEX
) -> BatchedElementsTns:
    """
    Collate (pad, correct index mask, stack) a list of tuples of tensors produced from BatchElement.to_tensors().
    Pads tensors to longest batch element: seq_len for sequence tensors, num_ent for ent tensors, num_spans for
    entity span mask tensor.
    :param batch_elements_tns: list of tensors produced from a list of BatchElements
    :return: tuple of tensors formed by collating list of items tensors (one element is a list of batch_elements obj)
    :param token_pad_value: token pad value
    :param ner_pad_value: ner pad value
    """
    batch_size = len(batch_elements_tns)
    b_token_ln = max(t.token_id_values.size(0) for t in batch_elements_tns)

    b_token_id_values = torch.zeros(
        (
            batch_size,
            b_token_ln,
        ),
        dtype=torch.long,
    )
    b_token_id_values.fill_(token_pad_value)
    b_token_type_values = torch.zeros(
        (
            batch_size,
            b_token_ln,
        ),
        dtype=torch.long,
    )
    b_attention_mask_values = torch.zeros(
        (
            batch_size,
            b_token_ln,
        ),
        dtype=torch.long,
    )

    for elem_idx, elem in enumerate(batch_elements_tns):
        token_ln = elem.token_id_values.size(0)
        b_token_id_values[elem_idx, :token_ln] = elem.token_id_values
        b_token_type_values[elem_idx, :token_ln] = elem.token_type_values
        b_attention_mask_values[elem_idx, :token_ln] = elem.attention_mask_values

    if batch_elements_tns[0].batch_element.spans is None:
        return BatchedElementsTns(
            token_id_values=b_token_id_values,
            token_type_values=b_token_type_values,
            attention_mask_values=b_attention_mask_values,
            batch_elements=[x.batch_element for x in batch_elements_tns],
        )
    # TODO modularise this code

    max_candidates = max(t.pem_values.size(1) for t in batch_elements_tns)

    b_num_classes = 0
    b_num_ents = 0
    b_entity_mask_ln = 0

    for elem in batch_elements_tns:
        num_ents = elem.candidate_class_values.size(0)
        num_classes = elem.candidate_class_values.size(2)
        entity_mask_ln = elem.entity_mask_values.size(0)
        b_num_ents = num_ents if num_ents > b_num_ents else b_num_ents
        b_entity_mask_ln = entity_mask_ln if entity_mask_ln > b_entity_mask_ln else b_entity_mask_ln
        b_num_classes = num_classes if num_classes > b_num_classes else b_num_classes

    if batch_size > 0 and batch_elements_tns[0].class_target_values is None:
        b_class_target_values = None
    else:
        b_class_target_values = torch.zeros(
            (batch_size, b_num_ents, b_num_classes), dtype=torch.long
        )

    if batch_size > 0 and batch_elements_tns[0].candidate_target_values is None:
        b_candidate_target_values = None
    else:
        b_candidate_target_values = torch.zeros(
            (batch_size, b_num_ents, max_candidates + 1), dtype=torch.long
        )

    if batch_size > 0 and batch_elements_tns[0].gold_qcode_values is None:
        b_gold_qcode_values = None
    else:
        b_gold_qcode_values = torch.zeros(
            (
                batch_size,
                b_num_ents,
            ),
            dtype=torch.long,
        )

    b_pem_values = torch.zeros((batch_size, b_num_ents, max_candidates), dtype=torch.float32)
    b_candidate_class_values = torch.zeros(
        (batch_size, b_num_ents, max_candidates, b_num_classes), dtype=torch.long
    )
    b_candidate_qcode_values = torch.zeros(
        (batch_size, b_num_ents, max_candidates), dtype=torch.long
    )

    b_candidate_desc = None
    b_candidate_desc_emb = None

    if batch_elements_tns[0].candidate_desc is not None:
        b_candidate_desc = pad_sequence(
            [elem.candidate_desc for elem in batch_elements_tns], batch_first=True
        )

    if batch_elements_tns[0].candidate_desc_emb is not None:
        b_candidate_desc_emb = pad_sequence(
            [elem.candidate_desc_emb for elem in batch_elements_tns], batch_first=True
        )

    b_entity_index_mask_values = torch.zeros(
        (
            batch_size,
            b_num_ents,
        ),
        dtype=torch.bool,
    )
    b_token_acc_sum_values = torch.zeros(
        (
            batch_size,
            b_token_ln,
        ),
        dtype=torch.long,
    )
    b_entity_mask_values = torch.zeros((batch_size, b_entity_mask_ln), dtype=torch.long)
    b_ner_labels = torch.zeros(
        (
            batch_size,
            b_token_ln,
        ),
        dtype=torch.long,
    )
    b_ner_labels.fill_(ner_pad_value)

    for item_idx, batch in enumerate(batch_elements_tns):
        token_ln = batch.token_id_values.size(0)
        entity_mask_ln = batch.entity_mask_values.size(0)
        num_ents = batch.candidate_class_values.size(0)
        b_entity_index_mask_values[item_idx, :num_ents] = True

        b_token_acc_sum_values[item_idx, :token_ln] = batch.token_acc_sum_values
        b_token_id_values[item_idx, :token_ln] = batch.token_id_values
        b_ner_labels[item_idx, :token_ln] = batch.ner_labels
        b_entity_mask_values[item_idx, :entity_mask_ln] = batch.entity_mask_values
        b_token_type_values[item_idx, :token_ln] = batch.token_type_values
        b_attention_mask_values[item_idx, :token_ln] = batch.attention_mask_values

        b_pem_values[item_idx, :num_ents] = batch.pem_values
        b_candidate_class_values[item_idx, :num_ents] = batch.candidate_class_values

        if b_class_target_values is not None and batch.class_target_values is not None:
            b_class_target_values[item_idx, :num_ents] = batch.class_target_values

        if b_candidate_target_values is not None and batch.candidate_target_values is not None:
            b_candidate_target_values[item_idx, :num_ents] = batch.candidate_target_values

        if b_gold_qcode_values is not None and batch.gold_qcode_values is not None:
            b_gold_qcode_values[item_idx, :num_ents] = batch.gold_qcode_values

        if batch.candidate_qcode_values is not None:
            b_candidate_qcode_values[item_idx, :num_ents] = batch.candidate_qcode_values

    # unpack example b_pem_values[b_entity_index_mask_values]
    # (bs, num_ents (includes padding as not all docs have same num_ents), num_cands) -> (all_ents, num_cands)
    return BatchedElementsTns(
        b_token_id_values,
        b_token_acc_sum_values,
        b_entity_mask_values,
        b_class_target_values,
        b_attention_mask_values,
        b_token_type_values,
        b_candidate_target_values,
        b_pem_values,
        b_candidate_class_values,
        b_entity_index_mask_values,
        [x.batch_element for x in batch_elements_tns],
        b_gold_qcode_values,
        b_candidate_qcode_values,
        b_ner_labels,
        candidate_desc=b_candidate_desc,
        candidate_desc_emb=b_candidate_desc_emb
    )


def pad(inputs: List[List[Any]], seq_len: int, pad_value: Any = 0, post_pad: bool = True):
    """
    Pad a list of lists to a list of equal length lists.
    :param inputs: list of lists
    :param seq_len: seq len to pad to. -1 means take the longer seq length.
    :param pad_value: the value to pad with
    :param post_pad: true means pad to the right of the seq, false means left.
    :return: a padded list of lists
    """
    if seq_len == -1:
        seq_len = max(len(row) for row in inputs)
    assert all(
        len(row) <= seq_len for row in inputs
    ), "`seq len` should exceed the max row length to avoid truncation."
    result = []
    for row in inputs:
        if post_pad:
            result.append(row + [pad_value] * (seq_len - len(row)))
        else:
            result.append([pad_value] * (seq_len - len(row)) + row)
    return result
