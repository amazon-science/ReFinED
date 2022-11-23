import random
from typing import List

from refined.utilities.md_dataset_utils import bio_to_offset_pairs
from refined.data_types.modelling_types import BatchElementToken, BatchElement


def mask_mentions(
    batch_element: BatchElement,
    mask_token_id: int,
    vocab_size: int,
    mask_prob=0.80,
    random_word_prob=0.05,
) -> BatchElement:
    """
    Masks entity mentions in the batch element a random percentage of the time with the [MASK] token and sometimes
    with a random token from the vocab [0, vocab_len - 1]. Probabilities are expressed [0, 1].
    This method masks whole mentions.
    :param batch_element: the item to mask the mention entities for
    :param mask_prob: the probability that a mention will be masked
    :param random_word_prob: the probability that a masked mention will be masked with a random token instead of [MASK].
    :param vocab_size: vocab size
    :param mask_token_id: mask token id
    :return: the item with some mentions masked.
    """
    # TODO: consider replacing with popularity-based masking

    # determines which mentions to mask
    # 0 means accumulated span is normal span (non-mention)
    # 1 means span will be masked with [MASK] token
    # 2 means span will be replaced with random word from vocab [0, vocab_len - 1]
    acc_sum_for_masking = [
        1 if x == 1 and random.random() < mask_prob else 0 for x in batch_element.entity_mask
    ]
    acc_sum_for_mask_or_random_word = [
        2 if x == 1 and random.random() < random_word_prob else x for x in acc_sum_for_masking
    ]
    masked_training_tokens: List[BatchElementToken] = []
    for t in batch_element.tokens:
        if acc_sum_for_mask_or_random_word[t.acc_sum] == 1:
            # will mask mention
            masked_training_tokens.append(
                BatchElementToken(t.acc_sum, t.text, mask_token_id, t.start, t.end)
            )
        elif acc_sum_for_mask_or_random_word[t.acc_sum] > 0:
            # will substitute mask with random token
            masked_training_tokens.append(
                BatchElementToken(
                    t.acc_sum, t.text, random.randint(0, vocab_size - 1), t.start, t.end
                )
            )
        else:
            # will leave token the same
            masked_training_tokens.append(
                BatchElementToken(t.acc_sum, t.text, t.token_id, t.start, t.end)
            )

    return BatchElement(
        tokens=masked_training_tokens,
        entity_mask=batch_element.entity_mask,
        spans=batch_element.spans,
        text=batch_element.text,
        md_spans=batch_element.md_spans,
        doc_id=batch_element.doc_id
    )


def gold_bio_to_spans(text, ners):
    words = text.split(" ")
    gold_spans_bio = bio_to_offset_pairs(ners)
    gold_spans = []
    offset_mappings = []
    start = 0
    for word in words:
        ln = len(word)
        offset_mappings.append((start, start + ln))
        start += ln + 1
    for start_idx, end_idx in gold_spans_bio:
        doc_offset_start = offset_mappings[start_idx][0]
        doc_offset_end = offset_mappings[end_idx - 1][1]
        gold_spans.append((doc_offset_start, doc_offset_end, text[doc_offset_start:doc_offset_end]))
    gold_spans.sort(key=lambda x: x[0])
    return gold_spans
