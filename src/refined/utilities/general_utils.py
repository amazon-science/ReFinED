import logging
import os
import subprocess
import sys
from typing import Any, Iterable, List, Optional

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from refined.data_types.base_types import Span


def split_interval(start: int, end: int, num_splits: int = 1) -> List[List[int]]:
    """
    Split an interval into sub intervals. Start of interval is inclusive. End of interval is exclusive.
    If interval cannot be evenly split then the intervals are increased in size by 1 from right-to-left.
    If number of splits is larger than interval some splits zeroed i.e. have values [0, 0].
    Examples:
    split_interval(0, 3, 1) -> [[0, 1, 2]]
    split_interval(0, 3, 2) -> [[0, 1], [1, 3]]
    split_interval(0, 0, 2) -> [[0, 0], [0, 0]]
    :param start: main interval start
    :param end: main interval end
    :param num_splits: number of intervals to split main interval into
    :return: a list of intervals [[start, end],...] (start is inclusive, end is exclusive)
    """
    sub_intervals = [[0, 0] for _ in range(num_splits)]
    if start >= end:
        return sub_intervals
    interval_length = end - start
    sub_interval_length = interval_length // num_splits
    curr_idx = start
    for sub_interval in sub_intervals:
        sub_interval[0] = curr_idx
        curr_idx += sub_interval_length
        sub_interval[1] = curr_idx

    interval_delta = interval_length - curr_idx + start
    if interval_delta == 0:
        return sub_intervals

    # stretch last interval (ensuring to shift so interval only increases by 1)
    sub_intervals[-1][1] += interval_delta
    sub_intervals[-1][0] += interval_delta - 1

    # stretch sub intervals by 1 from right to left
    for sub_interval_idx in reversed(range(num_splits)):
        if sub_interval_idx == num_splits - 1:
            continue
        gap_length = sub_intervals[sub_interval_idx + 1][0] - sub_intervals[sub_interval_idx][1]
        if gap_length == 0:
            break
        sub_intervals[sub_interval_idx][1] = sub_intervals[sub_interval_idx][1] + gap_length
        sub_intervals[sub_interval_idx][0] = sub_intervals[sub_interval_idx][0] + gap_length - 1

    return sub_intervals


def unique(func, iterable):
    seen_item_keys = set()
    for item in iterable:
        item_key = func(item)
        if item_key not in seen_item_keys:
            yield item
            seen_item_keys.add(item_key)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def get_logger(name: str = "main"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def batch_items(iterable: Iterable[Any], n: int = 1):
    """
    Batches an iterables by yielding lists of length n. Final batch length may be less than n.
    :param iterable: any iterable
    :param n: batch size (final batch may be truncated)
    """
    current_batch = []
    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == n:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch


def round_list(lst, n=4):
    return [round(float(element), n) for element in lst]


def get_tokenizer(
    transformer_name: str,
    data_dir: Optional[str] = None,
    use_fast: bool = True,
    add_special_tokens: bool = False,
    add_prefix_space: bool = False,
) -> PreTrainedTokenizerFast:
    """
    Get `transformers` library tokenizer. If data directory is passed then look for a
    folder with the `transformer_name` to and read the tokenizer binary file. Otherwise,
    download the tokenizer using the Huggingface API.
    :param transformer_name: transformer name (e.g. roberta-base)
    :param data_dir: path to data directory (which contains folders with huggingface resources)
    :param use_fast: use `PreTrainedTokenizerFast`.
    :param add_prefix_space: Huggingface arg.
    :param add_special_tokens: Huggingface arg.
    """
    if data_dir is not None and os.path.exists(os.path.join(data_dir, transformer_name)):
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(data_dir, transformer_name),
            use_fast=use_fast,
            add_special_tokens=add_special_tokens,
            add_prefix_space=add_prefix_space
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            transformer_name,
            use_fast=use_fast,
            add_special_tokens=add_special_tokens,
            add_prefix_space=add_prefix_space,
        )

    return tokenizer


def correct_spans(spans: List[Span]) -> None:
    """
    Applies some minor corrections to a list of spans in-place.
    An example is removing newline characters from start and end of end.
    :param spans: a list of spans to modify in place
    """
    for span in spans:
        if (
                len(span.text) == 1
                or span.text == "\n\n"
                or span.text == "\n\n\n"
                or span.text == "\n\n\n\n"
                or span.text == "the"
        ):
            spans.remove(span)
        elif len(span.text) > 2:
            # check for double \n or \n\n to at start of string
            if span.text[0] == "\n":
                span.text = span.text[1:]
                span.start += 1
                span.ln -= 1
            if span.text[0] == "\n":
                span.text = span.text[1:]
                span.start += 1
                span.ln -= 1

            # check for double \n or \n\n to at end of string
            if span.text[-1] == "\n":
                span.text = span.text[:-1]
                span.ln -= 1
            if span.text[-1] == "\n":
                span.text = span.text[:-1]
                span.ln -= 1

            # check for unbalanced "
            # start "
            if span.text[0] == '"' and not span.text[-1] == '"':
                span.text = span.text[1:]
                span.start += 1
                span.ln -= 1
            # end "
            if span.text[-1] == '"' and not span.text[0] == '"':
                span.text = span.text[:-1]
                span.ln -= 1

            # filter individual "the"
            if span.text == "the":
                spans.remove(span)
                continue

            # fix title and first mention problem
            if span.start == 0 and "\n\n" in span.text and len(span.text.split("\n\n")) == 2:
                first_span_text, second_span_text = span.text.split("\n\n")
                first_span = Span(start=0, ln=len(first_span_text), text=first_span_text, coarse_type="MENTION")
                second_span = Span(
                    start=span.text.find("\n\n") + 2,
                    ln=len(second_span_text),
                    text=second_span_text,
                    coarse_type="MENTION"
                )
                spans.remove(span)
                spans.append(first_span)
                spans.append(second_span)

            # TODO make sure not only part of words from word pieces e.g. ...erman... instead of ...German...
            # requires text before and after


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


def wc(filename: str) -> int:
    return int(subprocess.check_output(["wc", "-l", filename]).split()[0])
