import logging
import os
import sys
from typing import Any, Iterable, List, Optional

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)


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
            add_prefix_space=False,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            transformer_name,
            use_fast=use_fast,
            add_special_tokens=add_special_tokens,
            add_prefix_space=add_prefix_space,
        )

    return tokenizer


def get_huggingface_model(
    transformer_name: str, data_dir: Optional[str] = None, **kwargs
) -> PreTrainedModel:
    """
    Get `transformers` library tokenizer. If data directory is passed then look for a
    folder with the `transformer_name` to and read the model file. Otherwise,
    download the model using the Huggingface API.
    :param transformer_name: transformer name (e.g. roberta-base)
    :param data_dir: path to data directory (which contains folders with huggingface resources)
    :param kwargs: additional kwargs to pass to AutoModel.from_pretrained(...).
    """
    if data_dir is not None and os.path.exists(os.path.join(data_dir, transformer_name)):
        model = AutoModel.from_pretrained(os.path.join(data_dir, transformer_name), **kwargs)
    else:
        model = AutoModel.from_pretrained(transformer_name, **kwargs)
    return model


def get_huggingface_config(transformer_name: str, data_dir: Optional[str] = None) -> Any:
    """
    Get `transformers` library model config. If data directory is passed then look for a
    folder with the `transformer_name` to and read the model config file. Otherwise,
    download the model config using the Huggingface API.
    :param transformer_name: transformer name (e.g. roberta-base)
    :param data_dir: path to data directory (which contains folders with huggingface resources)
    """
    if data_dir is not None and os.path.exists(os.path.join(data_dir, transformer_name)):
        cfg = AutoConfig.from_pretrained(os.path.join(data_dir, transformer_name))
    else:
        cfg = AutoConfig.from_pretrained(transformer_name)
    return cfg


wikidata_to_spacy_types = {
    "Q215627": "PERSON",  # person
    "Q5": "PERSON",  # person
    "Q41710": "NORP",  # ethnic group
    "Q4392985": "NORP",  # religious identity
    "Q844569": "NORP",  # identity
    "Q179805": "NORP",  # political philosophy
    "Q5891": "NORP",  # philosophy
    "Q1860557": "NORP",  # self-concept
    # "Q11862829": "NORP",  # academic discipline
    "Q13226383": "FAC",  # facility
    "Q41176": "FAC",  # building
    "Q35127": "ORG",  # website
    "Q4830453": "ORG",  # business
    "Q5341295": "ORG",  # educational organisation
    "Q15265344": "ORG",  # broadcaster
    "Q4438121": "ORG",  # sports organisation
    "Q45103187": "ORG",  # scientific organisation
    "Q7210356": "ORG",  # political organisation
    "Q1664720": "ORG",  # institute
    "Q2659904": "ORG",  # government agency
    "Q43229": "ORG",  # organization
    "Q15642541": "GPE",  # human-geographic territorial entity
    "Q82794": "GPE",  # geographical region
    "Q12076836": "GPE",  # area ins single country
    "Q35145263": "LOC",  # natural geographic object
    "Q1485500": "PRODUCT",  # tangible good
    "Q1656682": "EVENT",  # event
    "Q838948": "WORK_OF_ART",  # work of art
    "Q732577": "WORK_OF_ART",  # publication
    "Q34770": "LANGUAGE",  # language
    "Q820655": "LAW",  # legislative act
    # currently not explicitly supported
    "Q186408": "DATE",  # point in time
    # "Q186081": "TIME",  # time interval
    "Q11229": "PERCENT",  # percent
    "Q1368": "MONEY",  # money
    "Q47574": "QUANTITY",  # unit of measurement
    "Q923933": "ORDINAL",  # ordinal numeral
    "Q63116": "CARDINAL",  # numeral
}
