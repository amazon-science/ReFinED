import json
import random
from typing import List, Optional, Set, Iterator

import torch
from dataset_reading.dataset_utils import mask_mentions
from doc_preprocessing.dataclasses import BatchedElementsTns, Doc, Span, merge_spans
from doc_preprocessing.doc_preprocessor import DocumentPreprocessorMemoryBased
from doc_preprocessing.preprocessing_utils import (
    convert_batch_elements_to_batched_tns,
    correct_spans,
)
from utilities.general_utils import split_interval


class WikipediaDataset(torch.utils.data.IterableDataset):
    """
    This is a dataset reader for Wikipedia links dataset.
    It is a torch iterable dataset because it is too large to hold in memory.
    The dataset is split across >5000 files to enable the work to be divided among workers processes.
    The iterator returns batches of elements instead of individual elements so no further collation is required.
    Similar length documents are batched together to improve efficiency.
    Uses the deprecated class TrainingItem but should use BatchElement instead.
    """

    def __init__(
        self,
        preprocessor: DocumentPreprocessorMemoryBased,
        start: int = 0,
        end: int = 10,
        seed=0,
        num_workers: int = 1,
        batch_size: int = 16,
        mask: float = 0.0,
        random_mask: float = 0.0,
        dataset_path: str = "",
        prefetch: int = 100,
        lower_case_prob: float = 0.05,
        candidate_dropout: float = 0.0,
        max_mentions: Optional[int] = None,
        sample_k_candidates: Optional[int] = None,
        add_main_entity: bool = True,
        qcodes_to_filter: Optional[Set[str]] = None,
    ):
        """
        Constructs instance of Wikipedia dataset (should be used in conjunction with torch Dataloader).
        :param preprocessor: preprocessor provides methods Wikidata lookups and other useful preprocessing steps
        :param start: the start number in the range of file numbers to read
        :param end: the end number in the range of file numbers to read
        :param seed: seed used for reproducibility of shuffling
        :param num_workers: number of worker processes to use (a copy of the instance will be made by each worker so
         memory issues will be caused if there are too many workers. Must match or <= torch dataloader workers if used.
        :param batch_size: number of batch elements to collate (note that some batches may be smaller than this)
        :param mask: determines whether to mask entity mentions with a probability
        :param dataset_path: path the dataset directory
        :param prefetch: number of dataset lines to read in memory per worker at a time before batching
        :param candidate_dropout: candidate dropout
        :param max_mentions: max_mentions per batch (saves memory)
        :param sample_k_candidates: sample_k_candidates sample k candidates from top 30
        :param add_main_entity: add main entity by matching title and mentions

        """
        random.seed(seed)
        self.qcodes_to_filter = qcodes_to_filter if qcodes_to_filter is not None else set()
        self.add_main_entity = add_main_entity
        self.lower_case_prob = lower_case_prob
        self.worker_intervals = split_interval(start=start, end=end, num_splits=max(num_workers, 1))
        # each worker will work on an interval of the lines (e.g. worker 1: 0-10, worker 2: 11-20)
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.mask = mask
        self.random_mask = random_mask
        self.dataset_path = dataset_path
        self.prefetch = prefetch
        self.candidate_dropout = candidate_dropout
        self.max_mentions = max_mentions
        self.sample_k_candidates = sample_k_candidates

    def __iter__(self) -> Iterator[BatchedElementsTns]:
        """
        Iterates over the batch elements in each file.
        Each worker has a different set of files to process.
        Files contain ~ 1000 batch elements.
        :return: tuple of tensors for a batch of batch elements
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_num = 0
        # get relevant files for worker
        if worker_info is not None:
            worker_num = worker_info.id
        file_line_start, file_line_end = self.worker_intervals[worker_num]
        if file_line_end <= file_line_start:
            # worker has no lines of the file to process
            return

        with open(self.dataset_path, "r") as dataset_file:
            batch_elements = []
            for current_line_num, dataset_line in enumerate(dataset_file):
                if current_line_num >= file_line_end:
                    break
                if file_line_start <= current_line_num < file_line_end:
                    # when within workers allotted range
                    # read n (prefetch) lines then sort into efficient batches
                    # text and spans to Doc then batch elements
                    parsed_line = json.loads(dataset_line)
                    ents = parsed_line["hyperlinks_clean"]
                    text = parsed_line["text"]
                    spans: List[Span] = []
                    for ent in ents:
                        qcode = ent["qcode"]
                        if qcode in self.qcodes_to_filter:
                            continue
                        ln = ent["end"] - ent["start"]
                        start_idx = ent["start"]
                        spans.append(
                            Span(
                                start=start_idx,
                                ln=ln,
                                text=text[start_idx : start_idx + ln],
                                gold_entity_id=qcode,
                                coarse_type="MENTION"
                            )
                        )

                    if "predicted_spans" in parsed_line:
                        md_spans = [
                            Span(start=start, ln=end - start, text=text, coarse_type=coarse_type)
                            for start, end, text, coarse_type in parsed_line["predicted_spans"]
                        ]
                        correct_spans(md_spans)
                        md_spans.sort(key=lambda x: x.start, reverse=False)
                    else:
                        md_spans = None

                    # add main entities
                    if self.add_main_entity:
                        spans = self.merge_in_main_entity_mentions(
                            title=parsed_line["title"], spans=spans, md_spans=md_spans
                        )

                    doc_batch_elements = Doc.from_text_with_spans(
                        text,
                        spans,
                        self.preprocessor,
                        lower_case_prob=self.lower_case_prob,
                        md_spans=md_spans,
                        candidate_dropout=self.candidate_dropout,
                        sample_k_candidates=self.sample_k_candidates,
                    ).to_batch_elements(
                        max_mentions=self.max_mentions, data_dir=self.preprocessor.data_dir
                    )
                    if self.mask > 0.0:
                        doc_batch_elements = map(
                            lambda x: mask_mentions(
                                x,
                                mask_prob=self.mask,
                                random_word_prob=self.random_mask,
                                mask_token_id=self.preprocessor.mask_id,
                                vocab_size=self.preprocessor.vocab_size,
                            ),
                            doc_batch_elements,
                        )
                    batch_elements.extend(doc_batch_elements)
                    del parsed_line, ents, text
                    if len(batch_elements) > self.prefetch:
                        for batch_tns in convert_batch_elements_to_batched_tns(
                            batch_elements,
                            self.preprocessor,
                            max_batch_size=self.batch_size,
                            sort_by_tokens=False,
                        ):
                            yield batch_tns
                            del batch_tns
                        del batch_elements, doc_batch_elements
                        batch_elements = []

            if len(batch_elements) > 0:
                for batch_tns in convert_batch_elements_to_batched_tns(
                    batch_elements,
                    self.preprocessor,
                    max_batch_size=self.batch_size,
                    sort_by_tokens=False,
                ):
                    yield batch_tns

    def __len__(self):
        # approximation for training split
        return 11906136 // self.batch_size

    def merge_in_main_entity_mentions(
        self, title: str, spans: List[Span], md_spans: List[Span]
    ) -> List[Span]:
        """
        Merges main entity mentions into spans (hyperlinks_clean).
        :param title: Wikipedia title.
        :param spans: spans (hyperlinks) with entity labels.
        :param md_spans: spans (span (mention) detection) without entity labels.
        :return: spans with some (maybe) md_spans merged in with entity label set as main entity id (qcode).
        """
        title = title.replace(" ", "_")
        qcode = self.preprocessor.map_title_to_qcode(title)

        if qcode is None or qcode in self.qcodes_to_filter:
            # nothing to merge in because title cannot be mapped to entity id (qcode)
            # do not include filtered qcodes
            return spans

        main_label = self.preprocessor.qcode_to_label[qcode].replace("'s", "")
        if qcode in self.preprocessor.human_qcodes:
            # add surname, first name, middle name (e.g. "Joe", "Biden" for "Joe Biden")
            labels = set(main_label.split(" ") + [main_label])
        else:
            labels = {main_label}

        # correct spans identified by span detection (MD) model by applying simple rules to remove common mistakes
        correct_spans(md_spans)
        md_spans.sort(key=lambda x: x.start, reverse=False)
        if len(md_spans) > 0 and md_spans[0].start == 0:
            # assume articles that start with an entity start with the title entity
            main_spans = [md_spans[0]]
        else:
            main_spans = []

        main_spans.extend(
            [s for s in md_spans if s.text.replace("'s", "") in labels and s.start != 0]
        )

        # add title entity id to the spans matching the title (main entity)
        for s in main_spans:
            s.gold_entity_id = qcode

        # merge main spans with existing spans (from hyperlinks) if they overlap prefer hyperlinks
        final_spans = merge_spans(additional_spans=main_spans, prioritised_spans=spans)

        return final_spans
