import math
import random
from typing import Iterable, List

import torch
from dataset_reading.dataset_utils import mask_mentions
from doc_preprocessing.dataclasses import BatchedElementsTns, BatchElement, BatchElementTns, Doc
from doc_preprocessing.doc_preprocessor import DocumentPreprocessorMemoryBased
from doc_preprocessing.preprocessing_utils import (
    collate_batch_elements_tns,
    convert_batch_element_to_tensors,
)
from torch.utils.data.dataset import Dataset, IterableDataset


class DocDataset(Dataset):
    """
    This is a dataset reader for CoNLL-AIDA dataset.
    For consistency, the files used were generated from GERBIL benchmarking framework:
    https://github.com/dice-group/gerbil/blob/master/src/main/java/org/aksw/gerbil/dataset/impl/aida/AIDACoNLLDataset.java
    """

    def __init__(
        self,
        docs,
        preprocessor: DocumentPreprocessorMemoryBased,
        seed=0,
        max_seq: int = 510,
        mask: bool = False,
    ):
        """
        Constructs dataset object.
        :param preprocessor: utility class that provides methods for WikiData lookups and general pre-processing
        :param seed: for reproducibility of masking
        :param dataset_split: dataset split to use ('train', 'dev', 'test')
        :param data_dir: path directory containing the dataset files
        :param filter_disambiguation: determines whether to filter annotations that refer to a disambiguation page
        :param mask: determines whether to mask the entity mentions (may be useful for training)
        :param max_num_classes: maximum number of classes an entity can belong to
        """
        random.seed(seed)
        self.max_candidates = preprocessor.max_candidates
        self.num_classes = len(preprocessor.classes)
        self.preprocessor = preprocessor
        self.max_seq = max_seq
        self.mask = mask
        self.batch_elements: List[BatchElement] = []
        self.load_docs(docs=list(docs))

    def load_docs(self, docs: List[Doc]):
        for doc in docs:
            self.batch_elements.extend(
                [
                    b
                    for b in doc.to_batch_elements(
                        data_dir=self.preprocessor.data_dir, override_max_seq=self.max_seq
                    )
                    if len(b.spans) > 0
                ]
            )

    def __getitem__(self, index: int) -> BatchElementTns:
        """
        Returns the tensors for a single batch element.
        :param index: the index of the batch element to return
        :return: batch element converted to a tuple of tensors
        """
        if self.mask:
            batch_elem = mask_mentions(
                self.batch_elements[index],
                mask_token_id=self.preprocessor.mask_id,
                vocab_size=self.preprocessor.vocab_size,
            )
        else:
            batch_elem = self.batch_elements[index]
        return convert_batch_element_to_tensors(
            batch_element=batch_elem, processor=self.preprocessor
        )

    def __len__(self) -> int:
        """
        Returns number of batch elements.
        :return: Number of batch elements.
        """
        return len(self.batch_elements)

    def collate(self, batch_elements_tns: List[BatchElementTns]) -> BatchedElementsTns:
        """
        Collates a list of tensors for batch elements into a single batch.
        - Pads tensors to longest length in batch
        - Corrects masks to prevent information loss
        - Stacks tensors
        :param batch_elements_tns: list of tensors for batch elements
        :return: a tuple of tensors where all tensors in the list are vertically stacked
        """
        return collate_batch_elements_tns(
            batch_elements_tns=batch_elements_tns, token_pad_value=self.preprocessor.pad_id
        )


class DocIterDataset(IterableDataset):
    """
    This is a dataset reader for CoNLL-AIDA dataset.
    For consistency, the files used were generated from GERBIL benchmarking framework:
    https://github.com/dice-group/gerbil/blob/master/src/main/java/org/aksw/gerbil/dataset/impl/aida/AIDACoNLLDataset.java
    """

    def __init__(
        self,
        docs: Iterable[Doc],
        preprocessor: DocumentPreprocessorMemoryBased,
        seed=0,
        max_seq: int = 510,
        mask: bool = False,
        mask_prob: float = 0.80,
        random_mask_prob: float = 0.05,
    ):
        """
        Constructs dataset object.
        :param preprocessor: utility class that provides methods for WikiData lookups and general pre-processing
        :param seed: for reproducibility of masking
        :param dataset_split: dataset split to use ('train', 'dev', 'test')
        :param data_dir: path directory containing the dataset files
        :param filter_disambiguation: determines whether to filter annotations that refer to a disambiguation page
        :param mask: determines whether to mask the entity mentions (may be useful for training)
        :param max_num_classes: maximum number of classes an entity can belong to
        """
        random.seed(seed)
        self.mask_prob = mask_prob
        self.random_mask_prob = random_mask_prob
        self.docs = docs
        self.max_candidates = preprocessor.max_candidates
        self.num_classes = len(preprocessor.classes)
        self.preprocessor = preprocessor
        self.max_seq = max_seq
        self.mask = mask
        self.batch_elements: List[BatchElement] = []
        self.start = 0
        if isinstance(docs, list):
            self.end = len(docs)
        else:
            self.end = 2200000

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # single worker
            iter_start = self.start
            iter_end = self.end
        else:
            # multiple workers
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        return self.load_docs(start=iter_start, end=iter_end)

    def load_docs(self, start: int, end: int):
        for doc_idx, doc in enumerate(self.docs):
            if start <= doc_idx < end:
                batch_elements = [
                    b
                    for b in doc.to_batch_elements(
                        data_dir=self.preprocessor.data_dir, override_max_seq=self.max_seq
                    )
                    if len(b.spans) > 0
                ]
                for batch_element in batch_elements:
                    if self.mask:
                        batch_element = mask_mentions(
                            batch_element,
                            mask_token_id=self.preprocessor.mask_id,
                            vocab_size=self.preprocessor.vocab_size,
                            mask_prob=self.mask_prob,
                            random_word_prob=self.random_mask_prob,
                        )
                    yield convert_batch_element_to_tensors(
                        batch_element=batch_element, processor=self.preprocessor
                    )

    def __len__(self) -> int:
        """
        Returns number of batch elements.
        :return: Number of batch elements.
        """
        # this is the length of WikiLinksNED training
        return 2200000

    def collate(self, batch_elements_tns: List[BatchElementTns]) -> BatchedElementsTns:
        """
        Collates a list of tensors for batch elements into a single batch.
        - Pads tensors to longest length in batch
        - Corrects masks to prevent information loss
        - Stacks tensors
        :param batch_elements_tns: list of tensors for batch elements
        :return: a tuple of tensors where all tensors in the list are vertically stacked
        """
        return collate_batch_elements_tns(
            batch_elements_tns=batch_elements_tns, token_pad_value=self.preprocessor.pad_id
        )
