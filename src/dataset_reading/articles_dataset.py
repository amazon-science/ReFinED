import itertools
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from doc_preprocessing.dataclasses import BatchElement, Doc, Span
from doc_preprocessing.doc_preprocessor import DocumentPreprocessorMemoryBased
from doc_preprocessing.preprocessing_utils import (
    collate_batch_elements_tns,
    convert_batch_element_to_tensors,
)
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info
from utilities.general_utils import batch_items


class ArticlesDataset(IterableDataset):
    """
    This is a dataset for pre-processing of an iterator of articles.
    It is a torch iterable dataset because it is too large to hold in memory.
    The iterator returns batches of elements instead of individual elements so no further collation is required.
    Similar length documents are batched together to improve efficiency.
    """

    def __init__(
        self,
        articles: Iterator[Tuple[str, List[Span]]],
        doc_preprocessor: DocumentPreprocessorMemoryBased,
        articles_in_memory: int = 1000,
        num_classes: int = 1308,
        max_num_classes: int = 150,
        max_candidates: int = 30,
        ner_threshold: float = 0.5,
        return_candidates: bool = True,
        return_candidates_scores: bool = True,
        return_ner_types: bool = True,
        max_batch_size: int = 32,
        filter_types: Optional[Sequence[str]] = ("DATE", "CARDINAL", "ORDINAL", "QUANTITY", "TIME"),
    ):
        """
        :param articles: iterator which yields a tuple of format (article_text, article_spans). article_text is the
                         text of the article, and article_spans is a list of Span objects
        :param doc_preprocessor: instance of DocumentPreprocessorMemoryBased to handle lookup of candidate entities, PEM
                                 scores etc.
        :param articles_in_memory: how many articles each instance of this dataset should read into memory at once
        :param num_classes: number of classes/types the model uses for NER types
        :param max_num_classes: max number of classes an entity can have
        :param max_candidates: maximum number of candidate entities to select (top k based on P(e|m))
        :param ner_threshold: threshold for which NER types are added to spans (does not affect model predictions)
        :param return_candidates: determines whether considered candidates are added to the spans
        :param return_candidates_scores: determines whether ranked candidates are added to the spans
        :param return_ner_types: determines whether NER types are added to the spans
        :param max_batch_size: max batch size
        :param filter_types: if provided, the model will not attempt to resolved mentions which have these NER types
                             to entities. Excluding types (such as numbers) which do not have Wikipedia pages can
                             speed up processing.
        """
        super(ArticlesDataset, self).__init__()

        self.articles: Iterator[Tuple[str, List[Span]]] = articles
        self.articles_in_memory: int = articles_in_memory

        self.doc_preprocessor: DocumentPreprocessorMemoryBased = doc_preprocessor

        self.num_classes: int = num_classes
        self.max_num_classes: int = max_num_classes
        self.max_candidates: int = max_candidates
        self.ner_threshold: float = ner_threshold
        self.return_candidates: bool = return_candidates
        self.return_candidates_scores: bool = return_candidates_scores
        self.return_ner_types: bool = return_ner_types
        self.max_batch_size: int = max_batch_size
        self.filter_types: Sequence[str] = filter_types

    @staticmethod
    def get_worker_info() -> Tuple[int, int]:
        """
        Determines which process this instance of the dataset is in, to allow splitting of articles between processes
        :return: the integer id of the worker, and the total number of workers
        """
        worker_info = get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        return worker_id, num_workers

    def group_articles(self) -> Iterator[List[Tuple[str, List[Span]]]]:
        """
        Reads articles from the iterator, and yields batches of articles of len self.articles_in_memory at a time
        :return: a list of tuples of format (article_text, article_spans)
        """
        worker_id, num_workers = self.get_worker_info()

        batch_articles = []

        for ix, article in enumerate(self.articles):

            if ix % num_workers == worker_id:

                batch_articles.append(article)

                if len(batch_articles) == self.articles_in_memory:
                    yield batch_articles
                    batch_articles = []

        if len(batch_articles) > 0:
            yield batch_articles

    @staticmethod
    def try_and_create_batch(
        doc_batches: List[Tuple[Doc, List[BatchElement]]],
        max_batch_size: int,
        cutoff_ratio: float = 0.6,
    ):
        """
        Attempts to group small articles together into a single batch. If a batch can be created which is larger
        than max_batch_size*cutoff_ratio, create and return it. Otherwise keep the batch_elements in memory
        and wait for more articles
        :param :doc_batches list of Docs with their corresponding batch elements
        :param :max_batch_size maximum batch size
        :param :cutoff_ratio max_batch_size*cutoff_ratio defines the minimum batch size which we will return
        :return: docs in the batch, batch_elements in the batch, and the remaining (doc, batch_elements) tuples which
                 have not been assigned to a batch yet
        """
        cutoff = int(max_batch_size * cutoff_ratio)

        # Sort by number of batch elements
        doc_batches = sorted(doc_batches, key=lambda b: len(b[1]))

        # Start batch off with shortest doc
        batch = [doc_batches[0]]

        # All other docs - will try to add these into a batch with the shortest doc
        remaining = doc_batches[1:]

        total_batch_len = len(batch[0][1])
        unassigned = []

        while len(remaining) > 0:

            to_add = remaining.pop()

            new_len = len(to_add[1]) + total_batch_len

            if new_len > max_batch_size:
                unassigned.append(to_add)
            else:
                batch.append(to_add)
                total_batch_len += len(to_add[1])

        if total_batch_len > cutoff:
            docs, batch_items = list(zip(*batch))
            batch_items = list(itertools.chain.from_iterable(batch_items))
            return docs, batch_items, unassigned
        else:
            return None, None, doc_batches

    def process_text_batch(self, articles: List[Tuple[str, List[Span]]]):
        """
        Batching of articles. Leaves long articles in single (or multiple) batches, and batches shorter articles
        together
        :param articles: a list of tuples of format (article_text, article_spans)
        :return: list of docs and associated batch elements
        """

        to_batch = []

        for article in articles:

            text = article["text"]
            spans = article["spans"] if "spans" in article else None
            metadata = article["metadata"] if "metadata" in article else None

            doc_batch_elements, doc = self.text_to_batch_elements(text, spans, metadata)

            # Empty article - doc will be returned without running model
            if len(doc.text) == 0:
                yield [doc], None
                continue

            # No spans - doc will be returned without running model
            if len(doc.spans) == 0:
                yield [doc], None
                continue

            # Remove batch elements that have no spans
            doc_batch_elements = list(filter(lambda x: len(x.spans) > 0, doc_batch_elements))

            batched_elements: List[List[BatchElement]] = list(
                batch_items(doc_batch_elements, n=self.max_batch_size)
            )

            # No batch elements remaining after filtering
            if len(batched_elements) == 0:
                yield [doc], None
                continue

            # Long article which fills multiple batches by itself - return individually to keep all batch elements
            # together with the doc
            elif len(batched_elements) > 1:
                tensors = [self.to_tensors(b) for b in batched_elements]
                yield [doc], tensors
                continue

            else:
                # Short article - try and batch together with other short articles
                to_batch.append((doc, batched_elements[0]))
                docs, batch, to_batch = self.try_and_create_batch(to_batch, self.max_batch_size)

                if docs is not None:
                    tns = self.to_tensors(batch)
                    yield docs, [tns]

        if len(to_batch) > 0:
            for doc, batch in to_batch:
                yield [doc], [self.to_tensors(batch)]

    def to_tensors(self, batch_elements: List[BatchElement]):
        """
        Converts batch element to tuple of tensors.
        :param batch_elements: the batch element to convert
        :return: tuple of tensors for the batch element
        """
        elems = []
        for elem_idx, elem in enumerate(batch_elements):
            elems.append(convert_batch_element_to_tensors(elem, processor=self.doc_preprocessor))
        batch_tns = collate_batch_elements_tns(elems, token_pad_value=self.doc_preprocessor.pad_id)
        if all(span is not None for elem in batch_elements for span in elem.spans):
            batch_spans = [span for elem in batch_elements for span in elem.spans]
        else:
            batch_spans = None

        return batch_tns, batch_spans

    def text_to_batch_elements(
        self, text: str, spans: List[Span] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[BatchElement], Doc]:
        """
        Convert text and optionally spans to a list of batch element (format that can be converted to tensors (1-to-1)) and
        return the Doc to represent the full document with the given spans or identified spans (if no spans are provided).
        :param text: text for a document/article, can have arbitrary length
        :param spans: spans of interest (for NER and ED predictions)
        :param metadata: metadata associated with the article
        :return: list of batch element (format that can be converted to tensors (1-to-1)) and
        return the Doc to represent the full document
        """
        if spans is not None:
            doc = Doc.from_text_with_spans(text, spans, self.doc_preprocessor, metadata=metadata)
        else:
            doc = Doc.from_text(
                text,
                transformer_name=self.doc_preprocessor.transformer_name,
                data_dir=self.doc_preprocessor.data_dir,
                metadata=metadata,
            )

        return doc.to_batch_elements(self.doc_preprocessor.data_dir), doc

    def __iter__(self) -> Tuple[Tensor]:
        """
        Returns a batch in correct format for ReFined model, and the associated docs (articles) which this batch
        contains text from
        :return: list of docs and list of corresponding batch_elements to input to the model
        """
        # Returns articles_in_memory articles/spans at a time
        article_batches = self.group_articles()

        for articles in article_batches:

            batches = self.process_text_batch(articles)

            for docs, batch_elements in batches:
                yield docs, batch_elements
