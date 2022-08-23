import json
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from dataset_reading.dataset_utils import mask_mentions
from doc_preprocessing.dataclasses import (
    BatchedElementsTns,
    BatchElement,
    BatchElementTns,
    Doc,
    Span,
    tokenize,
)
from doc_preprocessing.doc_preprocessor import DocumentPreprocessorMemoryBased
from doc_preprocessing.preprocessing_utils import (
    collate_batch_elements_tns,
    convert_batch_element_to_tensors,
)
from torch.utils.data.dataset import Dataset


class AidaPPRDataset(Dataset):
    """
    This is a dataset reader for CoNLL-AIDA dataset.
    For consistency, the files used were generated from GERBIL benchmarking framework:
    https://github.com/dice-group/gerbil/blob/master/src/main/java/org/aksw/gerbil/dataset/impl/aida/AIDACoNLLDataset.java
    """

    def __init__(
        self,
        preprocessor: DocumentPreprocessorMemoryBased,
        seed=0,
        dataset_split: str = "train",
        data_dir: str = "/big_data",
        filter_disambiguation: bool = True,
        mask: bool = False,
        max_num_classes: int = 150,
        max_seq: int = 510,
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
        assert dataset_split in ["train", "dev", "test"]
        random.seed(seed)
        split_to_filename = {
            "train": "aida_train_ppr.json",
            "dev": "aida_dev_ppr.json",
            "test": "aida_test_ppr.json",
        }
        file_path = f"{data_dir}/{split_to_filename[dataset_split]}"
        if filter_disambiguation:
            self.disambiguation_qcodes: Set[str] = preprocessor.disambiguation_qcodes
        else:
            self.disambiguation_qcodes: Set[str] = set()
        self.max_seq = max_seq
        self.max_candidates = preprocessor.max_candidates
        self.num_classes = len(preprocessor.classes)
        self.max_num_classes = max_num_classes
        self.preprocessor = preprocessor
        self.batch_elements: List[BatchElement] = []
        self._load_aida_file_contents(file_path)
        self.mask = mask

    def _load_aida_file_contents(self, filename: str):
        """
        Loads AIDA file into self.batch_elements list.
        :param filename: file path for AIDA dataset file
        """
        with open(filename, "r") as f:
            for idx, line in enumerate(f):
                parsed_line = json.loads(line)
                self.batch_elements.extend(
                    [
                        b
                        for b in self.from_ppr_dataset_line(parsed_line).to_batch_elements(
                            self.preprocessor.data_dir, override_max_seq=self.max_seq
                        )
                        if len(b.spans) > 0
                    ]
                )

    def from_ppr_span(
        self, ppr_span: Dict[Any, Any], person_coref: Dict
    ) -> Tuple[Optional[Span], Dict[str, Any]]:
        text = ppr_span["text"]
        start = ppr_span["start"]
        ln = ppr_span["end"] - ppr_span["start"]
        gold_qcode = self.preprocessor.map_title_to_qcode(ppr_span["gold_titles"][0])
        pruned_candidates = {
            self.preprocessor.map_title_to_qcode(c) for c in ppr_span["candidates"]
        }
        pruned_candidates -= {None}
        if gold_qcode is None or gold_qcode in self.preprocessor.disambiguation_qcodes:
            return None, person_coref
        candidate_qcodes, person_coref = self.preprocessor.get_candidates(
            surface_form=text, person_coref_ref=person_coref, pruned_candidates=pruned_candidates
        )
        return (
            Span(
                text=text,
                start=start,
                ln=ln,
                gold_entity_id=gold_qcode,
                candidate_entity_ids=candidate_qcodes,
                pruned_candidates=pruned_candidates,
            ),
            person_coref,
        )

    def from_ppr_dataset_line(self, ppr_dataset_line: Dict[str, Any]) -> "Doc":
        """
        Constructs `Doc` object from gerbil dataset line.
        e.g. code used in Gerbil benchmarking framework to read dataset:
        https://github.com/dice-group/gerbil/blob/master/src/main/java/org/aksw/gerbil/dataset/impl/aida/AIDACoNLLDataset.java
        :param ppr_dataset_line: python dict for output of a standard gerbil dataset reader
        :return: `Doc`, spans contains candidates and gold entities so can be used for training and evaluation
        """
        ppr_spans = ppr_dataset_line["spans"]
        ppr_spans.sort(key=lambda x: x["start"])
        doc_text = ppr_dataset_line["doc_text"]
        spans: List[Span] = []
        line_persons_coref: Dict = dict()
        md_spans: List[Span] = []

        for ppr_span in ppr_spans:
            # populate persons_coref. This enables backward person co-reference.
            _, line_persons_coref = self.from_ppr_span(
                ppr_span=ppr_span, person_coref=line_persons_coref
            )
        for ppr_span in ppr_spans:
            span, line_persons_coref = self.from_ppr_span(
                ppr_span=ppr_span, person_coref=line_persons_coref
            )
            text = ppr_span["text"]
            start = ppr_span["start"]
            ln = ppr_span["end"] - ppr_span["start"]
            md_spans.append(Span(start=start, ln=ln, text=text))
            if span is not None:
                spans.append(span)
        doc = Doc.from_text_with_spans(doc_text, spans, self.preprocessor, md_spans=md_spans)
        return doc

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