from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import torch
from transformers import PreTrainedTokenizer, PretrainedConfig, PreTrainedModel

from refined.doc_preprocessing.candidate_generator import CandidateGenerator, CandidateGeneratorExactMatch
from refined.doc_preprocessing.class_handler import ClassHandler
from refined.resource_management.data_lookups import LookupsInferenceOnly
from refined.data_types.base_types import Token, Span
from refined.resource_management.resource_manager import ResourceManager, get_mmap_shape
from refined.model_components.config import ModelConfig
from refined.resource_management.aws import S3Manager
from refined.utilities.general_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Preprocessor(ABC):
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
    tokenizer: PreTrainedTokenizer = field(init=False)
    transformer_config: PretrainedConfig = field(init=False)
    transformer_model: PreTrainedModel = field(init=False)
    max_seq: int = 510

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
    def get_candidates(
            self,
            surface_form: str,
            person_coref_ref: Dict[str, List[Tuple[str, float]]] = None,
            sample_k_candidates: Optional[int] = None
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        """
        Given a surface form (e.g. "Donald Trump") the method will return the top k (MAX_CANDIDATES) based on
        P(e|m) lookup dictionary. The method returns List[(qcodes, pem_value, pme_value)], person_coref.
        Person coref dictionary keeps track of humans mentioned in the document and copies the values from full name
        over partial name mentions (e.g. from "Donald Trump" to "Trump").
        :param surface_form: surface form to fetch candidates for
        :param person_coref_ref: a dictionary of previous human mentions with partial names in the dictionary
        :param sample_k_candidates: randomly sample candidates (hard, random, and gold)
        :return: List[(qcode, pem_value)], person_coref
        """
        pass

    @abstractmethod
    def add_candidates_to_spans(
            self,
            spans: List['Span'],
            backward_coref: bool = False,
            person_coreference: Optional[Dict[str, List[Tuple[str, float]]]] = None,
            sample_k_candidates: Optional[int] = None,
    ):
        """
        Adds candidate entities to each span.
        :param spans: list of spans
        :param backward_coref: if true will do 2 passes over spans (first pass collects human candidate entities,
                               and copies candidate over to partial mentions (e.g. surname instead of full name),
                               otherise will build person coreference dictionary sequentially so only forward coref
                               occurs.
        :param sample_k_candidates: randomly sample candidates (hard, random, and gold)
        :param gold_qcode: gold candidate only used during training when `sample_k_candidates` is provide
        :return: no return (modifies spans in-place)
        """
        pass

    @abstractmethod
    def split_sentences(self, text: str) -> List[Tuple[int, int]]:
        """
        Given a text, generates (start, end) spans of sentences
        in the text.
        @param text: input text
        @return: list of tuple of (start, end) offsets
        """
        pass

    @abstractmethod
    def tokenize(self, text: str):
        pass

    @abstractmethod
    def get_transformer_model(self):
        pass


class PreprocessorInferenceOnly(Preprocessor):
    """
    Implementation of Preprocessor that uses memmaped objects and only contains data needed for inference.
    See `Preprocessor` for additional documentation.
    """

    def __init__(
            self,
            transformer_name: str,
            entity_set: str,
            max_candidates: int,
            ner_tag_to_ix: Dict[str, int],
            data_dir: str,
            debug: bool = False,
            use_precomputed_description_embeddings: bool = True,
            model_description_embeddings_file: Optional[str] = None,
            inference_only: bool = True,
            return_titles: bool = False,
            max_seq: int = 510
    ):
        """
        Constructs PreprocessorInferenceOnly object.
        :param data_dir: path to directory containing the resources files
        :param max_candidates: max number of candidate entities to fetch
        :param debug: debug only loads a line for each file
        """
        if use_precomputed_description_embeddings:
            assert model_description_embeddings_file is not None, \
                "`model_description_embeddings_file` can not be None when" \
                "`use_precomputed_description_embeddings` is True."
        super().__init__(
            data_dir=data_dir, transformer_name=transformer_name,
            max_candidates=max_candidates,
            ner_tag_to_ix=ner_tag_to_ix
        )
        self.max_seq = max_seq
        self.use_precomputed_description_embeddings = use_precomputed_description_embeddings
        self.debug = debug
        self.transformer_name = transformer_name
        self.entity_set = entity_set
        self.inference_only = inference_only
        self.return_titles = return_titles
        self.max_candidates = max_candidates

        self.lookups = LookupsInferenceOnly(
            data_dir=data_dir,
            entity_set=entity_set,
            use_precomputed_description_embeddings=use_precomputed_description_embeddings,
            return_titles=return_titles
        )
        self.tokenizer = self.lookups.tokenizers
        self.qcode_to_idx = self.lookups.qcode_to_idx
        self.class_to_idx = self.lookups.class_to_idx
        self.class_to_label = self.lookups.class_to_label
        self.qcode_idx_to_class_idx = self.lookups.qcode_idx_to_class_idx
        self.index_to_class = self.lookups.index_to_class
        self.num_classes = len(self.class_to_idx)
        # `descriptions_tns` is None when use_precomputed_description_embeddings is True
        self.descriptions_tns = self.lookups.descriptions_tns
        self.qcode_to_wiki = self.lookups.qcode_to_wiki
        self.nltk_sentence_splitter_english = self.lookups.nltk_sentence_splitter_english
        self.transformer_config = self.lookups.transformer_model_config
        self.max_num_classes_per_ent = self.qcode_idx_to_class_idx.shape[1]
        self.class_handler = ClassHandler(subclasses=self.lookups.subclasses,
                                          qcode_to_idx=self.qcode_to_idx,
                                          index_to_class=self.index_to_class,
                                          qcode_idx_to_class_idx=self.qcode_idx_to_class_idx)
        self.candidate_generator: CandidateGenerator = CandidateGeneratorExactMatch(
            max_candidates=max_candidates,
            pem=self.lookups.pem,
            human_qcodes=self.lookups.human_qcodes
        )
        self.human_qcodes = self.lookups.human_qcodes

        if self.use_precomputed_description_embeddings:
            # (num_ents, description_embeddings_dim)
            self.precomputed_descriptions = np.memmap(
                model_description_embeddings_file,
                shape=get_mmap_shape(model_description_embeddings_file),
                dtype=np.float16,  # was float 32  #using float using 0.8387 from 0.8393 with 39s to 23s and half memory
                # check with GPU not being used may be different
                mode="r",
                order="C"
            )
            assert self.precomputed_descriptions[0].sum() == 0.0, "First row should be 0s to ensure proper masking of " \
                                                                  "descriptions for padded entities in description " \
                                                                  "encoder"
        else:
            self.precomputed_descriptions = None

        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.mask_id = self.tokenizer.mask_token_id
        self.vocab_size = self.tokenizer.vocab_size
        # non inference preprocessor can extend/reuse this and add additional functionality

    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenize text using tokenizer fast.
        :param text: text
        :return: list of tokens
        """
        if len(text) == 0:
            return []
        try:
            token_res = self.tokenizer.encode_plus(
                text,
                return_offsets_mapping=True,
                return_token_type_ids=False,
                return_attention_mask=False,
                add_special_tokens=False
            )
            word_pieces = self.tokenizer.convert_ids_to_tokens(
                token_res["input_ids"]
            )  # stores word pieces for debugging
            return [
                Token(word_piece, token_id, start, end)
                for word_piece, token_id, (start, end) in zip(
                    word_pieces, token_res["input_ids"], token_res["offset_mapping"]
                )
            ]
        except IndexError:
            logger.debug(f"Skipping article as failed to tokenize: {text}")
            return []

    def split_sentences(self, text: str):
        """
        Given a text, generates (start, end) spans of sentences
        in the text.
        @param text: input text
        @return: list of tuple of (start, end) offsets
        """
        return self.nltk_sentence_splitter_english.span_tokenize(text)

    def get_classes_idx_for_qcode_batch(
            self, qcodes: List[str], shape: Tuple[int, ...] = None
    ) -> torch.Tensor:
        """
        Retrieves all of the classes indices for the qcodes (from various relations used to construct the lookup).
        :param qcodes: qcodes
        :param shape: shape/size of tensor returned
        :return: long tensor with shape (num_ents, self.max_num_classes) (qcode index 0 is used for padding all classes)
        """
        return self.class_handler.get_classes_idx_for_qcode_batch(qcodes=qcodes, shape=shape)

    def get_descriptions_emb_for_qcode_batch(
            self, qcodes: List[str], shape: Tuple[int, ...] = None
    ) -> torch.Tensor:
        """
        Retrieves descriptions input_ids for a batch of qcodes and optionally reshapes tensor.
        :param qcodes: qcodes
        :param shape: shape/size of tensor returned
        :return: long tensor with shape (num_ents, self.max_num_classes) (qcode index 0 is used for padding all classes)
        """
        result = torch.from_numpy(
            self.precomputed_descriptions[
                [self.qcode_to_idx[qcode] if qcode in self.qcode_to_idx else 0 for qcode in qcodes]
            ]
        ).type(torch.float32)  # added casting to resolve float16 issue on CPU.
        # TODO: only cast to float32 when using CPU (note that this requires knowledge of the device being used).
        if shape is not None:
            return result.view(shape)
        else:
            return result

    def get_descriptions_for_qcode_batch(
            self, qcodes: List[str], shape: Tuple[int, ...] = None
    ) -> torch.Tensor:
        """
        Retrieves descriptions input_ids for a batch of qcodes and optionally reshapes tensor.
        :param qcodes: qcodes
        :param shape: shape/size of tensor returned
        :return: long tensor with shape (num_ents, self.max_num_classes) (qcode index 0 is used for padding all classes)
        """
        result = self.descriptions_tns[
            [self.qcode_to_idx[qcode] if qcode in self.qcode_to_idx else 0 for qcode in qcodes]
        ]
        if shape is not None:
            return result.view(shape)
        else:
            return result

    def get_candidates(self, *args, **kwargs) -> Any:
        return self.candidate_generator.get_candidates(*args, **kwargs)

    def add_candidates_to_spans(
            self,
            spans: List[Span],
            backward_coref: bool = False,
            person_coreference: Optional[Dict[str, List[Tuple[str, float]]]] = None,
            sample_k_candidates: Optional[int] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        return self.candidate_generator.add_candidates_to_spans(spans=spans,
                                                                backward_coref=backward_coref,
                                                                person_coreference=person_coreference,
                                                                sample_k_candidates=sample_k_candidates)

    @classmethod
    def from_model_config_file(
            cls,
            filename: str,
            entity_set: str,
            data_dir: str,
            debug: bool = False,
            download_files: bool = False,
            inference_only: bool = True,
            use_precomputed_description_embeddings: bool = True,
            model_description_embeddings_file: Optional[str] = None,
            return_titles: bool = False,
            max_candidates: Optional[int] = None
    ):

        if download_files:
            resource_manager = ResourceManager(s3_manager=S3Manager(),
                                               data_dir=data_dir,
                                               entity_set=entity_set,
                                               inference_ony=inference_only,
                                               load_qcode_to_title=return_titles,
                                               load_descriptions_tns=not use_precomputed_description_embeddings,
                                               model_name=None)
            resource_manager.download_data_if_needed()

        config = ModelConfig.from_file(filename, data_dir=data_dir)
        config.debug = debug
        return cls(
            data_dir=config.data_dir,
            entity_set=entity_set,
            transformer_name=config.transformer_name,
            max_candidates=max_candidates or config.max_candidates,
            debug=config.debug,
            use_precomputed_description_embeddings=use_precomputed_description_embeddings,
            model_description_embeddings_file=model_description_embeddings_file,
            ner_tag_to_ix=config.ner_tag_to_ix,
            return_titles=return_titles,
        )

    def class_check_spans(self, spans_to_check: List[Span]):
        """
        Modifies spans in-place. Checks the predicted fine-grained entity type and type of
        the predicted entity match.
        @param self: self
        @param spans_to_check: list of spans to check
        """
        self.class_handler.class_check_spans(spans_to_check)

    def prune_classes(self, class_ids: FrozenSet[str]) -> FrozenSet[str]:
        return self.class_handler.prune_classes(class_ids=class_ids)

    def get_transformer_model(self):
        return self.lookups.get_transformer_model()
