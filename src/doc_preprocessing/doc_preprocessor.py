import json
import os
from random import random, sample
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import nltk
import numpy as np
import smart_open
import torch
from doc_preprocessing.dataclasses import AdditionalEDFeatures, DocPreprocessor, Span
from doc_preprocessing.resource_manager import get_data_files, ResourceManager, get_mmap_shape
from utilities.aws import S3Manager
from model_components.config import ModelConfig
from utilities.general_utils import get_logger, get_tokenizer, unique
from utilities.lookup_utils import (
    load_disambiguation_qcodes,
    load_human_qcode,
    load_pem,
    load_qcode_to_idx,
    load_redirects,
    load_subclasses,
    load_wikipedia_to_qcode,
    normalize_surface_form,
    title_to_qcode,
    load_labels
)

import ujson as json

logger = get_logger(__name__)


class DocumentPreprocessorMemoryBased(DocPreprocessor):
    """
    Implementation of DocPreprocessor that used in-memory python dictionaries.
    See `DocPreprocessor` for additional documentation.
    """

    def __init__(
        self,
        transformer_name: str,
        entity_set: str,
        max_candidates: int,
        ner_tag_to_ix: Dict[str, int],
        data_dir: str,
        debug: bool = False,
        requires_redirects_and_disambig: bool = True,
        zero_features: bool = False,
        zero_string_features: bool = True,
        load_descriptions_tns: bool = False,
    ):
        """
        Constructs DocumentPreprocessorMemoryBased object.
        :param data_dir: path to directory containing the resources files
        :param max_candidates: max number of candidate entities to fetch
        :param debug: debug only loads a line for each file
        :param requires_redirects_and_disambig: whether to load redirects (only needed for pre-processing links)
        """
        super().__init__(
            data_dir=data_dir, transformer_name=transformer_name, max_candidates=max_candidates,
            ner_tag_to_ix=ner_tag_to_ix
        )
        self.load_descriptions_tns = load_descriptions_tns
        self.precomputed_descriptions = None
        self.requires_redirects_and_disambig = requires_redirects_and_disambig
        self.debug = debug
        self.zero_string_features = zero_string_features
        self.zero_features = zero_features
        self.transformer_name = transformer_name
        self.entity_set = entity_set

        _tokenizer = get_tokenizer(transformer_name=transformer_name, data_dir=data_dir)
        self.cls_id = _tokenizer.cls_token_id
        self.sep_id = _tokenizer.sep_token_id
        self.pad_id = _tokenizer.pad_token_id
        self.mask_id = _tokenizer.mask_token_id
        self.vocab_size = _tokenizer.vocab_size
        self.max_candidates = max_candidates

        logger.info("Loading resources")
        self.load_resources()

    def get_resource_uris(self, entity_set: str) -> Dict[str, str]:
        resource_to_uri: Dict[str, str] = dict()
        data_files = get_data_files(entity_set=entity_set)
        for resource_name, resource_locations in data_files.items():
            resource_to_uri[resource_name] = os.path.join(
                self.data_dir, resource_locations["local_filename"]
            )
        return resource_to_uri

    def load_resources(self):
        resource_to_uri = self.get_resource_uris(entity_set=self.entity_set)
        # shape = (num_ents, max_num_classes)
        print("Loading qcode_idx_to_class_idx")
        with smart_open.smart_open(resource_to_uri["qcode_idx_to_class_idx"], "rb") as f:
            self.qcode_idx_to_class_idx = np.memmap(
                resource_to_uri["qcode_idx_to_class_idx"],
                shape=get_mmap_shape(resource_to_uri["qcode_idx_to_class_idx"]),
                mode="r",
                dtype=np.int16,
            )
        print("Loaded qcode_idx_to_class_idx")

        if self.load_descriptions_tns:
            with smart_open.smart_open(resource_to_uri["descriptions_tns"], "rb") as f:
                self.descriptions_tns = torch.load(f)
            # with smart_open.smart_open(resource_to_uri["descriptions_tns"], "rb") as f:
            #     if self.debug:
            #         self.descriptions_tns = torch.ones([6500000, 32])
            #     else:
            #         self.descriptions_tns = torch.load(io.BytesIO(f.read())).long()

        self.pem: Dict[str, List[Tuple[str, float]]] = load_pem(
            pem_file=resource_to_uri["wiki_pem"], is_test=self.debug, max_cands=self.max_candidates
        )

        with smart_open.smart_open(resource_to_uri["class_to_label"], "r") as f:
            self.class_to_label: Dict[str, Any] = json.load(f)

        self.human_qcodes: Set[str] = load_human_qcode(
            resource_to_uri["human_qcodes"], is_test=self.debug
        )

        self.wiki_to_qcode: Dict[str, str] = load_wikipedia_to_qcode(
            resource_to_uri["wiki_to_qcode"], is_test=self.debug
        )
        self.qcode_to_wiki: Dict[str, str] = {v: k for k, v in self.wiki_to_qcode.items()}

        print("temp")
        self.qcode_to_label = {}
        # self.qcode_to_label: Dict[str, str] = load_labels(resource_to_uri["qcode_to_label"],
        #                                                   is_test=self.debug)

        if self.requires_redirects_and_disambig:
            self.redirects: Dict[str, str] = load_redirects(
                resource_to_uri["redirects"], is_test=self.debug
            )
            self.disambiguation_qcodes: Set[str] = load_disambiguation_qcodes(
                resource_to_uri["disambiguation_qcodes"], is_test=self.debug
            )

        self.subclasses, self.subclasses_reversed = load_subclasses(
            resource_to_uri["subclasses"], is_test=self.debug
        )

        self.qcode_to_idx: Dict[str, int] = load_qcode_to_idx(
            resource_to_uri["qcode_to_idx"], is_test=self.debug
        )

        with smart_open.smart_open(resource_to_uri["class_to_idx"], "r") as f:
            self.class_to_idx = json.load(f)

        self.index_to_class = {y: x for x, y in self.class_to_idx.items()}
        self.classes = list(self.class_to_idx.keys())
        self.max_num_classes_per_ent = self.qcode_idx_to_class_idx.shape[1]
        self.num_classes = len(self.class_to_idx)
        self.precomputed_descriptions = None
        logger.info("Loaded all")

    def set_precomputed_descriptions(self, file: str):
        print("Reading description embeddings memory map", file)
        with smart_open.smart_open(file, "rb") as f:
            self.precomputed_descriptions = np.memmap(
                file,
                shape=get_mmap_shape(file),
                # shape=(33831487, 300),
                dtype=np.float32,
                mode="r",
                order="C"
            )
            assert self.precomputed_descriptions[0].sum() == 0.0, "First row should be 0s to ensure proper masking of " \
                                                                  "descriptions for padded entities in description " \
                                                                  "encoder"

    def get_string_sim_features(
        self, qcodes: List[str], mention_surface_form: str
    ) -> List[AdditionalEDFeatures]:
        features = []
        for qcode in qcodes:
            if qcode in self.qcode_to_label:
                # https://arxiv.org/pdf/1705.02494v3.pdf
                # TODO: remove all these features as have no impact on performance
                label_surface_form = self.qcode_to_label[qcode]
                edit_dist = float(nltk.edit_distance(label_surface_form, mention_surface_form))
                label_equals_m = float(label_surface_form == mention_surface_form)
                label_contains_m = float(mention_surface_form in label_surface_form)
                label_starts_with_m = float(label_surface_form.startswith(mention_surface_form))
                label_ends_with_m = float(label_surface_form.endswith(mention_surface_form))
                if not self.zero_string_features:
                    features.append(
                        AdditionalEDFeatures(
                            edit_dist=edit_dist,
                            label_equals_m=label_equals_m,
                            label_contains_m=label_contains_m,
                            label_starts_with_m=label_starts_with_m,
                            label_ends_with_m=label_ends_with_m,
                        )
                    )
                else:
                    features.append(AdditionalEDFeatures.get_pad_element())
            else:
                features.append(AdditionalEDFeatures.get_pad_element())
        return features

    def get_classes_idx_for_qcode_batch(
        self, qcodes: List[str], shape: Tuple[int, ...] = None
    ) -> torch.Tensor:
        """
        Retrieves all of the classes indices for the qcodes (from various relations used to construct the lookup).
        :param qcodes: qcodes
        :param shape: shape/size of tensor returned
        :return: long tensor with shape (num_ents, self.max_num_classes) (qcode index 0 is used for padding all classes)
        """
        result = torch.tensor(
            self.qcode_idx_to_class_idx[
                [self.qcode_to_idx[qcode] if qcode in self.qcode_to_idx else 0 for qcode in qcodes]
            ],
            dtype=torch.long,
        )
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
        )
        if shape is not None:
            return result.view(shape)
        else:
            return result

    def get_candidates(
        self,
        surface_form: str,
        person_coref_ref: Dict[str, List[Tuple[str, float]]] = None,
        pruned_candidates: Optional[Set[str]] = None,
        dropout=0.1,
        zero_features: bool = False,
        sample_k_candidates: Optional[int] = None,
        gold_qcode: Optional[str] = None,
    ) -> Tuple[List[Tuple[str, float]], Dict[str, List[Tuple[str, float]]]]:
        """
        Generates a list of candidate entities for a surface form using a pem lookup. Person name coreference is done
        by copying pem values from longer person name to short mentions of person name e.g. "Donald Trump" and "Trump".
        :param surface_form: surface form to generate candidate for (does not need to be normalised)
        :param person_coref_ref: person co-reference dictionary (keeps track of person names and part of names)
        :param dropout: dropout
        :param pruned_candidates: optional set of candidates keep (rest will be pruned if this is arg is set used)
        :param sample_k_candidates: sample k candidates from the top 30 list of candidates
        :param gold_qcode: gold entity id (only used for training sampling)
        :param zero_features: remove P(e|m) features
        :return: (list of candidates with pem value, dictionary of person name to candidates with pem value).
        """
        max_cands = sample_k_candidates if sample_k_candidates is not None else self.max_candidates
        person_coref_pem_cap = (
            0.80  # the cap for pem value to be set when origin is person name co-reference
        )
        person_coref_pem_min = 0.05  # minimum pem value to apply person name co-reference to
        if person_coref_ref is None:
            person_coref_ref: Dict[str, List[Tuple[str, float]]] = dict()

        surface_form_norm = normalize_surface_form(surface_form, remove_the=True)
        if surface_form_norm not in self.pem:
            if surface_form_norm in person_coref_ref:
                cands = person_coref_ref[surface_form_norm]
                cands = [(cand[0], 0.5) for cand in cands] if zero_features else cands
                return (cands + [("Q0", 0.0)] * max_cands)[:max_cands], person_coref_ref
            else:
                return [("Q0", 0.0)] * max_cands, person_coref_ref

        # surface is in pem
        # direct candidates means was directly in pem lookup
        direct_cands = self.pem[surface_form_norm]

        # add short names to person_coref for all people candidates
        person_short_names = surface_form_norm.split(" ")
        short_name_cands = []
        for qcode, pem_value in direct_cands:
            if qcode in self.human_qcodes and pem_value > person_coref_pem_min:
                short_name_cands.append((qcode, min(pem_value, person_coref_pem_cap)))  # cap
        if len(short_name_cands) > 0 and len(person_short_names) > 1:
            for short_name in person_short_names:
                person_coref_ref[short_name] = short_name_cands

        # check to see if surface form is a person name co-reference
        if surface_form_norm in person_coref_ref:
            indirect_cands = person_coref_ref[surface_form_norm]
            cands = list(
                unique(
                    lambda x: x[0],
                    sorted(direct_cands + indirect_cands, key=lambda x: x[1], reverse=True),
                )
            )
        else:
            cands = direct_cands

        if sample_k_candidates is not None:
            popular_negatives = sample_k_candidates // 2  # half are popular
            random_negatives = max(
                sample_k_candidates - popular_negatives - 1, 0
            )  # half are random

            assert gold_qcode is not None, "gold_qcode must be set when sample_k_candidate is set"
            # assuming it is already sorted TODO: check this
            negative_cands = [cand for cand in cands[:30] if cand[0] != gold_qcode]
            gold_cand = [cand for cand in cands[:50] if cand[0] == gold_qcode]
            sampled_cands = negative_cands[:popular_negatives]
            sampled_cands += sample(
                negative_cands[popular_negatives:],
                k=min(random_negatives, len(negative_cands[popular_negatives:])),
            )
            if random() > dropout:
                sampled_cands += gold_cand
            cands = sampled_cands

        if zero_features:
            cands = [(qcode, 0.50) for qcode, pem in cands]

        if pruned_candidates:
            cands = [(qcode, pem_value) for qcode, pem_value in cands if qcode in pruned_candidates]
            if len(cands) == 0:
                uniform_prob = 1 / (len(pruned_candidates) + 1e-8)
                cands = [(qcode, uniform_prob) for qcode in pruned_candidates]
            else:
                lowest_pem_value = min(x[1] for x in cands)
                cand_qcodes = {x[0] for x in cands}
                for pruned_candidate in pruned_candidates:
                    if pruned_candidate not in cand_qcodes:
                        cands.append((pruned_candidate, lowest_pem_value))

                # normalize
                total = sum(x[1] for x in cands) + 1e-8
                cands = [(qcode, pem_value / total) for qcode, pem_value in cands]

        return (cands + [("Q0", 0.0)] * max_cands)[:max_cands], person_coref_ref

    _get_implied_classes_cache = (
        {}
    )  # lru_cache does not work with instances methods so basic implementation is done

    def _get_implied_classes(
        self, direct_classes: FrozenSet[str], remove_self=True
    ) -> FrozenSet[str]:
        """
        From a set of (direct) classes this method will generate all of the classes that can be implied.
        When remove_self is True it means that a class cannot be implied from itself (but it can still be implied
        by other of the direct classes).
        :param direct_classes: the set of classes for implied classes to be generated from
        :param remove_self: when true a classes implication is not reflexive (e.g. human does not imply human)
        :return: set of classes that can be implied from direct_classes
        """
        cache_key = (direct_classes, remove_self)
        if cache_key in self._get_implied_classes_cache:
            return self._get_implied_classes_cache[cache_key]

        if remove_self:
            all_implied_classes = set()
        else:
            all_implied_classes = set(direct_classes)

        # keep track of the classes that have been explored to prevent work from being repeated
        explored_classes = set()
        for direct_class in direct_classes:
            implied_classes = self._explore_class_tree(direct_class, frozenset(explored_classes))
            if remove_self:
                implied_classes = implied_classes - {direct_class}

            explored_classes.update(implied_classes)
            all_implied_classes.update(implied_classes)

        result = frozenset(all_implied_classes)
        self._get_implied_classes_cache[cache_key] = result
        if len(self._get_implied_classes_cache) > self.MAX_CACHE_ITEMS:
            self._get_implied_classes_cache.popitem()
        return result

    def _explore_class_tree(
        self, class_id: str, explored_classes: FrozenSet[str]
    ) -> FrozenSet[str]:
        """
        Recursively explores the class hierarchy (parent classes, parent of parents, etc.)
        Returns all of the explored classes (these are all impliable from the class provided as an argument (evi_class))
        :param class_id: class id for class to explore
        :param explored_classes: the classes impliable from class_id
        :return: a set of classes that are (indirect) direct ancestors of class_id
        """
        # This method will explore evi_class so add it to the explored_classes set to prevent repeating the work
        explored_classes = set(explored_classes)
        explored_classes.add(class_id)
        explored_classes.copy()

        # Base case: Evi class has no super classes so return the explored classes
        if class_id not in self.subclasses:
            return frozenset(explored_classes)

        # General case: Explore all unexplored super classes
        for super_class in self.subclasses[class_id]:
            if super_class not in explored_classes:
                explored_classes.add(super_class)
                explored_super_classes = self._explore_class_tree(
                    super_class, frozenset(explored_classes)
                )
                explored_classes.update(explored_super_classes)
        return frozenset(explored_classes)

    MAX_CACHE_ITEMS = 10000000
    _prune_classes_cache = (
        {}
    )  # lru_cache does not work with instances methods so basic implementation is done here

    def prune_classes(self, class_ids: FrozenSet[str]) -> FrozenSet[str]:
        """
        Prune classes that are impliable from other provided classes.
        Note that this also filters country and sport relation as well.
        :param class_ids: a set of classes
        :return: set of fine-grained classes that are not inferred by each other (i.e. classes are different sub-trees)
        """
        if class_ids in self._prune_classes_cache:
            return self._prune_classes_cache[class_ids]
        classes = frozenset({class_id for class_id in class_ids if "<" not in class_id})
        implied_classes = self._get_implied_classes(classes, remove_self=True)
        result = frozenset({str(x) for x in classes - implied_classes})
        self._prune_classes_cache[class_ids] = result
        if len(self._prune_classes_cache) > self.MAX_CACHE_ITEMS:
            self._prune_classes_cache.popitem()
        return result

    def add_candidates_to_spans(
        self,
        spans: List[Span],
        backward_coref: bool = False,
        candidate_dropout: float = 0.0,
        sample_k_candidates: Optional[int] = None,
        add_additional_ed_features: bool = True,
        person_coreference: Optional[Dict[str, List[Tuple[str, float]]]] = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Add candidate entities to spans
        :param spans: list of spans
        :param backward_coref: if true will do 2 passes over spans (first pass collects human candidate entities,
                               and copies candidate over to partial mentions (e.g. surname instead of full name),
                               otherise will build person coreference dictionary sequentially so only forward coref
                               occurs.
        :param candidate_dropout: candidate dropout
        :param sample_k_candidates: sample k random candidates from top 30
        :param add_additional_ed_features: add additional ed string similarity features
        :param person_coreference: person coreference dictionary
        """
        if person_coreference is None:
            person_coreference: Dict[str, List[Tuple[str, float]]] = dict()
        if backward_coref:
            for (
                span
            ) in (
                spans
            ):  # pre-populate person coreference dictionary so coreference can occur backwards.
                _, person_coreference = self.get_candidates(
                    surface_form=span.text,
                    person_coref_ref=person_coreference,
                    pruned_candidates=span.pruned_candidates,
                    dropout=0.0,
                    zero_features=self.zero_features,
                )
        for span in spans:
            candidates_qcodes, person_coreference = self.get_candidates(
                surface_form=span.text,
                person_coref_ref=person_coreference,
                pruned_candidates=span.pruned_candidates,
                dropout=candidate_dropout,
                zero_features=self.zero_features,
                sample_k_candidates=sample_k_candidates,
                gold_qcode=span.gold_entity_id,
            )
            span.candidate_entity_ids = candidates_qcodes
            if add_additional_ed_features and not self.zero_string_features:
                span.additional_ed_features = self.get_string_sim_features(
                    [x[0] for x in span.candidate_entity_ids], span.text
                )
        return person_coreference

    def map_title_to_qcode(self, title: str) -> Optional[str]:
        return title_to_qcode(
            title, redirects=self.redirects, wikipedia_to_qcode=self.wiki_to_qcode
        )

    @classmethod
    def from_model_config_file(
        cls,
        filename: str,
        entity_set: str,
        data_dir: str,
        debug: bool = False,
        requires_redirects_and_disambig: bool = False,
        load_descriptions_tns: bool = False,
        download_files: bool = False
    ):

        if download_files:
            resource_manager = ResourceManager(s3_manager=S3Manager(), data_dir=data_dir)
            resource_manager.download_data_if_needed(load_descriptions_tns=load_descriptions_tns,
                                                     entity_set=entity_set)

        config_file_uri = filename

        config = ModelConfig.from_file(config_file_uri, data_dir=data_dir)
        config.debug = debug
        return cls(
            data_dir=config.data_dir,
            entity_set=entity_set,
            transformer_name=config.transformer_name,
            max_candidates=config.max_candidates,
            debug=config.debug,
            requires_redirects_and_disambig=requires_redirects_and_disambig,
            load_descriptions_tns=load_descriptions_tns,
            ner_tag_to_ix=config.ner_tag_to_ix
        )

    def class_check_span(self, span_to_check: Span):
        if span_to_check.pred_entity_id is not None and "wikidata_qcode" in span_to_check.pred_entity_id[0]:
            predicted_entity = span_to_check.pred_entity_id[0]["wikidata_qcode"]
            predicted_classes = {qcode for qcode, label, conf in span_to_check.pred_types}
            entity_classes = self.get_classes_idx_for_qcode_batch([predicted_entity])
            class_indices = entity_classes[entity_classes != 0].numpy().tolist()
            entity_classes = [self.index_to_class[idx] for idx in class_indices]
            entity_classes = [c for c in entity_classes if "<" not in c]
            entity_classes = self._get_implied_classes(frozenset(entity_classes), remove_self=False)
            if len(set(predicted_classes) & entity_classes) > 0 or len(entity_classes) == 0:
                span_to_check.failed_class_check = False
            else:
                span_to_check.pred_entity_id = (span_to_check.pred_entity_id[0], -1.0)
                span_to_check.failed_class_check = True

    def class_check_spans(self, spans_to_check: List[Span]):
        for span_to_check in spans_to_check:
            self.class_check_span(span_to_check)
