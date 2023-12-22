import itertools
import os
from collections import defaultdict
from typing import Iterable, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.cuda.amp import autocast
from torch.nn import DataParallel
from tqdm import trange

from refined.constants.mapping_constants import wikidata_to_spacy_types
from refined.constants.resources_constants import model_name_to_files
from refined.data_types.base_types import Entity, Span
from refined.data_types.doc_types import Doc
from refined.data_types.modelling_types import BatchedElementsTns, ModelReturn
from refined.doc_preprocessing.preprocessor import PreprocessorInferenceOnly, Preprocessor
from refined.model_components.config import ModelConfig
from refined.model_components.refined_model import RefinedModel
from refined.resource_management.aws import S3Manager
from refined.resource_management.resource_manager import ResourceManager
from refined.utilities.general_utils import round_list, get_logger, batch_items
from refined.utilities.numeric_handling.date_utils import SpanDateHandler
from refined.utilities.preprocessing_utils import convert_doc_to_tensors, convert_batch_elements_to_batched_tns

LOG = get_logger(__name__)


class Refined(object):
    """
    This class provides methods to process text.
    It wraps around document preprocessor and the ReFinED model class.
    """

    def __init__(
            self,
            model_file_or_model: Union[str, RefinedModel],
            model_config_file_or_model_config: Union[str, ModelConfig],
            entity_set: Optional[str] = None,
            data_dir: Optional[str] = None,
            n_gpu: int = 1,
            use_cpu: bool = False,
            device: Optional[str] = None,
            debug: bool = False,
            backward_coref: bool = False,
            max_seq: int = 510,
            use_precomputed_descriptions: bool = True,
            model_description_embeddings_file: Optional[str] = None,
            download_files: bool = False,
            inference_only: bool = True,
            return_titles: bool = False,
            preprocessor: Optional[Preprocessor] = None,
            max_candidates: int = 30,
    ):
        """
        Constructs instance of Refined class.
        :param data_dir: directory containing the data files (pickled and JSON dictionaries)
        :param model_file_or_model: path to directory containing the ReFinED model to use (directory must include training args)
        :param model_config_file_or_model_config: path to directory containing the ReFinED model to use (directory must include training
        args)
        :param use_cpu: use cpu instead of GPU
        :param device: GPU device to use (defaults to cuda:0) when `use_cpu` is False and GPU is available
        :param n_gpu: how many GPUs to use
        :param use_precomputed_descriptions: use precomputed descriptions when True
        :param backward_coref: if true will do 2 passes over spans (first pass collects human candidate entities,
                               and copies candidate over to partial mentions (e.g. surname instead of full name),
                               otherwise will build person co-reference dictionary sequentially so only forward coref
                               occurs.
        """

        self.model_file = model_file_or_model
        self.max_seq = max_seq
        if preprocessor is None:
            assert entity_set is not None, "`entity_set` must be provided when preprocessor is not provided as " \
                                           "an argument"
            assert data_dir is not None, "`data_dir` must be provided when preprocessor is not provided as an argument"
            self.preprocessor = PreprocessorInferenceOnly.from_model_config_file(
                filename=model_config_file_or_model_config,
                entity_set=entity_set,
                data_dir=data_dir,
                debug=debug,
                use_precomputed_description_embeddings=use_precomputed_descriptions,
                model_description_embeddings_file=model_description_embeddings_file,
                download_files=download_files,
                inference_only=inference_only,
                return_titles=return_titles,
                max_candidates=max_candidates,
            )
        else:
            self.preprocessor = preprocessor

        self.backward_coref = backward_coref

        if isinstance(model_file_or_model, RefinedModel) or isinstance(model_file_or_model, DataParallel):
            # RefinedModel or DataParallel model
            self.model = model_file_or_model
        else:
            assert isinstance(model_config_file_or_model_config, str), f"model_config_file_or_model_config must be" \
                                                                       f"a string when model_file_or_model is a string."
            self.model = RefinedModel.from_pretrained(
                model_file=model_file_or_model,
                model_config_file=model_config_file_or_model_config,
                preprocessor=self.preprocessor,
                use_precomputed_descriptions=use_precomputed_descriptions
            )

        # ensure all parameters are unfrozen
        # this is done in the case that the model is trained/fine-tuned
        # note that this is fine for inference as inference is run with no_grad() mode
        for params in self.model.parameters():
            params.requires_grad = True

        if device is not None:
            self.device = device
        else:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() and not use_cpu else "cpu"
            )

        if n_gpu is not None and n_gpu > 1 and isinstance(self.model, RefinedModel):
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(n_gpu)))
            self.model.to(f"cuda:{self.model.device_ids[0]}")
            self.device = f"cuda:{self.model.device_ids[0]}"
        self.n_gpu = n_gpu
        if n_gpu == 1:
            self.model.to(self.device)

        self.model.eval()

        self.special_span_handlers = {
            "DATE": SpanDateHandler(),
            # other handlers can be added here (to parse different entity types)
        }

    def process_text(
            self,
            text: str,
            spans: Optional[List[Span]] = None,
            ner_threshold: float = 0.5,
            prune_ner_types: bool = True,
            max_batch_size: int = 16,
            apply_class_check: bool = False,
            return_special_spans: bool = True,
    ) -> List[Span]:
        """
        Predicts the types and entity for each span (if spans are not provided they will be identified).
        Performs the preprocessing and batches the text doing multiple forward passes of the network when the
        batch size exceeds the limit (currently set as 16 with sequence length 300).
        :param text: the text to process (sentence/document) (will be batched if exceeds max seq)
        :param spans: optional argument to specify the spans to process.
        :param ner_threshold: threshold which must be met for the type to be returned (does not affect model - only
        has an effect on the return ner_type returned from this method)
        :param prune_ner_types: returns only fine-grained (non-impliable) NER types (does not affect ED)
        :param max_batch_size: max batch size
        :param apply_class_check: check the chosen entity has at least one of the predicted fine-grained types
        :param return_special_spans: if True, return all detected span types (i.e. return DATE spans etc.)
        :return: a list of spans with predictions attached to each span (sorted by start indices)
                 or a list of mention dictionary (in distant supervision dataset format) when ds_format is True.
        """
        all_spans = []
        if spans is not None:
            doc = Doc.from_text_with_spans(
                text, spans, self.preprocessor, backward_coref=self.backward_coref
            )
        else:
            doc = Doc.from_text(
                text,
                preprocessor=self.preprocessor
            )
        tns: Iterable[BatchedElementsTns] = convert_doc_to_tensors(
            doc,
            self.preprocessor,
            collate=True,
            max_batch_size=max_batch_size,
            sort_by_tokens=False,
            max_seq=self.max_seq,
        )
        self.model.eval()
        for batch_idx, batch in enumerate(tns):
            batch_spans = self.process_tensors(batch=batch, ner_threshold=ner_threshold,
                                               return_special_spans=return_special_spans)
            all_spans.extend(batch_spans)

        if prune_ner_types:
            self._prune_spans_ner(all_spans)

        # when spans are batched they could get shuffled up for batch efficiency
        # TODO if multiple documents are provided ensure the spans are sorted per doc instead of globally
        all_spans.sort(key=lambda x: x.start)
        if apply_class_check:
            self.preprocessor.class_check_spans(all_spans)

        # can add labels and other entity ids to spans here

        return all_spans

    def process_text_batch(
            self,
            texts: List[str],
            spanss: Optional[List[List[Span]]] = None,
            ner_threshold: float = 0.5,
            prune_ner_types: bool = True,
            max_batch_size: int = 16,
            apply_class_check: bool = False,
            return_special_spans: bool = True,
            sort_by_tokens: bool = True
    ) -> List[Doc]:
        """
        Batched version of process_text().
        This method uses the convert_batch_elements_to_batched_tns() method which could be much more
        efficient by directly returning tensors that do not require indexing - like done in _identify_entity_mentions().
        Currently, performance gain from batching is mostly present when document texts
        are short (such as question text).
        TODO: add optional already "indexed into" tensors to RefinedModel.forward() and use this during inference.
        # Note that "indexed into" tensors should not be used during multi-gpu training as it can cause issues.
        # `sort_by_tokens` is used to batch similar length chunks together for efficiency.
        # The current implementation means that the person name co-rereference trick will no longer work
        # across (512 token) chunks of the same document but there is a simple fix in _identify_entity_mentions().
        """
        all_spans = []

        docs = []
        if spanss is not None:
            for i, (text, spans) in enumerate(zip(texts, spanss)):
                doc = Doc.from_text_with_spans(
                    text, spans, self.preprocessor, backward_coref=self.backward_coref,
                    doc_id=i
                )
                docs.append(doc)
        else:
            for i, text in enumerate(texts):
                doc = Doc.from_text(
                    text, preprocessor=self.preprocessor, doc_id=i
                )
                docs.append(doc)
        batch_elements = [elem for doc in docs for elem in doc.to_batch_elements(
            preprocessor=self.preprocessor, override_max_seq=self.max_seq
        )]
        tns: Iterable[BatchedElementsTns] = convert_batch_elements_to_batched_tns(
                batch_elements,
                self.preprocessor,
                max_batch_size=max_batch_size,
                sort_by_tokens=sort_by_tokens,
            # TODO: currently sort_by_tokens=True means some person name co-reference can be missed.
            # To make this equivalent to running with sort_by_tokens=False, adjust the
            # in _identify_entity_mentions() is adjusted to keep track of names per doc and not assume batch_elements
            # for each document are sequential.
        )

        self.model.eval()

        for batch_idx, batch in enumerate(tns):
            batch_spans = self.process_tensors(batch=batch, ner_threshold=ner_threshold,
                                               return_special_spans=return_special_spans)
            all_spans.extend(batch_spans)

        if prune_ner_types:
            self._prune_spans_ner(all_spans)

        # when spans are batched they could get shuffled up for batch efficiency
        # TODO if multiple documents are provided ensure the spans are sorted per doc instead of globally
        all_spans.sort(key=lambda x: x.start)
        if apply_class_check:
            self.preprocessor.class_check_spans(all_spans)

        doc_id_to_doc = {doc.doc_id: doc for doc in docs}
        for doc in docs:
            doc.spans = []
        for span in all_spans:
            doc_id_to_doc[span.doc_id].spans.append(span)
        return docs

    def process_tensors(self, batch: BatchedElementsTns, ner_threshold: float = 0.5,
                        return_special_spans: bool = True) -> List[Span]:
        """
        Performs an forward pass of ReFinED model and returns the spans for the batch.
        :param batch: tensors for the batch elements in the batch
        :param ner_threshold: threshold for which NER types are added to spans (does not affect model predictions)
        :param return_special_spans: if True, return all detected span types (i.e. return DATE spans etc.)
        :return spans : (predetermined or predicted) spans with predictions attached
        """
        if batch.batch_elements[0].spans is not None:
            spans = [span for b in batch.batch_elements for span in b.spans if b.spans]
            if len(spans) == 0:
                # no spans to process (in predetermined spans mode)
                return []
        else:
            spans = None

        if self.n_gpu == 1 or True:
            batch = batch.to(self.device)
        self.model.eval()
        with autocast():
            if hasattr(torch, "inference_mode"):
                inference_mode = torch.inference_mode
            else:
                inference_mode = torch.no_grad
            with inference_mode():
                # if not end to end then spans do not need to be returned
                assert (spans is not None and len(spans) > 0) or (self.n_gpu == 1), (
                    "n_gpu must be 1 when spans are not " "provided."
                )
                output: ModelReturn = self.model(batch=batch)

        spans = output.entity_spans
        if not return_special_spans:
            predicted_other_spans = {}
        else:
            predicted_other_spans = output.other_spans

        # Resolve special spans (DATE, CARDINAL etc.)
        for span_ner_type, ner_type_spans in predicted_other_spans.items():
            # filter spans that are not in the NER types that we currently have a parser for.
            if span_ner_type in self.special_span_handlers:
                handler = self.special_span_handlers[span_ner_type]
                predicted_other_spans[span_ner_type] = handler.resolve_spans(spans=ner_type_spans)

        predicted_other_spans = list(itertools.chain.from_iterable(predicted_other_spans.values()))

        if spans is None or len(spans) == 0:
            # no predicted spans to process
            return [] + sorted(predicted_other_spans, key=lambda x: x.start)

        device = output.et_activations.device

        # add -1 to candidate entity id tensor for entity_not_in_list predictions
        cand_ids = torch.cat(
            [output.cand_ids, torch.ones((output.cand_ids.size(0), 1), device=device, dtype=torch.long) * -1], 1
        )

        ed_targets_predictions = output.ed_activations.argmax(dim=1)
        ed_targets_softmax = output.ed_activations.softmax(dim=1)

        description_scores = output.candidate_description_scores.detach().cpu().numpy()

        predicted_entity_ids = (
            cand_ids[torch.arange(cand_ids.size(0)), ed_targets_predictions].cpu().numpy().tolist()
        )
        predicted_entity_confidence = round_list(
            ed_targets_softmax[torch.arange(ed_targets_softmax.size(0)), ed_targets_predictions]
                .cpu()
                .numpy()
                .tolist(),
            4,
        )
        span_to_classes = defaultdict(list)
        span_indices, pred_class_indices = torch.nonzero(
            output.et_activations > ner_threshold, as_tuple=True
        )
        for span_idx, pred_class_idx, conf in zip(
                span_indices.cpu().numpy().tolist(),
                pred_class_indices.cpu().numpy().tolist(),
                round_list(
                    output.et_activations[(span_indices, pred_class_indices)].cpu().numpy().tolist(), 4
                ),
        ):
            if pred_class_idx == 0:
                continue  # skip padding class label
            class_id = self.preprocessor.index_to_class.get(pred_class_idx, "Q0")
            class_label = self.preprocessor.class_to_label.get(class_id, "no_label")
            span_to_classes[span_idx].append((class_id, class_label, conf))

        sorted_entity_ids_scores, old_indices = ed_targets_softmax.sort(descending=True)
        sorted_entity_ids_scores = sorted_entity_ids_scores.cpu().numpy().tolist()
        sorted_entity_ids = self.sort_tensor(cand_ids, old_indices).cpu().numpy().tolist()

        for span_idx, span in enumerate(spans):
            wikidata_id = f'Q{str(predicted_entity_ids[span_idx])}'
            span.predicted_entity = Entity(
                wikidata_entity_id=wikidata_id,
                wikipedia_entity_title=self.preprocessor.qcode_to_wiki.get(wikidata_id)
                if self.preprocessor.qcode_to_wiki is not None else None
            )
            span.entity_linking_model_confidence_score = predicted_entity_confidence[span_idx]
            span.top_k_predicted_entities = [
                (Entity(wikidata_entity_id=f'Q{entity_id}',
                        wikipedia_entity_title=self.preprocessor.qcode_to_wiki.get(wikidata_id)
                        if self.preprocessor.qcode_to_wiki is not None else None
                        ),
                 round(score, 4))
                for entity_id, score in
                zip(sorted_entity_ids[span_idx], sorted_entity_ids_scores[span_idx])
                if entity_id != 0
            ]

            span.candidate_entities = [
                (qcode, round(conf, 4))
                for qcode, conf in filter(lambda x: not x[0] == "Q0", span.candidate_entities)
            ]
            span.description_scores = round_list(
                description_scores[span_idx].tolist(), 4
            )  # matches candidate order
            span.predicted_entity_types = span_to_classes[span_idx]

        return sorted(spans + predicted_other_spans, key=lambda x: x.start)

    @staticmethod
    def sort_tensor(data_tns, permutation_tns) -> Tensor:
        """
        Based on https://discuss.pytorch.org/t/how-to-sort-tensor-by-given-order/61625.
        :param data_tns: input tensor
        :param permutation_tns: tensor with same shape as x that determine an ordering
        :return: an ordered tensor using permutation tensor
        """
        d1, d2 = data_tns.size()
        return data_tns[
            torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(), permutation_tns.flatten()
        ].view(d1, d2)

    def _prune_span_ner(self, span: Span) -> None:
        """
        Removes impliable classes from span (in-place).
        :param span: span
        """
        if span.predicted_entity_types is not None:
            # map Wikidata type to coarse spaCy type
            coarse_entity_types = [
                (wikidata_to_spacy_types[wikidata_id], conf)
                for wikidata_id, label, conf in span.predicted_entity_types
                if wikidata_id in wikidata_to_spacy_types
            ]
            coarse_entity_types.sort(key=lambda x: x[1], reverse=True)  # descending order
            if len(coarse_entity_types) > 0:
                span.coarse_mention_type = coarse_entity_types[0][0]

            # prune Wikidata types
            class_ids = [x[0] for x in span.predicted_entity_types]
            minimal_ids = self.preprocessor.prune_classes(frozenset(class_ids))
            span.predicted_entity_types = [
                type_pred for type_pred in span.predicted_entity_types if type_pred[0] in minimal_ids
            ]

            # override coarse type with pruned spaCy type
            coarse_entity_types = [
                (wikidata_to_spacy_types[wikidata_id], conf)
                for wikidata_id, label, conf in span.predicted_entity_types
                if wikidata_id in wikidata_to_spacy_types
            ]
            coarse_entity_types.sort(key=lambda x: x[1], reverse=True)  # descending order
            if len(coarse_entity_types) > 0:
                span.coarse_mention_type = coarse_entity_types[0][0]

    def _prune_spans_ner(self, spans: List[Span]) -> None:
        """
        Removes impliable classes from spans (in-place).
        :param spans: spans
        """
        for span in spans:
            self._prune_span_ner(span)

    @classmethod
    def from_pretrained(
            cls,
            model_name: str,
            entity_set: str,
            data_dir: Optional[str] = None,
            debug: bool = False,
            device: Optional[str] = None,
            use_precomputed_descriptions: bool = True,
            download_files: bool = True,
            return_titles: bool = True,
    ):
        """
        Load a pretrained ReFinED model.
        :param model_name: model name (e.g. wikipedia_model)
        :param data_dir: data directory path (by default will download files when needed from S3)
        :param debug: debug mode (loads data partially to speed up debugging)
        :param device: gpu or cpu device for model.
        :param use_precomputed_descriptions: use_precomputed_descriptions
        :return: `Refined` processor which wraps ReFinED model and provides easy to use methods.
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.expanduser('~'), '.cache', 'refined')
        assert entity_set in ["wikipedia", "wikidata"], f"entity_set should be one of ['wikipedia', 'wikidata'] but " \
                                                        f"is {entity_set}"
        if model_name in model_name_to_files:
            # model_name is one of the library provided models {wikipedia, wikipedia_with_numbers}
            resource_manager = ResourceManager(s3_manager=S3Manager(),
                                               data_dir=data_dir,
                                               model_name=model_name,
                                               entity_set=entity_set,
                                               load_descriptions_tns=not use_precomputed_descriptions,
                                               load_qcode_to_title=return_titles
                                               )
            model_files = resource_manager.get_model_files()
        else:
            # model_name refers to the file path of a custom trained/fine-tuned ReFinED model
            # the file path must contain a file for model.pt and config.json,
            model_files = {
                'model': os.path.join(model_name, 'model.pt'),
                'model_config': os.path.join(model_name, 'config.json'),
                'model_description_embeddings': {
                    # TODO: remove hard-coded size of entity sets
                    'wikipedia': os.path.join(model_name, 'precomputed_entity_descriptions_emb_wikipedia_6269457-300.np'),
                    'wikidata': os.path.join(model_name, 'precomputed_entity_descriptions_emb_wikidata_33831487-300.np')
                }[entity_set],
            }
            # model_description_embeddings
            if not os.path.exists(model_files["model_description_embeddings"]):
                LOG.info(f"Precomputed entity description embeddings cannot be loaded from from "
                         f"{model_files['model_description_embeddings']} "
                         f"Use precompute_description_embeddings.py script to generate these embeddings to get a "
                         f"significant inference speed improvement.")
                del model_files["model_description_embeddings"]

        return cls(
            model_file_or_model=model_files["model"],
            model_config_file_or_model_config=model_files["model_config"],
            model_description_embeddings_file=model_files.get("model_description_embeddings"),
            entity_set=entity_set,
            data_dir=data_dir,
            device=device,
            use_precomputed_descriptions=use_precomputed_descriptions,
            debug=debug,
            download_files=download_files,
            return_titles=return_titles,
        )

    def precompute_description_embeddings(self, output_dir: Optional[str] = None):
        self.model.eval()
        if output_dir is None:
            output_dir = os.path.dirname(self.model_file)
        os.makedirs(output_dir, exist_ok=True)
        dim = self.model.ed_2.description_encoder.output_dim
        shape = (self.preprocessor.descriptions_tns.size(0), dim)
        # using float16 instead of float32 decreases performance by about 0.06 F1, but halves memory requirement
        # so it is worth the trade-off.
        output_filename = os.path.join(output_dir, f"precomputed_entity_descriptions_emb_{self.preprocessor.entity_set}_{shape[0]}-{shape[1]}.np")

        precomputed_desc = np.memmap(output_filename,
                                     shape=shape,
                                     dtype=np.float16,
                                     mode="w+")
        for indices in batch_items(trange(shape[0]), n=256):
            with torch.no_grad():
                desc = self.model.ed_2.description_encoder(
                    self.preprocessor.descriptions_tns[indices].unsqueeze(0).to(self.device).long())
                precomputed_desc[indices] = desc.cpu().numpy()

        # set masked to 0 (could do this during for loop or after) assumes index 0 has no description
        precomputed_desc[(precomputed_desc[:, 0] == precomputed_desc[0, 0])] = 0
        assert precomputed_desc[0].sum() == 0.0, "First row should be 0s as used for masking" \
                                                 " of padded descriptions in " \
                                                 "description encoder."

        LOG.info('Saving precomputed_desc', precomputed_desc.shape)
        LOG.info(f"Saved precomputed description embeddings to {output_filename}")
