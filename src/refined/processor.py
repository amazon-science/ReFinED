import os
import itertools
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

import torch
from doc_preprocessing.dataclasses import BatchedElementsTns, Doc, Span
from doc_preprocessing.doc_preprocessor import DocumentPreprocessorMemoryBased
from doc_preprocessing.preprocessing_utils import convert_doc_to_tensors
from doc_preprocessing.resource_manager import get_model_files, ResourceManager
from model_components.refined_model import RefinedModel
from torch import Tensor
from utilities.general_utils import round_list, wikidata_to_spacy_types
from utilities.date_utils import SpanDateHandler
from utilities.aws import S3Manager


class Refined(object):
    """
    This class provides methods to process text.
    It wraps around document preprocessor and the ReFinED model class.
    """

    def __init__(
            self,
            model_file: str,
            model_config_file: str,
            entity_set: str,
            data_dir: str,
            n_gpu: int = 1,
            use_cpu: bool = False,
            device: Optional[str] = None,
            debug: bool = False,
            requires_redirects_and_disambig: bool = False,
            backward_coref: bool = False,
            max_seq: int = 510,
            use_precomputed_descriptions: bool = True,
            model_description_embeddings_file: Optional[str] = None,
            load_descriptions_tns: bool = False,
            download_files: bool = False
    ):
        """
        Constructs instance of Refined class.
        :param data_dir: directory containing the data files (pickled and JSON dictionaries)
        :param model_file: path to directory containing the ReFinED model to use (directory must include training args)
        :param model_config_file: path to directory containing the ReFinED model to use (directory must include training args)
        :param use_cpu: use cpu instead of GPU
        :param device: GPU device to use (defaults to cuda:0) if use_cpu is false and GPI is available
        :param n_gpu: how many GPUs to use
        :param requires_redirects_and_disambig: requires_redirects_and_disambiguation lookups
        :param use_precomputed_descriptions: use precomputed descriptions when True
        :param backward_coref: if true will do 2 passes over spans (first pass collects human candidate entities,
                               and copies candidate over to partial mentions (e.g. surname instead of full name),
                               otherwise will build person coreference dictionary sequentially so only forward coref
                               occurs.
        :param load_descriptions_tns: load description tensor (should not be True in production) as it is slower
                                      than using precomputed descriptions.
        """
        self.max_seq = max_seq
        self.preprocessor = DocumentPreprocessorMemoryBased.from_model_config_file(
            filename=model_config_file,
            entity_set=entity_set,
            data_dir=data_dir,
            debug=debug,
            requires_redirects_and_disambig=requires_redirects_and_disambig,
            load_descriptions_tns=load_descriptions_tns or not use_precomputed_descriptions,
            download_files=download_files
        )
        self.backward_coref = backward_coref

        self.model = RefinedModel.from_pretrained(
            model_file=model_file,
            model_config_file=model_config_file,
            preprocessor=self.preprocessor,
            use_precomputed_descriptions=use_precomputed_descriptions,
            description_embeddings_file=model_description_embeddings_file,
        )
        # ensure all parameters are unfrozen
        for params in self.model.parameters():
            params.requires_grad = True

        if device is not None:
            self.device = device
        else:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() and not use_cpu else "cpu"
            )

        if n_gpu is not None and n_gpu > 1:
            print("using multi gpu", n_gpu)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(n_gpu)))
            self.model.to(f"cuda:{self.model.device_ids[0]}")
            self.device = f"cuda:{self.model.device_ids[0]}"
        self.n_gpu = n_gpu
        if n_gpu == 1:
            self.model.to(self.device)
        self.model.eval()

        self.special_span_handlers = {
            "DATE": SpanDateHandler()
        }

    def process_text(
            self,
            text: str,
            spans: Optional[List[Span]] = None,
            ner_threshold: float = 0.5,
            prune_ner_types: bool = True,
            max_batch_size: int = 16,
            apply_class_check: bool = True,
            return_special_spans: bool = True,
    ) -> Union[List[Span], List[Dict[str, Any]]]:
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
                transformer_name=self.preprocessor.transformer_name,
                data_dir=self.preprocessor.data_dir,
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
        all_spans.sort(key=lambda x: x.start)
        if apply_class_check:
            self.preprocessor.class_check_spans(all_spans)

        # add Wikidata labels to the linked entity
        for span in all_spans:
            if span.pred_entity_id is not None:
                span.entity_label = self.preprocessor.qcode_to_label.get(
                    span.pred_entity_id[0].get("wikidata_qcode", "none")
                )

        return all_spans

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
            batch = BatchedElementsTns(
                *[
                    x.to(self.device, non_blocking=True) if isinstance(x, Tensor) else x
                    for x in batch
                ]
            )
        self.model.eval()
        with torch.no_grad():
            # if not end to end then spans do not need to be returned
            assert (spans is not None and len(spans) > 0) or (self.n_gpu == 1), (
                "n_gpu must be 1 when spans are not " "provided."
            )
            (
                _,
                md_activations,
                _,
                et_activations,
                _,
                ed_activations,
                spans,
                predicted_other_spans,
                cand_ids,
                _,
                description_scores,
            ) = self.model(
                token_ids=batch.token_id_values,
                token_acc_sums=batch.token_acc_sum_values,
                entity_mask=batch.entity_mask_values,
                class_targets=batch.class_target_values,
                attention_mask=batch.attention_mask_values,
                token_type_ids=batch.token_type_values,
                candidate_entity_targets=batch.candidate_target_values,
                candidate_pem_values=batch.pem_values,
                candidate_classes=batch.candidate_class_values,
                candidate_pme_values=batch.pme_values,
                entity_index_mask_values=batch.entity_index_mask_values,
                spans=spans,
                batch_elements=batch.batch_elements,
                cand_ids=batch.candidate_qcode_values,
                cand_desc=batch.candidate_desc,
                cand_desc_emb=batch.candidate_desc_emb,
                candidate_features=batch.candidate_features,
                batch_elements_included=torch.arange(
                    batch.entity_index_mask_values.size(0)
                ).unsqueeze(-1)
                if batch.entity_index_mask_values is not None
                else None,
            )

        if not return_special_spans:
            predicted_other_spans = {}

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

        device = et_activations.device

        # add -1 to candidate entity id tensor for entity_not_in_list predictions
        cand_ids = torch.cat(
            [cand_ids, torch.ones((cand_ids.size(0), 1), device=device, dtype=torch.long) * -1], 1
        )

        ed_targets_predictions = ed_activations.argmax(dim=1)
        ed_targets_softmax = ed_activations.softmax(dim=1)

        description_scores = description_scores.detach().cpu().numpy()

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
            et_activations > ner_threshold, as_tuple=True
        )
        for span_idx, pred_class_idx, conf in zip(
                span_indices.cpu().numpy().tolist(),
                pred_class_indices.cpu().numpy().tolist(),
                round_list(
                    et_activations[(span_indices, pred_class_indices)].cpu().numpy().tolist(), 4
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
            sitelinks = self.get_sitelinks_qcode("Q" + str(predicted_entity_ids[span_idx]))
            span.pred_entity_id = (sitelinks, predicted_entity_confidence[span_idx])
            sorted_entity_ids_sitelinks = map(
                lambda x: self.get_sitelinks_qcode("Q" + str(x)),
                filter(lambda x: not x == 0, sorted_entity_ids[span_idx]),
            )
            span.pred_ranked_entity_ids = [
                (qcode, round(conf, 4))
                for qcode, conf in list(
                    zip(sorted_entity_ids_sitelinks, sorted_entity_ids_scores[span_idx])
                )
            ]

            span.candidate_entity_ids = [
                (qcode, round(conf, 4))
                for qcode, conf in filter(lambda x: not x[0] == "Q0", span.candidate_entity_ids)
            ]
            span.description_scores = round_list(
                description_scores[span_idx].tolist(), 4
            )  # matches candidate order
            span.pred_types = span_to_classes[span_idx]
            span.additional_ed_features = None

        return sorted(spans + predicted_other_spans, key=lambda x: x.start)

    def get_sitelinks_qcode(self, qcode: str) -> Dict[str, str]:
        """
        Returns dictionary of sitelinks for a qcode (e.g. "Q5"), including the wikidata sitelink itself.
        :param qcode: qcode
        :return: Wikipedia title, and Wikidata qcode (itself) in dictionary when they exist.
        """
        if qcode == "Q-1":
            return {"entity not in list": ""}
        sitelinks: Dict[str, str] = dict()
        if int(qcode.replace('Q', '')) < -1:
            # additional entity id

            sitelinks["wikidata_qcode"] = qcode.replace('Q', 'A')
        else:
            sitelinks["wikidata_qcode"] = qcode
        return sitelinks

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

    def process_text_batch(
            self,
            texts: List[str],
            spanss: Optional[List[Optional[List[Span]]]] = None,
            return_ner_types: Optional[bool] = True,
            ner_threshold: float = 0.5,
            max_batch_size=64,
            prune_ner_types: bool = True,
            ds_format: bool = False,
    ) -> List[List[Span]]:
        """
        Efficiently batched version of the method process_text().
        Equivalent to process_text() but batches text together based on length to improve throughput.
        Predicts the types and entity for each span (if spans are not provided spaCy ner is used).
        Performs the preprocessing and batches the text doing multiple forward passes of the network when the
        batch size exceeds the limit (currently set as 16 with sequence length 300).
        :param texts: the text to process (sentence/document) (will be batched if exceeds max seq)
        :param spanss: optional argument to specify the spans to process.
        :param return_ner_types: when true will return the ner types
        :param ner_threshold: threshold which must be met for the type to be returned (does not affect model - only
        has an effect on the return ner_type returned from this method)
        :param max_batch_size: max batch size
        :param prune_ner_types: returns only fine-grained (non-impliable) NER types (does not affect ED)
        :param ds_format: return distant supervision data format
        :return: a list of spans with predictions attached to each span (sorted by start indices)
        """
        raise NotImplementedError("This method has not been implemented yet.")

    def process_text_iterator(
            self,
            articles: Iterator,
            ner_threshold: float = 0.5,
            num_workers: int = 16,
            max_batch_size: int = 64,
            prune_ner_types: bool = True,
    ) -> Iterator[Doc]:
        """
        Version of the method process_text which accepts an iterator of articles as input. The articles will be batched
        together for processing. Note that due to multiprocessing of the data-preprocessing steps, the order in which
        articles are returned will NOT match the original order.
        :param articles: iterator which yields a dictionaries of format {'text': article_text, 'spans': article_spans,
          'metadata': metadata}.
           Where article_text is the text of the article, and article_spans is an optional list of Span objects,
           and metadata is any optional dictionary (Dict[str, Any])
        :param ner_threshold: threshold which must be met for the type to be returned (does not affect model - only
        has an effect on the return ner_type returned from this method)
        :param num_workers: the number of workers for the torch dataloader
        :param max_batch_size: max batch size
        :param prune_ner_types: returns only fine-grained (non-impliable) NER types (does not affect ED)
        :return: a list of DOCS (not spans) with predictions attached to each spans (where spans are sorted by indices)
        """
        raise NotImplementedError("This method has not been implemented yet.")

    def _prune_span_ner(self, span: Span) -> None:
        """
        Removes impliable classes from span (in-place).
        :param span: span
        """
        if span.pred_types is not None:
            # map Wikidata type to coarse spaCy type
            coarse_entity_types = [
                (wikidata_to_spacy_types[wikidata_id], conf)
                for wikidata_id, label, conf in span.pred_types
                if wikidata_id in wikidata_to_spacy_types
            ]
            coarse_entity_types.sort(key=lambda x: x[1], reverse=True)  # descending order
            if len(coarse_entity_types) > 0:
                span.coarse_mention_type = coarse_entity_types[0][0]

            # prune Wikidata types
            class_ids = [x[0] for x in span.pred_types]
            minimal_ids = self.preprocessor.prune_classes(frozenset(class_ids))
            span.pred_types = [
                type_pred for type_pred in span.pred_types if type_pred[0] in minimal_ids
            ]

            # override coarse type with pruned spaCy type
            coarse_entity_types = [
                (wikidata_to_spacy_types[wikidata_id], conf)
                for wikidata_id, label, conf in span.pred_types
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
            data_dir: str,
            debug: bool = False,
            device: Optional[str] = None,
            use_precomputed_descriptions: bool = True,
            requires_redirects_and_disambig: bool = False,
            download_files: bool = False
    ):
        """
        Load a pretrained ReFinED model.
        :param model_name: model name (e.g. wikipedia_model)
        :param data_dir: data directory path (by default will download files when needed from S3)
        :param debug: debug mode (loads data partially to speed up debugging)
        :param device: gpu or cpu device for model.
        :param use_precomputed_descriptions: use_precomputed_descriptions
        :param requires_redirects_and_disambig: requires_redirects_and_disambiguation lookups
        :return: `Refined` processor which wraps ReFinED model and provides easy to use methods.
        """
        model_files = get_model_files(model_name=model_name)

        # Download the model files
        if download_files:
            resource_manager = ResourceManager(s3_manager=S3Manager(), data_dir=data_dir)
            resource_manager.download_models_if_needed(model_name=model_name)

        assert entity_set in ["wikipedia", "wikidata"], f"entity_set should be one of ['wikipedia', 'wikidata'] but " \
                                                        f"is {entity_set}"
        return cls(
            model_file=os.path.join(data_dir, model_files["model"]["local_filename"]),
            model_config_file=os.path.join(data_dir, model_files["model_config"]["local_filename"]),
            model_description_embeddings_file=os.path.join(
                data_dir, model_files[f"model_description_embeddings_{entity_set}"]["local_filename"]
            ),
            entity_set=entity_set,
            data_dir=data_dir,
            device=device,
            use_precomputed_descriptions=use_precomputed_descriptions,
            load_descriptions_tns=not use_precomputed_descriptions,
            debug=debug,
            requires_redirects_and_disambig=requires_redirects_and_disambig,
            download_files=download_files
        )
