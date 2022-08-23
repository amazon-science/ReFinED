import io
from typing import Any, Iterable, List, Optional, Tuple, Union, Dict
from collections import defaultdict

import smart_open
import torch
from dataclasses import astuple
from dataset_reading.mention_detection.md_dataset_utils import bio_to_offset_pairs
from doc_preprocessing.dataclasses import AdditionalEDFeatures, BatchElement, DocPreprocessor, Span
from doc_preprocessing.doc_preprocessor import DocumentPreprocessorMemoryBased
from doc_preprocessing.preprocessing_utils import pad
from model_components.config import ModelConfig
from model_components.ed_layer_2 import EDLayer
from model_components.entity_disambiguation_layer import EntityDisambiguation
from model_components.entity_typing_layer import EntityTyping
from model_components.mention_detection_layer import MentionDetection
from model_components.model_utils import fill_tensor
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from utilities.general_utils import get_huggingface_config, get_huggingface_model, get_logger

LOG = get_logger(__name__)


class RefinedModel(nn.Module):
    """
    Model for MD, ET, ED built on top of a transformer.
    """

    def __init__(
        self,
        config: ModelConfig,
        preprocessor: Optional[DocPreprocessor] = None,
        use_precomputed_descriptions: bool = False,
        use_kg_layer: bool = False,
        ignore_descriptions: bool = False,
        ignore_types: bool = False,
    ):
        """
        Constructs model for NER and ED.
        :param config: the config for the model
        :param preprocessor: preprocessor
        :param use_precomputed_descriptions: use precomputed descriptions when True
        :param ignore_descriptions: ignore descriptions in the scoring function (for ablation study)
        :param ignore_types: ignore types in the scoring function (for ablation study)
        """
        super().__init__()

        self.ignore_descriptions = ignore_descriptions
        self.ignore_types = ignore_types
        self.use_kg_layer = use_kg_layer
        self.use_precomputed_descriptions = use_precomputed_descriptions
        self.detach_ed_layer = config.detach_ed_layer
        self.ner_tag_to_ix = config.ner_tag_to_ix
        self.ix_to_ner_tag = {ix: tag for tag, ix in self.ner_tag_to_ix.items()}
        if preprocessor:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = DocumentPreprocessorMemoryBased(
                data_dir=config.data_dir,
                transformer_name=config.transformer_name,
                max_candidates=config.max_candidates,
                debug=config.debug,
            )
        self.num_classes = self.preprocessor.num_classes + 1
        self.only_ner = config.only_ner
        self.only_ed = config.only_ed

        # instance variables
        self.transformer_config = get_huggingface_config(
            transformer_name=config.transformer_name, data_dir=config.data_dir
        )

        # sub-modules
        self.mention_detection: nn.Module = MentionDetection(
            num_labels=len(config.ner_tag_to_ix),
            dropout=config.md_layer_dropout,
            hidden_size=self.transformer_config.hidden_size,
        )
        self.mention_embedding_dropout = nn.Dropout(config.ner_layer_dropout)

        self.entity_typing: nn.Module = EntityTyping(
            dropout=config.ner_layer_dropout,
            num_classes=self.num_classes,
            encoder_hidden_size=self.transformer_config.hidden_size,
        )
        self.entity_disambiguation: nn.Module = EntityDisambiguation(
            dropout=config.ed_layer_dropout,
            num_classes=self.num_classes,
            ignore_descriptions=ignore_descriptions,
            ignore_types=ignore_types,
        )

        self.init_weights()

        # restores common weights for description encoder layers
        # description bi encoder
        self.ed_2: nn.Module = EDLayer(
            mention_dim=self.transformer_config.hidden_size, data_dir=preprocessor.data_dir
        )

        # restore common weights for context encoder layers
        self.transformer: PreTrainedModel = get_huggingface_model(
            transformer_name=config.transformer_name, data_dir=config.data_dir
        )

        # freeze layers (if applicable)
        if config.freeze_all_bert_layers:
            for param in self.transformer.parameters():
                param.requires_grad = False
        if config.freeze_embedding_layers:
            for param in list(self.transformer.embeddings.parameters()):
                param.requires_grad = False
            LOG.debug(f"Froze BERT embedding layers (weights will be fixed during training)")

        # freeze_layers is list [0, 1, 2, 3] representing layer number
        for layer_idx in config.freeze_layers:
            for param in list(self.transformer.encoder.layer[layer_idx].parameters()):
                param.requires_grad = False
            LOG.debug(f"Froze encoder layer {layer_idx} (weights will be fixed during training)")

    def get_parameters_to_scale(self) -> List[nn.Parameter]:
        """
        Gets parameters that can be scaled during training. Usually top/last layers.
        :return: list of parameters
        """
        md_params = list(self.mention_detection.parameters())
        et_params = list(self.entity_typing.parameters())
        desc_parmas = list(self.ed_2.get_parameters_to_scale())
        ed_params = list(self.entity_disambiguation.parameters())
        return md_params + et_params + ed_params + desc_parmas + ed_params

    def get_md_params(self) -> List[nn.Parameter]:
        return list(self.mention_detection.parameters())

    def get_et_params(self) -> List[nn.Parameter]:
        return list(self.entity_typing.parameters())

    def get_desc_params(self) -> List[nn.Parameter]:
        return list(self.ed_2.get_parameters_to_scale())

    def get_ed_params(self) -> List[nn.Parameter]:
        return list(self.entity_disambiguation.parameters())

    def get_kg_params(self) -> List[nn.Parameter]:
        return list(self.kg_layer.parameters())

    def get_final_ed_params(self) -> List[nn.Parameter]:
        return list(self.final_ed_layer.parameters())

    def get_parameters_not_to_scale(self) -> List[nn.Parameter]:
        """
        Gets parameters that can be scaled during training. Usually top/last layers.
        :return: list of parameters
        """
        mention_transformer_params = list(self.transformer.parameters())
        entity_description_transformer_params = list(self.ed_2.get_parameters_not_to_scale())
        return mention_transformer_params + entity_description_transformer_params

    def init_weights(self):
        """Initialize weights for all member variables with type nn.Module"""
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.transformer_config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        token_ids: Tensor = None,
        attention_mask: Tensor = None,
        token_type_ids: Tensor = None,
        batch_elements: List[BatchElement] = None,
        token_acc_sums: Tensor = None,
        entity_mask: Tensor = None,
        class_targets: Tensor = None,
        candidate_entity_targets: Tensor = None,
        candidate_pem_values: Tensor = None,
        candidate_classes: Tensor = None,
        candidate_pme_values: Tensor = None,
        entity_index_mask_values: Tensor = None,
        ner_labels: Tensor = None,
        spans: List[Span] = None,
        cand_ids: Tensor = None,
        cand_desc: Tensor = None,
        cand_desc_emb: Tensor = None,
        candidate_features: Tensor = None,
        batch_elements_included: Tensor = None,
        md_only: bool = False
    ) -> Tuple[Any, Any, Any, Any, Any, Any, Union[Optional[List[Span]], Any], Any, Any, Any]:
        """
        Full forward pass including transformer, MD, ET, ED layers.
        :param token_ids: token ids for tokens for BERT. Shape = (bs, seq_len)
        :param token_acc_sums: accumulated sum that increments when the new span starts or ends. Shape = (bs, seq_len)
        :param entity_mask: mask to determine which accumulated sum values represent mention spans.
          Shape = (bs, num_ents)
        :param class_targets: indices for the gold classes for the mention spans. Shape = (bs, num_ents, num_classes)
        :param attention_mask: the attention mask for BERT (should be 1 for all tokens and 0 for [PAD] tokens).
          Shape = (bs, seq_len)
        :param token_type_ids: token type ids for BERT (should be 1 for all tokens). Shape = (bs, seq_len)
        :param candidate_entity_targets: gold entity for each entity mention (max_candidates) means not in top k pem.
          Shape = (bs, num_ents, max_candidate + 1)
        :param candidate_pem_values: the P(e|m) for each entity mention for each candidate entity.
          Shape = (bs, num_ents, max_candidate)
        :param candidate_classes: indices for the classes for each entity mention for each candidate entity.
          Shape = (bs, num_ents, max_candidate , num_classes)
        :param candidate_pme_values: the P(m|e) for each entity mention for each candidate entity.
          Shape = (bs, num_ents, max_candidate)
        :param entity_index_mask_values: index mask select all mention predictons and gold values across the batch
          (loses the batch dimension (dim=0)). Shape = (bs, num_ents)
        :param ner_labels:  NER (coarse) labels for tokens (length matches contextualised_embeddings, including special tokens)
        :param batch_elements:  batch_elements (used when doing E2E EL)
        :param spans: spans optional
        :param cand_ids: candidates_qcodes optional
        :param cand_desc: candidate entity descriptions optional
        :param candidate_features: Optional, candidate features. Shape = (bs, num_ents, 5)
        :param batch_elements_included: a long tensor with shape (bs, 1) where the values indicate which batch elements
                                        are included in the current batch. This is only needed when DataParallel is
                                        used because DataParallel does not split lists it only splits tensors along the
                                        batch dimension.
        :param md_only: if True, only run the MD element of the forward pass. Used during MD evaluation
        :return: ModelReturn is a named tuple with elements.
          (md_loss, md_activations, et_loss, et_activations, ed_loss, ed_activations, spans, cands).
        """
        if batch_elements_included is not None:
            include_indices = set(batch_elements_included.flatten().detach().cpu().numpy().tolist())
            batch_elements = [b for idx, b in enumerate(batch_elements) if idx in include_indices]

        current_device = str(token_ids.device)

        # forward pass of transformer's (e.g. BERT) layers
        # unpacking does not work for object
        output: BaseModelOutputWithPoolingAndCrossAttentions = self.transformer(
            input_ids=token_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
        )
        contextualised_embeddings = output.last_hidden_state

        # forward pass of mention detection layer
        md_loss, md_activations = self.mention_detection(
            contextualised_embeddings=contextualised_embeddings, ner_labels=ner_labels
        )

        if md_only:
            return md_loss, md_activations, None, None, None, None, None, None, None, None, None

        # prepare tensors for ET and ED layers
        if token_acc_sums is None:
            # mention-entity spans are not provided so the result from md layer must be used to determine spans
            token_acc_sums, entity_mask, entity_spans, other_spans, candidate_tensors = self._identify_entity_mentions(
                attention_mask=attention_mask,
                batch_elements=batch_elements,
                device=current_device,
                md_activations=md_activations,
            )
            if len(entity_spans) == 0:
                # no point in continuing
                return md_loss, md_activations, None, None, None, None, [], other_spans, None, None, None
            (
                cand_ids,
                candidate_pem_values,
                candidate_pme_values,
                candidate_classes,
                cand_desc,
                candidate_features,
                cand_desc_emb,
            ) = candidate_tensors

        else:
            # expand the tensors for the predetermined mention-entity spans
            expandable_args = (
                cand_ids,
                candidate_pem_values,
                candidate_pme_values,
                candidate_entity_targets,
                cand_desc,
                candidate_features,
                cand_desc_emb,
            )
            expanded_args = self._expand_tensors(
                expandable_args, index_tensor=entity_index_mask_values
            )
            (
                cand_ids,
                candidate_pem_values,
                candidate_pme_values,
                candidate_entity_targets,
                cand_desc,
                candidate_features,
                cand_desc_emb
            ) = expanded_args
            candidate_classes = self._expand_candidates_classes_tensor(
                candidates_classes=candidate_classes,
                index_tensor=entity_index_mask_values,
                device=current_device,
            )
            # cand_desc_emb = None

            # At present none of the evaluation ER datasets provide DATE spans, so all spans if they are provided by
            # the dataset are standard entity spans
            entity_spans = spans
            other_spans = {}

        class_targets = self._expand_class_targets(
            class_targets, index_tensor=entity_index_mask_values
        )

        mention_embeddings = self._get_mention_embeddings(
            sequence_output=contextualised_embeddings,
            token_acc_sums=token_acc_sums,
            entity_mask=entity_mask,
        )

        # candidate_description_scores.shape = (num_ents, num_cands)
        description_loss, candidate_description_scores = self.ed_2(
            candidate_desc=cand_desc,
            mention_embeddings=mention_embeddings,
            candidate_entity_targets=candidate_entity_targets,
            candidate_desc_emb=cand_desc_emb,
        )

        # forward pass of entity typing layer (using predetermined spans if provided else span identified by md layer)
        et_loss, et_activations = self.entity_typing(
            mention_embeddings=mention_embeddings, span_classes=class_targets
        )

        # forward pass of entity disambiguation layer
        ed_loss, ed_activations = self.entity_disambiguation(
            class_activations=et_activations.detach() if self.detach_ed_layer else et_activations,
            candidate_entity_targets=candidate_entity_targets,
            candidate_pem_values=candidate_pem_values,
            candidate_classes=candidate_classes,
            candidate_pme_values=candidate_pme_values,
            candidate_description_scores=candidate_description_scores.detach(),  # detach or not
            candidate_features=candidate_features,
            current_device=current_device,
        )

        return (
            md_loss,
            md_activations,
            et_loss,
            et_activations,
            ed_loss,
            ed_activations,
            entity_spans,
            other_spans,
            cand_ids,
            description_loss,
            candidate_description_scores,
        )

    def _get_mention_embeddings(
        self, sequence_output: Tensor, token_acc_sums: Tensor, entity_mask: Tensor
    ):
        sequence_output = self.mention_embedding_dropout(sequence_output)
        # [batch, seq_len]
        second_dim_ix = token_acc_sums
        # [batch, seq_len, embed_size]
        embeddings = sequence_output
        batch_size, seq_len, _ = embeddings.size()
        # Generate first dimension indices
        first_dim_ix = (
            torch.arange(batch_size, device=sequence_output.device)
            .unsqueeze(1)
            .expand(batch_size, seq_len)
        )
        indices = (first_dim_ix, second_dim_ix)
        # Max size to fill
        max_len = int(torch.max(second_dim_ix).int() + 1)
        # Mention lengths for averaging
        mention_length = torch.ones(size=(batch_size, seq_len, 1), device=sequence_output.device)
        mention_length = fill_tensor(
            max_len, indices, mention_length, accumulate=True, current_device=sequence_output.device
        )
        # Embeddings
        mention_embeddings = fill_tensor(
            max_len, indices, embeddings, accumulate=True, current_device=sequence_output.device
        )
        # mean pool the embeddings for each span
        mention_embeddings = mention_embeddings / mention_length
        # Class labels
        # Boolean mask for only extracting entities
        entity_mask = entity_mask[:, :max_len]
        boolean_mask = entity_mask != 0

        # Embeddings of entities only: [number_of_entities, embed_size]
        return mention_embeddings[boolean_mask]

    def _identify_entity_mentions(
        self,
        attention_mask: Tensor,
        batch_elements: List[BatchElement],
        device: str,
        md_activations: Tensor,
    ):
        """
        Note that this add spans to batch_elements in-place.
        :param attention_mask: attention mask
        :param batch_elements: batch_elements
        :param device: device
        :param md_activations: md_activations
        :return: acc_sums, b_entity_mask, spans, candidate_tensors
        """
        person_coreference = dict()

        # TODO: this can be optimized by tensorizing some of the steps such as pem lookups.
        spans: List[Span] = []
        special_type_spans: Dict[str, List[Span]] = defaultdict(list)

        # (bs, max_seq_ln) - includes [SEP],[1:] removes [CLS]
        # very small tensor (e.g. (bs, max_seq) and simple operations so fine on CPU)
        bio_preds = (md_activations.argmax(dim=2) * attention_mask)[:, 1:].detach().cpu().numpy()
        for batch_idx, batch_elem in enumerate(batch_elements):
            preds = [self.ix_to_ner_tag[p] for p in bio_preds[batch_idx].tolist()]
            bio_spans = bio_to_offset_pairs(preds, use_labels=True)
            spans_for_batch: List[Span] = []
            special_type_spans_for_batch: Dict[str, List[Span]] = defaultdict(list)
            for start_list_idx, end_list_idx, coarse_type in bio_spans:
                if end_list_idx > len(batch_elem.tokens):
                    # [SEP] or [PAD] token has been classified as part of span. Indicates the entity has been split
                    # so skip.
                    continue
                doc_char_start = batch_elem.tokens[start_list_idx].start
                doc_char_end = batch_elem.tokens[end_list_idx - 1].end
                span = Span(
                        start=doc_char_start,
                        ln=doc_char_end - doc_char_start,
                        text=batch_elem.text[doc_char_start:doc_char_end],
                        coarse_type=coarse_type,
                        coarse_mention_type=coarse_type if coarse_type != 'MENTION' else None
                    )
                if coarse_type == "MENTION":
                    spans_for_batch.append(span)
                else:
                    # Other spans (e.g. "DATE" spans)
                    special_type_spans_for_batch[coarse_type].append(span)

            spans_for_batch.sort(key=lambda x: x.start)
            for type_spans in special_type_spans_for_batch.values():
                type_spans.sort(key=lambda x: x.start)
            # Can return person coref and pass it to next batch so coref trick can be used for long pages
            # assumes same page passed in order within same batch
            person_coreference = self.preprocessor.add_candidates_to_spans(
                spans_for_batch, person_coreference=person_coreference
            )
            # TODO: do we need to add special type spans to batch element here?
            batch_elem.add_spans(spans_for_batch)
            spans.extend(spans_for_batch)

            for coarse_type, type_spans in special_type_spans_for_batch.items():
                special_type_spans[coarse_type] += type_spans

        num_ents = len([span for batch_elm in batch_elements for span in batch_elm.spans])
        if num_ents == 0:
            return None, None, [], special_type_spans, None
        max_seq = max([len(batch_elem.tokens) + 2 for batch_elem in batch_elements])

        acc_sums_lst = [
            [0] + list(map(lambda token: token.acc_sum, elem.tokens)) + [0]
            for elem in batch_elements
        ]
        acc_sums = torch.tensor(
            pad(acc_sums_lst, seq_len=max_seq, pad_value=0), device=device, dtype=torch.long
        )

        b_entity_mask_lst = [elem.entity_mask for elem in batch_elements]
        b_entity_mask = torch.tensor(
            pad(b_entity_mask_lst, seq_len=-1, pad_value=0), device=device, dtype=torch.long
        )

        pem_values: List[List[float]] = []
        pme_values: List[List[float]] = []
        candidate_qcodes: List[str] = []
        candidate_qcodes_ints: List[List[int]] = []
        candidate_features: List[List[Tuple[float, float, float, float]]] = []
        for batch_elem in batch_elements:
            for span in batch_elem.spans:
                pem_values.append(
                    [pem_value for _, pem_value in span.candidate_entity_ids]
                )  # TODO unpad and pad here
                pme_values.append(
                    [pem_value for _, pem_value in span.candidate_entity_ids]
                )  # TODO: remove pme
                candidate_qcodes.extend(
                    [qcode for qcode, _ in span.candidate_entity_ids]
                )  # should pad here
                # temporary hack (use negative IDs for additional entities IDs to avoid
                # collisions with Wikdata IDs
                candidate_qcodes_ints.append(
                    [int(qcode.replace("Q", "")) if 'Q' in qcode else int(qcode.replace("A", '-')) for qcode, _ in span.candidate_entity_ids]
                )
                if span.additional_ed_features is not None:
                    candidate_features.append([astuple(f) for f in span.additional_ed_features])

        num_cands = self.preprocessor.max_candidates
        cand_class_idx = self.preprocessor.get_classes_idx_for_qcode_batch(
            candidate_qcodes, shape=(num_ents, num_cands, -1)
        )

        if (
            self.use_precomputed_descriptions
            and self.preprocessor.precomputed_descriptions is not None
        ):
            b_cand_desc_emb = self.preprocessor.get_descriptions_emb_for_qcode_batch(
                candidate_qcodes, shape=(num_ents, num_cands, -1)
            ).to(device)
            b_cand_desc = None
        else:
            b_cand_desc_emb = None
            b_cand_desc = self.preprocessor.get_descriptions_for_qcode_batch(
                candidate_qcodes, shape=(num_ents, num_cands, -1)
            ).to(device)

        b_candidate_classes = torch.zeros(
            size=(num_ents, num_cands, self.num_classes), dtype=torch.float32, device=device
        )
        first_idx = (
            torch.arange(num_ents, device=device)
            .unsqueeze(1)
            .unsqueeze(1)
            .expand(cand_class_idx.size())
        )
        snd_idx = torch.arange(num_cands, device=device).unsqueeze(1)
        b_candidate_classes[first_idx, snd_idx, cand_class_idx] = 1
        b_pem_values = torch.tensor(pem_values, device=device, dtype=torch.float32)
        b_pme_values = torch.tensor(pme_values, device=device, dtype=torch.float32)
        b_candidate_qcode_values = torch.tensor(
            candidate_qcodes_ints, device=device, dtype=torch.long
        )
        if not self.preprocessor.zero_string_features:
            b_candidate_features = torch.tensor(
                candidate_features, dtype=torch.float32, device=device
            ).view((num_ents, num_cands, len(astuple(AdditionalEDFeatures.get_pad_element()))))
        else:
            b_candidate_features = torch.zeros(
                size=[num_ents, num_cands, len(astuple(AdditionalEDFeatures.get_pad_element()))],
                dtype=torch.float32,
                device=device,
            )

        candidate_tensors = (
            b_candidate_qcode_values,
            b_pem_values,
            b_pme_values,
            b_candidate_classes,
            b_cand_desc,
            b_candidate_features,
            b_cand_desc_emb,
        )
        return acc_sums, b_entity_mask, spans, special_type_spans, candidate_tensors

    @staticmethod
    def _expand_tensors(
        tensors: Iterable[Tensor], index_tensor: Tensor, device: Optional[str] = None
    ) -> Iterable[Optional[Tensor]]:
        """
        Expands tensors, A, using boolean tensor, B. Where B is a boolean tensor with B.shape = A.shape[:-1].
        For example, tensor A could represent candidate entity pem values and have the following shape:
        - (bs, num_ents (including padding), num_cands).
        Since the number of entities varies per sentence (bs dimension) some values in the second index should be
        removed. This method will take all the rows for non-masked entities across all sentences in the batch. The
        resulting shape will be (all_ents, num_cands). The batch dimension is lost and the remaining entities are
        non-masked.
        :param tensors: an iterable of input tensors
        :param index_tensor: an index tensor
        :param device: device (if not specificed the device of the tensor.device will be used)
        :return: iterable of expanded tensors where the order is maintained
        """
        result = []
        for tensor in tensors:
            if tensor is None or index_tensor is None:
                result.append(None)
            else:
                result.append(
                    tensor[index_tensor].to(device) if device is not None else tensor[index_tensor]
                )
        return tuple(result)

    def _expand_class_targets(
        self, class_targets: Tensor, index_tensor: Tensor
    ) -> Optional[Tensor]:
        if class_targets is None or index_tensor is None:
            return None
        class_targets = class_targets[index_tensor]
        num_ents = class_targets.size(0)
        device = class_targets.device
        class_targets_expanded = torch.zeros(
            size=(num_ents, self.num_classes), dtype=torch.float32, device=device
        )
        class_targets_expanded[
            torch.arange(num_ents).unsqueeze(1).expand(class_targets.size()), class_targets
        ] = 1
        return class_targets_expanded

    def _expand_candidates_classes_tensor(
        self, candidates_classes: Tensor, index_tensor: Tensor, device: str
    ) -> Optional[Tensor]:
        if candidates_classes is None or index_tensor is None:
            return None
        candidates_classes = candidates_classes[index_tensor]
        batch_size, num_ents, _ = candidates_classes.size()
        candidates_classes_expanded = torch.zeros(
            size=(batch_size, num_ents, self.num_classes), dtype=torch.float32, device=device
        )
        first_idx = (
            torch.arange(batch_size).unsqueeze(1).unsqueeze(1).expand(candidates_classes.size())
        )
        snd_idx = torch.arange(num_ents).unsqueeze(1)
        candidates_classes_expanded[first_idx, snd_idx, candidates_classes] = 1
        return candidates_classes_expanded

    @classmethod
    def from_pretrained(
        cls,
        model_file: str,
        model_config_file: str,
        preprocessor: DocPreprocessor,
        use_precomputed_descriptions: bool = True,
        description_embeddings_file: Optional[str] = None
    ):
        """
        Load a pretrained model.
        :param model_file: path to model file
        :param model_config_file: path to model config file
        :param preprocessor: preprocessor with lookups
        :param use_precomputed_descriptions: use precomputed embeddings
        :param description_embeddings_file: description_embeddings_file
        :return: model
        """
        description_embeddings_uri = description_embeddings_file
        config = ModelConfig.from_file(model_config_file, data_dir=preprocessor.data_dir)
        if use_precomputed_descriptions and description_embeddings_file is not None:
            preprocessor.set_precomputed_descriptions(description_embeddings_uri)

        model = cls(
            config=config,
            preprocessor=preprocessor,
            use_precomputed_descriptions=use_precomputed_descriptions
        )
        with smart_open.smart_open(model_file, "rb") as f:
            checkpoint = torch.load(io.BytesIO(f.read()), map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)

        return model
