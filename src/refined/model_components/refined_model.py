import io
from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable, List, Optional, Tuple, Union, Dict

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from refined.utilities.md_dataset_utils import bio_to_offset_pairs
from refined.data_types.modelling_types import BatchElement, BatchedElementsTns, ModelReturn
from refined.data_types.base_types import Span
from refined.doc_preprocessing.preprocessor import Preprocessor
from refined.utilities.preprocessing_utils import pad
from refined.model_components.config import ModelConfig
from refined.model_components.ed_layer_2 import EDLayer
from refined.model_components.entity_disambiguation_layer import EntityDisambiguation
from refined.model_components.entity_typing_layer import EntityTyping
from refined.model_components.mention_detection_layer import MentionDetection
from refined.utilities.model_utils import fill_tensor
from refined.utilities.general_utils import get_logger

LOG = get_logger(__name__)


class RefinedModel(nn.Module):
    """
    Model for MD, ET, ED built on top of a transformer.
    """

    def __init__(
            self,
            config: ModelConfig,
            preprocessor: Preprocessor,
            use_precomputed_descriptions: bool = False,
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
        self.use_precomputed_descriptions = use_precomputed_descriptions
        self.detach_ed_layer = config.detach_ed_layer
        self.ner_tag_to_ix = config.ner_tag_to_ix
        self.ix_to_ner_tag = {ix: tag for tag, ix in self.ner_tag_to_ix.items()}
        self.preprocessor = preprocessor

        self.num_classes = self.preprocessor.num_classes + 1
        self.only_ner = config.only_ner
        self.only_ed = config.only_ed

        # instance variables
        self.transformer_config = deepcopy(preprocessor.transformer_config)

        # sub-modules
        self.mention_detection: nn.Module = MentionDetection(
            num_labels=len(config.ner_tag_to_ix),
            dropout=config.md_layer_dropout,
            hidden_size=self.transformer_config.hidden_size
        )
        self.mention_embedding_dropout = nn.Dropout(config.ner_layer_dropout)

        self.entity_typing: nn.Module = EntityTyping(
            dropout=config.ner_layer_dropout,
            num_classes=self.num_classes,
            encoder_hidden_size=self.transformer_config.hidden_size
        )
        self.entity_disambiguation: nn.Module = EntityDisambiguation(
            dropout=config.ed_layer_dropout,
            num_classes=self.num_classes,
            ignore_descriptions=ignore_descriptions,
            ignore_types=ignore_types
        )

        self.init_weights()

        # restores common weights for description encoder layers
        # description bi encoder
        self.ed_2: nn.Module = EDLayer(
            mention_dim=self.transformer_config.hidden_size, preprocessor=preprocessor
        )

        # restore common weights for context encoder layers
        self.transformer: PreTrainedModel = preprocessor.get_transformer_model()

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
        self.config = config

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
            batch: BatchedElementsTns,
            batch_elements_included: Optional[Tensor] = None
    ) -> ModelReturn:
        """
        Full forward pass including transformer, MD, ET, ED layers.
        :param batch: BatchedElementsTns is a named tuple containg all the tensors needed for the model
            :token_ids: token ids for tokens for BERT. Shape = (bs, seq_len)
            :token_acc_sums: accumulated sum that increments when the new span starts or ends. Shape = (bs, seq_len)
            :entity_mask: mask to determine which accumulated sum values represent mention spans.
              Shape = (bs, num_ents)
            :class_targets: indices for the gold classes for the mention spans. Shape = (bs, num_ents, num_classes)
            :attention_mask: the attention mask for BERT (should be 1 for all tokens and 0 for [PAD] tokens).
              Shape = (bs, seq_len)
            :token_type_ids: token type ids for BERT (should be 1 for all tokens). Shape = (bs, seq_len)
            :candidate_entity_targets: gold entity for each entity mention (max_candidates) means not in top k pem.
              Shape = (bs, num_ents, max_candidate + 1)
            :candidate_pem_values: the P(e|m) for each entity mention for each candidate entity.
              Shape = (bs, num_ents, max_candidate)
            :candidate_classes: indices for the classes for each entity mention for each candidate entity.
              Shape = (bs, num_ents, max_candidate , num_classes)
            :entity_index_mask_values: index mask select all mention predictons and gold values across the batch
              (loses the batch dimension (dim=0)). Shape = (bs, num_ents)
            :ner_labels:  NER (coarse) labels for tokens (length matches contextualised_embeddings, including special tokens)
            :batch_elements:  batch_elements (used when doing E2E EL)
            :spans: spans optional
            :cand_ids: candidates_qcodes optional
            :cand_desc: candidate entity descriptions optional
        :param batch_elements_included: Provided from `DataParallelReFinED`. It is a long tensor with shape (bs, 1) where the values indicate which batch elements
                                            are included in the current batch. This is only needed when DataParallel is
                                            used because DataParallel does not split lists it only splits tensors along
                                            the batch dimension.
        :return: ModelReturn is a named tuple with elements.
          (md_loss, md_activations, et_loss, et_activations, ed_loss, ed_activations, spans, cands).
        """
        if batch_elements_included is not None:
            include_indices = set(batch_elements_included.flatten().detach().cpu().numpy().tolist())
            batch_elements = [b for idx, b in enumerate(batch.batch_elements) if idx in include_indices]
        else:
            batch_elements = batch.batch_elements

        current_device = str(batch.token_id_values.device)

        # forward pass of transformer's (e.g. BERT) layers
        # unpacking does not work for object
        output: BaseModelOutputWithPoolingAndCrossAttentions = self.transformer(
            input_ids=batch.token_id_values,
            attention_mask=batch.attention_mask_values,
            token_type_ids=batch.token_type_values,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
        )
        contextualised_embeddings = output.last_hidden_state

        # forward pass of mention detection layer
        md_loss, md_activations = self.mention_detection(
            contextualised_embeddings=contextualised_embeddings, ner_labels=batch.ner_labels
        )

        # prepare tensors for ET and ED layers
        if batch.token_acc_sum_values is None:
            # mention-entity spans are not provided so the result from md layer must be used to determine spans
            token_acc_sums, entity_mask, entity_spans, other_spans, candidate_tensors = self._identify_entity_mentions(
                attention_mask=batch.attention_mask_values,
                batch_elements=batch_elements,
                device=current_device,
                md_activations=md_activations,
                max_seq=batch.token_id_values.size(1)
            )
            if len(entity_spans) == 0:
                # no point in continuing
                num_ents = 0
                return ModelReturn(
                    md_loss=md_loss,
                    md_activations=md_activations,
                    et_loss=None,
                    et_activations=torch.zeros([num_ents, self.num_classes], device=current_device),
                    ed_loss=None,
                    ed_activations=torch.zeros([num_ents, self.preprocessor.max_candidates + 1], device=current_device),
                    entity_spans=[],
                    other_spans=other_spans,
                    cand_ids=torch.zeros([num_ents, self.preprocessor.max_candidates], device=current_device,
                                         dtype=torch.long),
                    description_loss=None,
                    candidate_description_scores=torch.zeros([num_ents, self.preprocessor.max_candidates + 1],
                                                             device=current_device),
                )
                # return ModelReturn(md_loss, md_activations, None, None, None, None, [], other_spans, None, None, None)
            (
                cand_ids,
                candidate_pem_values,
                candidate_classes,
                cand_desc,
                cand_desc_emb,
            ) = candidate_tensors
            candidate_entity_targets = batch.candidate_target_values
        else:
            # token_acc_sums, entity_mask, entity_spans, other_spans, candidate_tensors
            token_acc_sums = batch.token_acc_sum_values
            entity_mask = batch.entity_mask_values

            # expand the tensors for the predetermined mention-entity spans
            expandable_args = (
                batch.candidate_qcode_values,
                batch.pem_values,
                batch.candidate_target_values,
                batch.candidate_desc,
                batch.candidate_desc_emb,
            )
            expanded_args = self._expand_tensors(
                expandable_args, index_tensor=batch.entity_index_mask_values
            )
            (
                cand_ids,
                candidate_pem_values,
                candidate_entity_targets,
                cand_desc,
                cand_desc_emb
            ) = expanded_args
            candidate_classes = self._expand_candidates_classes_tensor(
                candidates_classes=batch.candidate_class_values,
                index_tensor=batch.entity_index_mask_values,
                device=current_device,
            )
            # cand_desc_emb = None

            # At present none of the evaluation ER datasets provide DATE spans, so all spans if they are provided by
            # the dataset are standard entity spans
            # TODO: may want to change this in the future so special spans can be provided
            entity_spans = [span for b in batch_elements for span in b.spans]
            other_spans = {}

        class_targets = self._expand_class_targets(
            batch.class_target_values, index_tensor=batch.entity_index_mask_values
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
            candidate_description_scores=candidate_description_scores.detach(),  # detach or not
            current_device=current_device,
        )

        return ModelReturn(
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
            max_seq: int
    ):
        """
        Note that this add spans to batch_elements in-place.
        :param attention_mask: attention mask
        :param batch_elements: batch_elements
        :param device: device
        :param md_activations: md_activations
        :param max_seq: maximum token sequence length in batch (used for padding tensors) - see comment below for
                        explanation about why this needs to be passed in.
        :return: acc_sums, b_entity_mask, spans, candidate_tensors
        """
        person_coreference = dict()

        # TODO: this can be optimized by tensorizing some of the steps such as pem lookups.
        spans: List[Span] = []
        special_type_spans: Dict[str, List[Span]] = defaultdict(list)

        # (bs, max_seq_ln) - includes [SEP],[1:] removes [CLS]
        # very small tensor (e.g. (bs, max_seq) and simple operations so fine on CPU)
        bio_preds = (md_activations.argmax(dim=2) * attention_mask)[:, 1:].detach().cpu().numpy()
        prev_page_title = None
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
                    coarse_mention_type=coarse_type if coarse_type != 'MENTION' else None,
                    doc_id=batch_elem.doc_id
                )
                if coarse_type == "MENTION":
                    spans_for_batch.append(span)
                else:
                    # Other spans (e.g. "DATE" spans)
                    special_type_spans_for_batch[coarse_type].append(span)

            spans_for_batch.sort(key=lambda x: x.start)
            for type_spans in special_type_spans_for_batch.values():
                type_spans.sort(key=lambda x: x.start)

            # Can return person coref and pass it to next batch so coref trick can be used for long pages,
            # assumes same pages passed in order within same batch.
            # Do not do person name co-reference across documents only do it within documents,
            # batch_elem.text contains the full original text for the document where the batch element comes from.
            # Note that if document chunks are interleaved then the person co-reference will not work.
            # It assumes documents are converted to batch_elements() sequentially.
            # TODO: use a dictionary to keep track of title -> person_name_coref instead of refreshing each time.
            current_page_title = batch_elem.text[:20]
            if prev_page_title is not None:
                if current_page_title != prev_page_title:
                    person_coreference = dict()
            prev_page_title = current_page_title
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

        # NOTE THAT THIS IS NOT THE MAX_SEQ FOR THE BATCH IF MULTI_GPU DataParallel WAS USED.
        # Use token_id.size(1) instead!
        # max_seq = max([len(batch_elem.tokens) + 2 for batch_elem in batch_elements])

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
        candidate_qcodes: List[str] = []
        candidate_qcodes_ints: List[List[int]] = []
        for batch_elem in batch_elements:
            for span in batch_elem.spans:
                pem_values.append(
                    [pem_value for _, pem_value in span.candidate_entities]
                )  # TODO unpad and pad here
                candidate_qcodes.extend(
                    [qcode for qcode, _ in span.candidate_entities]
                )  # should pad here
                # temporary hack (use negative IDs for additional entities IDs to avoid
                # collisions with Wikdata IDs
                candidate_qcodes_ints.append(
                    [int(qcode.replace("Q", "")) if 'Q' in qcode else int(qcode.replace("A", '-')) for qcode, _ in
                     span.candidate_entities]
                )

        num_cands = self.preprocessor.max_candidates
        cand_class_idx = self.preprocessor.get_classes_idx_for_qcode_batch(
            candidate_qcodes, shape=(num_ents, num_cands, -1)
        )

        if self.use_precomputed_descriptions:
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
        b_candidate_qcode_values = torch.tensor(
            candidate_qcodes_ints, device=device, dtype=torch.long
        )

        candidate_tensors = (
            b_candidate_qcode_values,
            b_pem_values,
            b_candidate_classes,
            b_cand_desc,
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
            preprocessor: Preprocessor,
            use_precomputed_descriptions: bool = True
    ):
        """
        Load a pretrained model.
        :param model_file: path to model file
        :param model_config_file: path to model config file
        :param preprocessor: preprocessor with lookups
        :param use_precomputed_descriptions: use precomputed embeddings
        :return: model
        """
        config = ModelConfig.from_file(model_config_file, data_dir=preprocessor.data_dir)

        model = cls(
            config=config,
            preprocessor=preprocessor,
            use_precomputed_descriptions=use_precomputed_descriptions
        )
        with open(model_file, "rb") as f:
            checkpoint = torch.load(io.BytesIO(f.read()), map_location="cpu")

        # ensure code backwards compatible
        ed_params = checkpoint['entity_disambiguation.classifier.weight']
        if ed_params.size(-1) == 1378:
            # old model checkpoint had weights for additional features
            # should be [1, 1372]
            # pme + pem
            ed_params[:, -9] += ed_params[:, -7]
            mask = torch.ones_like(ed_params, dtype=torch.bool)
            # remove additional features weights
            mask[:, -5:] = False
            # remove pme feature
            mask[:, -7] = False
            ed_params = ed_params[mask].unsqueeze(0)
            checkpoint['entity_disambiguation.classifier.weight'] = ed_params

        model.load_state_dict(checkpoint, strict=False)

        return model
