import json
import os

import torch
from torch.cpu.amp import autocast

from refined.utilities.md_dataset_utils import (
    bio_to_offset_pairs,
    create_collate_fn,
    tokenize_and_preserve_labels
)
from transformers import AutoModelForTokenClassification, AutoTokenizer
from typing import List, Dict

from refined.utilities.general_utils import batch_items


class Token:
    def __init__(self, word_piece, token_id, start, end):
        self.word_piece = word_piece
        self.token_id = token_id
        self.start = start
        self.end = end

    def __repr__(self):
        return json.dumps(self.__dict__)


class MentionDetector:
    def __init__(
        self,
        model_dir: str,
        ner_tag_to_num: Dict[str, int],
        transformer_name: str = "roberta-base",
        device: str = "cpu",
        max_seq: int = 510,
        n_gpu: int = 1,
    ):
        self.max_seq = max_seq
        self.transformer_name = transformer_name
        self.ner_tag_to_num = ner_tag_to_num
        self.num_labels = len(ner_tag_to_num)

        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            transformer_name,
            num_labels=self.num_labels,
            output_attentions=False,
            output_hidden_states=True,
        )

        model_path = os.path.join(model_dir, "model.pt")
        checkpoint = torch.load(model_path, map_location="cpu")

        self.num_to_ner_tag = {v: k for k, v in self.ner_tag_to_num.items()}

        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(device)
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(
                self.model, device_ids=list(range(n_gpu)), output_device=device
            ).to(device)

        self.n_gpu = n_gpu
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.device = device
        self.collate = create_collate_fn(self.tokenizer.pad_token_id)

    def process_text(self, text: str):
        # will show warning for long test.
        token_res = self.tokenizer.encode_plus(
            text,
            return_offsets_mapping=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            add_special_tokens=False
        )
        input_ids = token_res["input_ids"]
        offset_mappings = token_res["offset_mapping"]

        token_preds = self._predict_from_ids(input_ids=input_ids)

        token_preds = [self.num_to_ner_tag[num] for num in token_preds]

        bio_spans = bio_to_offset_pairs(token_preds, use_labels=True)
        spans = []
        for start_idx, end_idx, ner_type in bio_spans:
            doc_offset_start = offset_mappings[start_idx][0]
            doc_offset_end = offset_mappings[end_idx - 1][1]
            spans.append((doc_offset_start, doc_offset_end, text[doc_offset_start:doc_offset_end], ner_type))
        spans.sort(key=lambda x: x[0])
        return spans

    def process_words(self, words: List[str]) -> List[str]:

        # Tokenize
        tokens, _, mapping_to_words = tokenize_and_preserve_labels(words=words, text_labels=None,
                                                                   tokenizer=self.tokenizer)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Get MD/NER predictions for each token
        token_preds = self._predict_from_ids(input_ids=token_ids)

        assert len(token_preds) == len(token_ids), "Should have a single prediction per token id"

        token_preds = [self.num_to_ner_tag[num] for num in token_preds]

        # Map predictions back to original words
        word_preds = ['O' for _ in words]

        for token_ix, token in enumerate(token_preds):
            word_ix = mapping_to_words[token_ix]

            # If we have multiple predictions for the same word (i.e. when a word was split into two tokens during
            # tokenization) then only use the first prediction for the word
            if token_ix > 0 and mapping_to_words[token_ix - 1] == word_ix:
                continue

            word_preds[word_ix] = token

        return word_preds

    def _predict_from_ids(self, input_ids: List[int]):

        input_ids_batches = list(batch_items(input_ids, n=self.max_seq))
        token_preds = []
        for token_ids_grouped in batch_items(input_ids_batches, n=48):
            token_ids_grouped = [[self.cls_id] + x + [self.sep_id] for x in token_ids_grouped]
            input_ids_tns = [torch.tensor(token_ids_b) for token_ids_b in token_ids_grouped]
            labels_tns = [torch.zeros(len(token_ids_b)) for token_ids_b in token_ids_grouped]
            batch = self.collate(list(zip(input_ids_tns, labels_tns)))
            if self.n_gpu == 1 or True:
                batch = (tns.to(self.device) for tns in batch)

            tokens, attention_mask, labels = batch
            if hasattr(torch, "inference_mode"):
                inference_mode = torch.inference_mode
            else:
                inference_mode = torch.no_grad
            with inference_mode():
                # may need to pass pointers back to text if batching different texts
                # or could remember offsets of multiple texts and batch in order
                with autocast():
                    output = self.model(input_ids=tokens, attention_mask=attention_mask)
                preds = output[0]
                tokens_is_special = [
                    x == self.cls_id or x == self.sep_id for y in token_ids_grouped for x in y
                ]
                mask = attention_mask.flatten() == 1.0
                token_pred_with_special = (
                    preds.argmax(dim=2).flatten()[mask].detach().cpu().numpy().tolist()
                )
                token_pred = [
                    token
                    for token, is_spec in zip(token_pred_with_special, tokens_is_special)
                    if not is_spec
                ]
                token_preds.extend(token_pred)

        return token_preds

    @classmethod
    def init_from_pretrained(cls, model_dir: str, device: str = "cpu", n_gpu: int = 1) -> 'MentionDetector':

        hyperparams_path = os.path.join(model_dir, "training_args.json")
        hyperparams = json.load(open(hyperparams_path, 'rb'))

        return cls(
            model_dir=model_dir,
            ner_tag_to_num=hyperparams['ner_tag_to_num'],
            transformer_name=hyperparams['transformer_name'],
            max_seq=hyperparams["max_seq"],
            device=device,
            n_gpu=n_gpu,
        )

