import os
from typing import List, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from refined.offline_data_generation.dataclasses_for_preprocessing import AdditionalEntity
from refined.resource_management.loaders import load_labels, load_wikipedia_to_qcode, load_descriptions, \
    load_qcode_to_idx
from refined.training.train.training_args import parse_training_args
training_args = parse_training_args()

# TODO FIX THIS SO IT USES CORRECT QCODE_TO_IDX
def create_description_tensor(output_path: str, qcode_to_idx_filename: str, desc_filename: str, label_filename: str,
                              wiki_to_qcode: str, tokeniser: str = 'bert-base-multilingual-cased', is_test: bool = False,
                              include_no_desc: bool = True, keep_all_entities: bool = False,
                              additional_entities: Optional[List[AdditionalEntity]] = None):
    qcodes = {qcode for qcode in load_wikipedia_to_qcode(wiki_to_qcode).values()}
    labels = load_labels(label_filename, qcodes=qcodes, keep_all_entities=keep_all_entities, is_test=is_test)
    descriptions = load_descriptions(desc_filename, qcodes=qcodes, keep_all_entities=keep_all_entities, is_test=is_test)
    qcode_to_idx = load_qcode_to_idx(qcode_to_idx_filename)
    tokeniser = training_args.transformer_name
    
    if additional_entities is not None:
        print('Adding labels and descriptions from additional_entities')
        for additional_entity in additional_entities:
            labels[additional_entity.entity_id] = additional_entity.label
            descriptions[additional_entity.entity_id] = additional_entity.description

    # TODO: check no extra [SEP] tokens between label and description or extra [CLS] or [SEP] at end
    tokenizer = AutoTokenizer.from_pretrained(tokeniser, use_fast=True, add_prefix_space=False)
    descriptions_tns = torch.zeros((len(qcode_to_idx) + 2, 32), dtype=torch.int32)
    descriptions_tns.fill_(tokenizer.pad_token_id)

    qcode_has_label = 0
    qcode_has_desc = 0
    i = 0
    for qcode, idx in tqdm(qcode_to_idx.items()):
        if qcode in labels:
            qcode_has_label += 1
            label = labels[qcode]
            if qcode in descriptions and descriptions[qcode] is not None:
                qcode_has_desc += 1
                desc = descriptions[qcode]
            else:
                if not include_no_desc:
                    continue
                desc = 'no description'

            sentence = (label, desc)
            tokenised = tokenizer.encode_plus(sentence, truncation=True, max_length=32, padding='max_length',
                                              return_tensors='pt')['input_ids']
            descriptions_tns[idx] = tokenised
        i += 1
        if i % 250000 == 0:
            print(f'QCodes processed {i}, Qcodes with label: {qcode_has_label}, '
                  f'Qcodes with label and description: {qcode_has_desc}')

    torch.save(descriptions_tns, os.path.join(output_path, 'descriptions_tns.pt'))
