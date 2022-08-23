import argparse
import os
from typing import List, Optional

import torch
import ujson as json
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import sys

from offline_preprocessing.dataclasses_for_preprocessing import AdditionalEntity

sys.path.append('.')
from utilities.lookup_utils import load_labels, load_wikipedia_to_qcode, load_descriptions, load_qcode_to_idx


def get_triples(entity):
    qcode = entity['id']
    triples = {}
    for pcode, objs in entity['claims'].items():
        # group by pcode -> [list of qcodes]
        for obj in objs:
            if not obj['mainsnak']['datatype'] == 'wikibase-item' or obj['mainsnak']['snaktype'] == 'somevalue' \
                    or 'datavalue' not in obj['mainsnak']:
                continue
            if pcode not in triples:
                triples[pcode] = []
            triples[pcode].append(obj['mainsnak']['datavalue']['value']['id'])
    return {'qcode': qcode, 'triples': triples}


# TODO FIX THIS SO IT USES CORRECT QCODE_TO_IDX
def create_description_tensor(output_path: str, qcode_to_idx_filename: str, desc_filename: str, label_filename: str,
                              wiki_to_qcode: str, tokeniser: str = 'roberta-base', is_test: bool = False,
                              include_no_desc: bool = True, keep_all_entities: bool = False,
                              additional_entities: Optional[List[AdditionalEntity]] = None):
    qcodes = {qcode for qcode in load_wikipedia_to_qcode(wiki_to_qcode).values()}
    labels = load_labels(label_filename, qcodes=qcodes, keep_all_entities=keep_all_entities, is_test=is_test)
    descriptions = load_descriptions(desc_filename, qcodes=qcodes, keep_all_entities=keep_all_entities, is_test=is_test)
    qcode_to_idx = load_qcode_to_idx(qcode_to_idx_filename)

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
        # if is_test and i > 1000:
        #     break

    torch.save(descriptions_tns, os.path.join(output_path, 'descriptions_tns.pt'))


def main():
    parser = argparse.ArgumentParser(description='Build lookup dictionaries from Wikidata JSON dump.')
    parser.add_argument(
        "--qcode_to_idx",
        default='/big_data/qcode_to_idx_2.json',
        type=str,
        help="file path to JSON Wikidata dump file (latest-all.json.bz2)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='output',
        help="Directory where the lookups will be stored"
    )

    parser.add_argument(
        "--descriptions_file",
        type=str,
        default='/big_data/wikidata/output/desc.json',
        help="Directory where the lookups will be stored"
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        default='/big_data/wikidata/output/label.json',
        help="Directory where the lookups will be stored"
    )
    parser.add_argument(
        "--wiki_to_qcode_file",
        type=str,
        default='/big_data/wikidata/output/enwiki.json',
        help="Directory where the lookups will be stored"
    )

    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="mode for testing (only processes first 500 lines)"
    )
    parser.add_argument(
        "--include_no_desc",
        action="store_true",
        help="include entities without descriptions if they have a label"
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.rstrip('/')
    number_lines_to_process = 500 if args.test else 1e20
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use "
                         f"--overwrite_output_dir to overwrite.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    wiki_to_qcode = load_wikipedia_to_qcode('/big_data/wikidata/output/enwiki.json')
    wiki_qcodes = {qcode for wiki_title, qcode in wiki_to_qcode.items()}

    labels = load_labels('/big_data/wikidata/output/label.json', qcodes=wiki_qcodes)
    with open(f'/big_data/qcode_to_idx_2.json', 'r') as f:
        qcode_to_idx = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True, add_prefix_space=False)
    descriptions_tns = torch.zeros((len(qcode_to_idx) + 2, 32), dtype=torch.long)
    descriptions_tns.fill_(tokenizer.pad_token_id)

    # read descriptions
    descriptions = dict()
    with open('/big_data/wikidata/output/desc.json', 'r') as f:
        i = 0
        for line in tqdm(f, total=80000000, desc='Loading descriptions'):
            line = json.loads(line)
            qcode = line['qcode']
            descriptions[qcode] = line['values']
            i += 1
            if args.test and i > 1000:
                break

    print('len(qcode_to_idx), ', len(qcode_to_idx))
    qcode_has_label = 0
    qcode_has_desc = 0
    i = 0
    for qcode, idx in qcode_to_idx.items():
        if qcode in labels:
            qcode_has_label += 1
            label = labels[qcode]
            if qcode in descriptions:
                qcode_has_desc += 1
                desc = descriptions[qcode]
            else:
                desc = 'no description'

            sentence = (label, desc)
            tokenised = tokenizer.encode_plus(sentence, truncation=True, max_length=32, padding='max_length',
                                              return_tensors='pt')['input_ids']
            descriptions_tns[idx] = tokenised
        i += 1
        if i % 25000 == 0:
            print(f'QCodes processed {i}, Qcodes with label: {qcode_has_label}, '
                  f'Qcodes with label and description: {qcode_has_desc}')
        if args.test and i > 1000:
            break

    torch.save(descriptions_tns, os.path.join(args.output_dir, 'descriptions_tns.pt'))


if __name__ == '__main__':
    main()
