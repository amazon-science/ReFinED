import argparse
import logging
import os
import sys
from typing import Dict, Set, Any, List, Optional

import torch
import ujson as json
from tqdm.auto import tqdm

import numpy as np

from offline_preprocessing.dataclasses_for_preprocessing import AdditionalEntity
from utilities.lookup_utils import load_subclasses, load_instance_of, load_occuptations, load_sports, load_country, \
    ClassExplorer, load_qcode_to_idx

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
LOG = logging.getLogger(__name__)
EXCLUDE_LIST_AND_DISAMBIGUATION_PAGES = True


def main():
    parser = argparse.ArgumentParser(description='Process cleaned Wikipedia, extract links, merge files.')
    parser.add_argument(
        "--chosen_classes_file",
        type=str,
        default='3_august_chosen_classes.txt',
        help="File path for classes txt (classes separated by newline)."
    )
    parser.add_argument(
        "--pem_file",
        type=str,
        default='fixed_pem/wiki_pem.json',
        help="File path for pem."
    )
    parser.add_argument(
        "--occupation_file",
        type=str,
        default='output/occupation_p106.json',
        help="File path for occupations."
    )
    parser.add_argument(
        "--sport_file",
        type=str,
        default='output/sport_p641.json',
        help="File path for occupations."
    )
    parser.add_argument(
        "--country_file",
        type=str,
        default='output/country_p17.json',
        help="File path for occupations."
    )
    parser.add_argument(
        "--instance_of_file",
        type=str,
        default='output/instance_of_p31.json',
        help="File path for occupations."
    )
    parser.add_argument(
        "--subclass_file",
        type=str,
        default='output/subclass_p279.json',
        help="File path for occupations."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='output',
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
    args = parser.parse_args()
    args.output_dir = args.output_dir.rstrip('/')
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use "
                         f"--overwrite_output_dir to overwrite.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    class_to_idx: Dict[str, int] = create_class_to_idx(args.chosen_classes_file)
    chosen_classes: Set[str] = set(class_to_idx.keys())

    # qcode_to_idx: Dict[str, int] = create_qcode_to_idx(args.pem_file, args.test)

    print('len(qcode_to_idx)', len(qcode_to_idx))

    subclasses, _ = load_subclasses(args.subclass_file, args.test)
    instance_of = load_instance_of(args.instance_of_file, args.test)
    occupations = load_occuptations(args.occupation_file, args.test)
    sports = load_sports(args.sport_file, args.test)
    country = load_country(args.country_file, args.test)
    class_explorer = ClassExplorer(subclasses)

    # get max length to determine size of tensor
    i = 0
    max_classes_ln = 0
    for qcode, qcode_idx in tqdm(qcode_to_idx.items(), desc='Determining max number of classes'):
        classes = get_qcode_classes(qcode, occupations, sports, country, instance_of, class_explorer, chosen_classes,
                                    subclasses=subclasses)
        classes_idx = [class_to_idx[c] for c in classes]
        if len(classes_idx) > max_classes_ln:
            max_classes_ln = len(classes_idx)
        i += 1
        if i > 100 and args.test:
            break

    qcode_to_class_idx: torch.Tensor = torch.zeros(size=(len(qcode_to_idx) + 2, max_classes_ln + 2), dtype=torch.long)
    print(qcode_to_class_idx.shape)

    # fill tensor
    has_class = 0
    no_class = 0
    for qcode, qcode_idx in tqdm(qcode_to_idx.items(), desc='Filling qcode_to_class_idx tensor'):
        classes = get_qcode_classes(qcode, occupations, sports, country, instance_of, class_explorer, chosen_classes,
                                    subclasses=subclasses)
        classes_idx = [class_to_idx[c] for c in classes]
        qcode_to_class_idx[qcode_idx, :len(classes_idx)] = torch.tensor(classes_idx)
        if len(classes_idx) > 0:
            has_class += 1
        else:
            no_class += 1
        i += 1
        if i > 100 and args.test:
            break

    print(f'Has class: {has_class}, no class: {no_class}, {has_class / (has_class + no_class) * 100}%')
    print('qcode_to_class_idx row 0-10', list(enumerate(qcode_to_class_idx[:10])))
    print('qcode_to_idx row 0-10', list(enumerate(list(qcode_to_idx.items())[:10])))
    print('class_to_idx row 0-10', list(enumerate(list(class_to_idx.items())[:10])))
    torch.save(qcode_to_class_idx, f'{args.output_dir}/qcode_to_class_tns.pt')
    # with open(f'{args.output_dir}/qcode_to_idx.json', 'w') as f:
    #     json.dump(qcode_to_idx, f)
    with open(f'{args.output_dir}/class_to_idx.json', 'w') as f:
        json.dump(class_to_idx, f)


def create_tensors(resources_dir: str, additional_entities: Optional[List[AdditionalEntity]] = None,
                   is_test: bool = False):
    class_to_idx: Dict[str, int] = create_class_to_idx(os.path.join(resources_dir, 'chosen_classes.txt'))
    chosen_classes: Set[str] = set(class_to_idx.keys())

    qcode_to_idx = load_qcode_to_idx(os.path.join(resources_dir, 'qcode_to_idx.json'))
    # qcode_to_idx: Dict[str, int] = create_qcode_to_idx(os.path.join(resources_dir, 'wiki_pem.json'), is_test=is_test)
    print('len(qcode_to_idx)', len(qcode_to_idx))

    subclasses, _ = load_subclasses(os.path.join(resources_dir, 'subclass_p279.json'), is_test=is_test)
    instance_of = load_instance_of(os.path.join(resources_dir, 'instance_of_p31.json'), is_test=is_test)
    occupations = load_occuptations(os.path.join(resources_dir, 'occupation_p106.json'), is_test=is_test)
    sports = load_sports(os.path.join(resources_dir, 'sport_p641.json'), is_test=is_test)
    country = load_country(os.path.join(resources_dir, 'country_p17.json'), is_test=is_test)
    class_explorer = ClassExplorer(subclasses)

    # add entity types of additional entities
    if additional_entities is not None:
        print('Adding entity types for additional entities')
        for additional_entity in additional_entities:
            instance_of[additional_entity.entity_id] = set(additional_entity.entity_types)

    # get max length to determine size of tensor
    i = 0
    max_classes_ln = 0
    for qcode, qcode_idx in tqdm(qcode_to_idx.items(), desc='Determining max number of classes'):
        classes = get_qcode_classes(qcode, occupations, sports, country, instance_of, class_explorer, chosen_classes,
                                    subclasses=subclasses)
        classes_idx = [class_to_idx[c] for c in classes]
        if len(classes_idx) > max_classes_ln:
            max_classes_ln = len(classes_idx)
        i += 1
        # if i > 100 and is_test:
        #     break

    qcode_to_class_idx: torch.Tensor = torch.zeros(size=(len(qcode_to_idx) + 2, max_classes_ln + 2), dtype=torch.int16)
    print(qcode_to_class_idx.shape)

    # fill tensor
    has_class = 0
    no_class = 0
    for qcode, qcode_idx in tqdm(qcode_to_idx.items(), desc='Filling qcode_to_class_idx tensor'):
        classes = get_qcode_classes(qcode, occupations, sports, country, instance_of, class_explorer, chosen_classes,
                                    subclasses=subclasses)
        classes_idx = [class_to_idx[c] for c in classes]
        qcode_to_class_idx[qcode_idx, :len(classes_idx)] = torch.tensor(classes_idx)
        if len(classes_idx) > 0:
            has_class += 1
        else:
            no_class += 1
        i += 1
        # if i > 100 and is_test:
        #     break

    print(f'Has class: {has_class}, no class: {no_class}, {has_class / (has_class + no_class) * 100}%')
    print('qcode_to_class_idx row 0-10', list(enumerate(qcode_to_class_idx[:10])))
    print('qcode_to_idx row 0-10', list(enumerate(list(qcode_to_idx.items())[:10])))
    print('class_to_idx row 0-10', list(enumerate(list(class_to_idx.items())[:10])))
    torch.save(qcode_to_class_idx, f'{resources_dir}/qcode_to_class_tns.pt')

    qcode_to_class_np = np.memmap(f"{resources_dir}/qcode_to_class_tns_{qcode_to_class_idx.size(0)}-{qcode_to_class_idx.size(1)}.np",
                                  shape=qcode_to_class_idx.size(),
                                  dtype=np.int16,
                                  mode='w+')
    qcode_to_class_np[:] = qcode_to_class_idx[:]
    qcode_to_class_np.flush()

    # qcode_to_class_idx.numpy().tofile(f'{resources_dir}/qcode_to_class_tns.np')
    # with open(f'{resources_dir}/qcode_to_idx.json', 'w') as f:
    #     json.dump(qcode_to_idx, f)
    with open(f'{resources_dir}/class_to_idx.json', 'w') as f:
        json.dump(class_to_idx, f)


def create_qcode_to_idx(pem_file: str, is_test: bool) -> Dict[str, int]:
    all_qcodes = set()
    with open(pem_file, 'r') as f:
        for i, line in tqdm(enumerate(f), total=10e+6, desc='create_qcode_to_idx'):
            line = json.loads(line)
            all_qcodes.update(set(line['qcode_probs'].keys()))
            if is_test and i > 100000:
                break
    return {qcode: qcode_idx + 1 for qcode_idx, qcode in enumerate(list(all_qcodes))}


def create_class_to_idx(chosen_classes_file: str) -> Dict[str, int]:
    chosen_classes = []
    with open(chosen_classes_file, 'r') as f:
        for line in f:
            chosen_classes.append(line.rstrip('\n'))
    return {chosen_class: cls_idx + 1 for cls_idx, chosen_class in enumerate(chosen_classes)}


def get_qcode_classes(qcode: str, occupations, sports, country, instance_of, class_explorer, chosen_classes: Set[str],
                      subclasses: Dict[str, Any]) \
        -> Set[str]:
    classes = set()
    # subclasses disabled as types due to inconcistent usage in Wikidata
    # if qcode in subclasses_dict:
    #     classes.update({f'<subclass_of,{super_class}>' for super_class in subclasses_dict[qcode]})

    if qcode in occupations:
        # no brackets means subclasses will be used
        classes.update({super_class for super_class in occupations[qcode]})

    if qcode in sports:
        classes.update({f'<sports,{super_class}>' for super_class in sports[qcode]})

    if qcode in country:
        classes.update({f'<country,{super_class}>' for super_class in country[qcode]})

    if qcode in instance_of:
        classes.update(instance_of[qcode])

    # if qcode is a subclass then it implicitly is a class
    if qcode in subclasses:
        classes.add('Q16889133')

    return set(class_explorer.get_implied_classes(frozenset(classes))) & chosen_classes


if __name__ == '__main__':
    main()
