import argparse
import logging
import os
import sys
from collections import defaultdict
from typing import Dict

import ujson as json
from tqdm.auto import tqdm

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
        "--labels_file",
        type=str,
        default='output/label.json',
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
    with open(args.chosen_classes_file, 'r') as f:
        chosen_classes = {l.rstrip('\n') for l in f.readlines()}

    labels = load_labels(args.labels_file, False)
    cls_to_label: Dict[str, str] = dict()
    for cls in chosen_classes:
        if '<' in cls:
            relation = cls.split(',')[0][1:]
            object_qcode = cls.split(',')[1][:-1]
            if object_qcode in labels:
                object_qcode = labels[object_qcode]
            cls_to_label[cls] = f'<{relation},{object_qcode}>'
        else:
            if cls in labels:
                cls_to_label[cls] = labels[cls]
            else:
                cls_to_label[cls] = cls
    with open(f'{args.output_dir}/class_to_label.json', 'w') as f:
        json.dump(cls_to_label, f)

    logging.info('Written class to label')


def load_labels(file_path: str, is_test: bool):
    labels: Dict[str, str] = defaultdict(str)
    with open(file_path, 'r') as f:
        for i, line in tqdm(enumerate(f), total=80e+6, desc='Loading Wikidata labels'):
            line = json.loads(line)
            labels[line['qcode']] = line['values']
            if is_test and i > 100000:
                break
    return labels


if __name__ == '__main__':
    main()
