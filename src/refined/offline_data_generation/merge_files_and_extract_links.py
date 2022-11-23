import argparse
import json
import os
import re
from typing import Set
from urllib.parse import unquote
from random import shuffle

from tqdm import tqdm

from refined.offline_data_generation.preprocessing_utils import DENY_CLASSES
from refined.resource_management.loaders import load_redirects, load_instance_of, load_wikipedia_to_qcode

anchor_tag_pattern = re.compile('<a href="([^"]+)">([^>]+)</a>')


def main():
    parser = argparse.ArgumentParser(description='Process cleaned Wikipedia, extract links, merge files.')
    parser.add_argument(
        "--clean_wiki_dir",
        type=str,
        default='output',
        help="Directory where clean Wikipedia is stored (processed by clean_wikipedia_.py)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='cleaned_output',
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
    args.clean_wiki_dir = args.clean_wiki_dir.rstrip('/')
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use "
                         f"--overwrite_output_dir to overwrite.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    merge_files_and_extract_links(args.clean_wiki_dir, args.clean_wiki_dir, args.args.output_dir)


def process_line(line, redirects, wikipedia_to_qcode, instance_of, wikimedia_internal_classes, disambiguation_qcodes):
    ms = anchor_tag_pattern.finditer(line['text'])
    hyperlinks = []
    delta_string_length = 0

    for m in ms:
        offset = m.start() - delta_string_length
        hyperlinks.append({
                "uri": unquote(m.group(1)).replace(' ', '_'),
                "surface_form": m.group(2),
                "start": offset,
                "end": offset + len(m.group(2))
        })
        delta_string_length += len(m.group(0)) - len(m.group(2))

    new_text = re.sub('<a href="([^"]+)">([^>]+)</a>', lambda m: m.group(2), line['text'])
    line['text'] = new_text.rstrip('\n')
    line['hyperlinks'] = hyperlinks  # includes all hyperlinks can be used for span detection

    clean_hyperlinks = []

    for hyperlink in hyperlinks:
        wiki_page_title = hyperlink['uri'].replace('&amp;', '&').replace('&lt;', '<') \
            .replace('&gt;', '>').replace('&le;', '≤').replace('&ge;', '≥')
        wiki_page_title = wiki_page_title[0].upper() + wiki_page_title[1:]
        if wiki_page_title in redirects:
            wiki_page_title = redirects[wiki_page_title]
        if wiki_page_title in wikipedia_to_qcode:
            qcode = wikipedia_to_qcode[wiki_page_title]
            if qcode in disambiguation_qcodes or (qcode in instance_of and
                                                  len(instance_of[qcode] & wikimedia_internal_classes) > 0):
                # exclude list pages, disambiguation pages, surname pages
                continue
            clean_hyperlinks.append(hyperlink)
            clean_hyperlinks[-1]['qcode'] = qcode
    line['hyperlinks_clean'] = clean_hyperlinks
    return line


def merge_files_and_extract_links(input_dir: str, resources_dir: str, output_dir: str):
    redirects = load_redirects(os.path.join(resources_dir, 'redirects.json'))
    instance_of = load_instance_of(os.path.join(resources_dir, 'instance_of_p31.json'))
    title_to_qcode = load_wikipedia_to_qcode(os.path.join(resources_dir, 'enwiki.json'))

    with open(os.path.join(resources_dir, 'disambiguation_qcodes.txt'), 'r') as f:
        disambiguation_qcodes: Set[str] = {l.rstrip('\n') for l in f.readlines()}

    # list, surnames, redirects, and disambiguation (+ name disambiguation)
    deny_classes = DENY_CLASSES

    processed_clean_wiki_file = open(os.path.join(output_dir, 'wikipedia_links_aligned.json'), 'w')
    pbar = tqdm(total=6e+6)
    for base_path, _, file_names in os.walk(input_dir):
        shuffle(file_names)
        for file_name in file_names:
            with open(os.path.join(base_path, file_name), 'r') as f:
                for line in f:
                    line = json.loads(line)
                    line = process_line(line,
                                        redirects=redirects, wikipedia_to_qcode=title_to_qcode,
                                        instance_of=instance_of, wikimedia_internal_classes=deny_classes,
                                        disambiguation_qcodes=disambiguation_qcodes)
                    processed_clean_wiki_file.write(json.dumps(line) + '\n')
                    pbar.update(1)


if __name__ == '__main__':
    main()
