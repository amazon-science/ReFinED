import argparse
import gzip
import os
import re
import json
from typing import Dict
from types import SimpleNamespace

from tqdm import tqdm


def build_redirects(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='Build lookup dictionaries from Wikipedia dumps.')
        parser.add_argument(
            "--page_sql_gz_filepath",
            default='july_9_2020_enwiki-latest-page.sql.gz',
            type=str,
            help="file path to Wikipedia pages dump file (enwiki-latest-page.sql.gz)"
        )
        parser.add_argument(
            "--redirect_sql_gz_filepath",
            default='july_9_2020_enwiki-latest-redirect.sql.gz',
            type=str,
            help="file path to Wikipedia redirects dump file (enwiki-latest-redirect.sql.gz)"
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
    else:
        args = SimpleNamespace(**args)

    args.output_dir = args.output_dir.rstrip('/')

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use "
                         f"--overwrite_output_dir to overwrite.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if os.path.exists(os.path.join(args.output_dir, 'redirects.json')):
        return
    page_id_to_title: Dict[str, str] = generate_wiki_id_to_title(args.page_sql_gz_filepath, args.output_dir)
    generate_redirects(args.redirect_sql_gz_filepath, args.output_dir, page_id_to_title)


def generate_wiki_id_to_title(page_sql_gz_filepath: str, output_dir: str) -> Dict[str, str]:
    # page_id, namespace, title, restrictions, redirect, new, random, touched, links, latest, len, content_model, lang
    page_id_to_title: Dict[str, str] = dict()
    pattern = re.compile("([0-9]+),([0-9]+),(.+),(.+),([0-9]+),([0-9]+),(.+),(.+),(.+),([0-9]+),([0-9]+),(.+),(.+)")
    wiki_id_to_title_file = open(f'{output_dir}/wiki_id_to_title.json', 'w')
    with gzip.open(page_sql_gz_filepath, 'r') as f:
        for line in tqdm(f, total=5775):
            if len(line) < 500:
                continue
            parsed_line = line[27:].decode('utf-8')
            parsed_line = parsed_line.split('),(')
            parsed_line = [x[1:] if x[0] == '(' else x for x in parsed_line]
            parsed_line = [x[:-3] if x[-3:-1] == ');' else x for x in parsed_line]
            for x in parsed_line:
                m = pattern.match(x)
                if m is None:
                    continue
                groups = m.groups()
                page_id, namespace, title, restrictions, redirect, new, random, touched, links, \
                    latest, length, content_model, lang = groups
                if not namespace == '0':
                    continue
                title = title[1:-1]
                page_id_to_title[page_id] = title
                wiki_id_to_title_file.write(json.dumps({'wiki_page_id': page_id, 'wiki_title': title}) + '\n')
    wiki_id_to_title_file.close()
    return page_id_to_title


def generate_redirects(redirect_sql_gz_filepath: str, output_dir: str, page_id_to_title: Dict[str, str]):
    redirects_file = open(f'{output_dir}/redirects.json.part', 'w')
    pattern = re.compile("([0-9]+),([0-9]+),'(.+)','(.*)','(.*)'")
    with gzip.open(redirect_sql_gz_filepath, 'r') as f:
        for line in tqdm(f):
            if len(line) < 500:
                continue
            parsed_line = line[31:].decode('utf-8')
            parsed_line = parsed_line.split('),(')
            parsed_line = [x[1:] if x[0] == '(' else x for x in parsed_line]
            parsed_line = [x[:-3] if x[-3:-1] == ');' else x for x in parsed_line]
            for x in parsed_line:
                m = pattern.match(x)
                if m is None:
                    continue
                groups = m.groups()
                wiki_curid_surface, namespace, dest_wiki_title, _, _ = groups
                if not namespace == '0':
                    continue
                if wiki_curid_surface in page_id_to_title:
                    redirects_file.write(json.dumps({'wiki_title': page_id_to_title[wiki_curid_surface],
                                                     'dest_title': dest_wiki_title}) + '\n')

        os.rename(os.path.join(output_dir, 'redirects.json.part'), os.path.join(output_dir, 'redirects.json'))


if __name__ == '__main__':
    build_redirects()
