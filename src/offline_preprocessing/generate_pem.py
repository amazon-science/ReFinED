import sys

from offline_preprocessing.dataclasses_for_preprocessing import AdditionalEntity
from offline_preprocessing.preprocessing_utils import DENY_CLASSES

sys.path.append('.')

import argparse
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, Set, Iterator, Tuple, List, Optional
from urllib.parse import unquote

import ujson as json
from tqdm.auto import tqdm

from utilities.lookup_utils import normalize_surface_form, load_redirects, load_wikipedia_to_qcode, \
    load_instance_of, title_to_qcode, remove_wiki_brackets, load_aida_means, load_labels, load_aliases

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
LOG = logging.getLogger(__name__)
EXCLUDE_LIST_AND_DISAMBIGUATION_PAGES = True


def generate_pem(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='Process cleaned Wikipedia, extract links, merge files.')
        parser.add_argument(
            "--aligned_wiki_file",
            type=str,
            default='output_aligned/wikitext_aligned_wd_shuf.json',
            help="File path for cleaned wikipedia text with links extracted and mapped to Wikidata."
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
            "--use_tkid",
            action="store_true",
            help="Use tkid instead of qcodes (must provide title -> tkid file)."
        )
        parser.add_argument(
            "--title_to_tkid",
            type=str,
            default='title_to_tkid.json',
            help="Use tkid instead of qcodes (only needed when --use_tkid)"
        )
        parser.add_argument(
            "--redirects_file",
            type=str,
            default='wiki_output/redirects.json',
            help="File path for redirects file  (only needed when --use_tkid)"
        )
        parser.add_argument(
            "--wiki_to_qcode_file",
            type=str,
            default='output/enwiki.json',
            help="File path for wikipedia titles to qcodes"
        )
        parser.add_argument(
            "--disambiguation_file",
            type=str,
            default='output/disambiguation_qcodes.txt',
            help="File path for wikipedia titles to qcodes"
        )
        parser.add_argument(
            "--instance_of_file",
            type=str,
            default='output/instance_of_p31.json',
            help="File path for Wikidata instance_of (classes) file."
        )
        parser.add_argument(
            "--labels_file",
            type=str,
            default='output/label.json',
            help="File path for Wikidata labels file."
        )
        parser.add_argument(
            "--aliases_file",
            type=str,
            default='output/aliases.json',
            help="File path for Wikidata aliases file."
        )
        parser.add_argument(
            "--aida_means_file",
            type=str,
            default='output/aida_means.tsv.bz2',
            help="File path for AIDA means tsv file."
        )
        parser.add_argument(
            "--add_titles",
            action="store_true",
            help="Adds titles (as strings) to pem"
        )
        parser.add_argument(
            "--add_redirects",
            action="store_true",
            help="Adds redirects (as strings) to pem"
        )
        parser.add_argument(
            "--add_labels",
            action="store_true",
            help="Adds labels (as strings) to pem"
        )
        parser.add_argument(
            "--add_aliases",
            action="store_true",
            help="Adds aliases (as strings) to pem"
        )
        parser.add_argument(
            "--add_aida_means",
            action="store_true",
            help="Adds aliases (as strings) to pem"
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

    if args.use_tkid:
        title_to_tkid = read_wiki_to_tkid(args.title_to_tkid)
        generate_pem_tkid(args.aligned_wiki_file, title_to_tkid, args.output_dir,
                          redirects=load_redirects(args.redirects_file))
    else:
        build_pem_lookup(args.aligned_wiki_file, args.output_dir, resources_dir=os.path.dirname(args.redirects_file), is_test=args.test,
                         add_titles=args.add_titles, add_redirects=args.add_redirects,
                         add_aida_means=args.add_aida_means, add_labels=args.add_labels,
                         add_aliases=args.add_aliases)


def read_wiki_to_tkid(filename: str):
    title_to_tkid = dict()
    with open(filename, 'r') as f:
        for line in tqdm(f, total=6e+6, desc='Loading wiki_to_tkid.json file'):
            line = json.loads(line)
            title = unquote(line['left'][37:-3]).replace(' ', '_')
            tkid = line['right']
            title_to_tkid[title] = tkid
    return title_to_tkid


def generate_pem_tkid(aligned_wiki_file: str, title_to_tkid: Dict[str, str], output_dir: str,
                      redirects: Dict[str, str]):
    surface_form_to_link_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with open(aligned_wiki_file, 'r') as f:
        num_pages = 0
        for line in tqdm(f, total=6e+6):
            line = json.loads(line)
            num_pages += 1
            for ent in line['hyperlinks_clean']:
                wiki_page_title = ent['uri'].replace('&amp;', '&').replace('&lt;', '<') \
                    .replace('&gt;', '>').replace('&le;', '≤').replace('&ge;', '≥')
                wiki_page_title = wiki_page_title[0].upper() + wiki_page_title[1:]
                if wiki_page_title in redirects:
                    wiki_page_title = redirects[wiki_page_title]

                if wiki_page_title not in title_to_tkid:
                    continue
                surface_form = normalize_surface_form(ent['surface_form'])
                surface_form_to_link_counts[surface_form][title_to_tkid[wiki_page_title]] += 1
    pem: Dict[str, Dict[str, float]] = defaultdict(dict)
    for surface_form, qcode_link_counts in surface_form_to_link_counts.items():
        total_link_count = sum(link_count for qcode, link_count in qcode_link_counts.items())
        pem[surface_form] = dict(sorted([(qcode, link_count / total_link_count) for qcode, link_count in
                                         qcode_link_counts.items()], key=lambda x: x[1], reverse=True))
    with open(f'{output_dir}/wiki_pem_tkid.json', 'w') as output_file:
        for surface_form, qcode_probs in pem.items():
            output_file.write(json.dumps({'surface_form': surface_form, 'qcode_probs': qcode_probs}) + '\n')


def build_pem_lookup(aligned_wiki_file: str, output_dir: str, resources_dir: str, is_test: bool = False,
                     add_titles: bool = True,
                     add_redirects: bool = True, add_aida_means: bool = True, add_labels: bool = True,
                     add_aliases: bool = True, keep_all_entities: bool = True,
                     additional_entities: Optional[List[AdditionalEntity]] = None):
    # load lookups
    redirects = load_redirects(os.path.join(resources_dir, 'redirects.json'), is_test=is_test)
    instance_of = load_instance_of(os.path.join(resources_dir, 'instance_of_p31.json'), is_test=is_test)
    wiki_title_to_qcode = load_wikipedia_to_qcode(os.path.join(resources_dir, 'enwiki.json'), is_test=is_test and False)
    aida_means: Iterator[Tuple[str, str]] = load_aida_means(os.path.join(resources_dir, 'aida_means.tsv.bz2'))

    wikidata_wikipedia_qcodes: Set[str] = set()
    for qcode in wiki_title_to_qcode.values():
        if qcode is not None and not (qcode in instance_of and len(instance_of[qcode] & DENY_CLASSES) > 0):
            wikidata_wikipedia_qcodes.add(qcode)

    print('len(wikidata_wikipedia_qcodes)', len(wikidata_wikipedia_qcodes))

    # only add Wikidata entities with a Wikipedia page initially
    labels: Dict[str, str] = load_labels(os.path.join(resources_dir, 'label.json'), qcodes=wikidata_wikipedia_qcodes,
                                         keep_all_entities=keep_all_entities, is_test=is_test)
    aliases: Dict[str, List[str]] = load_aliases(os.path.join(resources_dir, 'aliases.json'),
                                                 keep_all_entities=keep_all_entities, qcodes=wikidata_wikipedia_qcodes,
                                                 is_test=is_test)

    surface_form_to_link_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # add additional entities
    if additional_entities is not None:
        for line_num, additional_entity in enumerate(tqdm(additional_entities, desc='Adding additional entities to PEM')):
            for surface_form in additional_entity.aliases + [additional_entity.label]:
                surface_form = normalize_surface_form(surface_form)
                surface_form_to_link_counts[surface_form][additional_entity.entity_id] += 1

            # if is_test and line_num > 10000:
            #     break

    # adds titles
    if add_labels:
        line_num = 0
        for qcode, surface_form in tqdm(labels.items(), desc='Adding Wikidata labels'):
            if 'Q' not in qcode:
                continue
            if qcode not in wikidata_wikipedia_qcodes and not keep_all_entities:
                continue
            line_num += 1
            if is_test and line_num > 1000:
                break
            if qcode is not None and not (qcode in instance_of and len(instance_of[qcode] & DENY_CLASSES) > 0):
                surface_form = normalize_surface_form(surface_form, remove_the=True)
                surface_form_to_link_counts[surface_form][qcode] += 1

    if add_aliases:
        line_num = 0
        for qcode, surface_forms in tqdm(aliases.items(), desc='Adding Wikidata aliases'):
            line_num += 1
            if 'Q' not in qcode:
                continue
            if is_test and line_num > 1000:
                break
            for surface_form in surface_forms:
                if qcode is not None and not (qcode in instance_of and len(instance_of[qcode] & DENY_CLASSES) > 0):
                    surface_form = normalize_surface_form(surface_form, remove_the=True)
                    surface_form_to_link_counts[surface_form][qcode] += 1
    with open(aligned_wiki_file, 'r') as f:
        num_pages = 0
        for line in tqdm(f, total=6e+6, desc='Adding links'):
            line = json.loads(line)
            num_pages += 1
            for ent in line['hyperlinks_clean']:
                if ent['qcode'] in instance_of and len(instance_of[ent['qcode']] & DENY_CLASSES) > 0:
                    continue
                surface_form = normalize_surface_form(ent['surface_form'], remove_the=True)
                surface_form_to_link_counts[surface_form][ent['qcode']] += 1
            if is_test and num_pages > 1000:
                break

    if add_titles:
        num_pages = 0
        for wiki_title, qcode in tqdm(wiki_title_to_qcode.items(), desc='Adding titles'):
            num_pages += 1
            if not (qcode in instance_of and len(instance_of[qcode] & DENY_CLASSES) > 0):
                surface_form = remove_wiki_brackets(normalize_surface_form(wiki_title.replace('_', ' '))
                            .replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&le;', '≤')
                            .replace('&ge;', '≥'))
                surface_form_to_link_counts[surface_form][qcode] += 1

            if is_test and num_pages > 1000:
                break

    # add redirects
    if add_redirects:
        for source_title, dest_title in tqdm(redirects.items(), desc='Adding redirects'):
            qcode = title_to_qcode(dest_title, redirects=redirects, wikipedia_to_qcode=wiki_title_to_qcode)
            if qcode is not None and not (qcode in instance_of and len(instance_of[qcode] & DENY_CLASSES) > 0):
                surface_form = remove_wiki_brackets(normalize_surface_form(source_title.replace('_', ' '))
                    .replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&le;', '≤')
                    .replace('&ge;', '≥'))
                surface_form_to_link_counts[surface_form][qcode] += 1

    if add_aida_means:
        line_num = 0
        for surface_form, wiki_page in aida_means:
            qcode = title_to_qcode(wiki_page, redirects=redirects, wikipedia_to_qcode=wiki_title_to_qcode)
            if qcode is not None and not (qcode in instance_of and len(instance_of[qcode] & DENY_CLASSES) > 0):
                surface_form = normalize_surface_form(surface_form)
                surface_form_to_link_counts[surface_form][qcode] += 1
            line_num += 1
            if is_test and line_num > 10000:
                break



    # TODO add Wikidata labels and Wikidata alias, and crosswikis to includes tables/list links in link counts
    # consider taking/storing top 30 only
    pem: Dict[str, Dict[str, float]] = defaultdict(dict)
    for surface_form, qcode_link_counts in tqdm(surface_form_to_link_counts.items(), desc='Writing file'):
        total_link_count = sum(link_count for qcode, link_count in qcode_link_counts.items())
        pem[surface_form] = dict(sorted([(qcode, link_count / total_link_count) for qcode, link_count in
                                         qcode_link_counts.items()], key=lambda x: x[1], reverse=True))

    with open(f'{output_dir}/wiki_pem.json.part', 'w') as output_file:
        for surface_form, qcode_probs in tqdm(pem.items()):
            output_file.write(json.dumps({'surface_form': surface_form, 'qcode_probs': qcode_probs}) + '\n')

    os.rename(f'{output_dir}/wiki_pem.json.part', f'{output_dir}/wiki_pem.json')


def load_pem():
    pem = dict()
    from tqdm.auto import tqdm
    with open('output/wiki_pem.json', 'r') as f:
        for line in tqdm(f, total=6e+6):
            line = json.loads(line)
            pem[line['surface_form']] = line['qcode_probs']


if __name__ == '__main__':
    generate_pem()
