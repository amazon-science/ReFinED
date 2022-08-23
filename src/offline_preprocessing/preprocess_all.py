# This is a script that does all the data pre-processing necessary to generate the data needed to
# train a new ReFinED ER model from scratch.
# Data files are written for intermediate steps so work will resume if the script is stopped.

# This is common pattern of jobs and some can be concurrent.
# Might make simple library to concurrently work on jobs efficiently and handle data dependencies (files) automatically.
# One simple option could be to run a process for each step that waits for the files it needs.

# Run jobs in parallel and don’t recompute results if nothing has changed. Seems like joblib.
# Can it read and write files faster? well tensors are fast.
import subprocess
from typing import Set, Dict, List
import sys
import copy
from argparse import ArgumentParser

from offline_preprocessing.dataclasses_for_preprocessing import AdditionalEntity

sys.path.append('.')

from offline_preprocessing.class_selection import select_classes
from offline_preprocessing.clean_wikipedia import preprocess_wikipedia
from offline_preprocessing.generate_descriptions_tensor import create_description_tensor
from offline_preprocessing.generate_qcode_to_type_indices import create_tensors
from offline_preprocessing.merge_files_and_extract_links import merge_files_and_extract_links
from offline_preprocessing.preprocessing_utils import download_url_with_progress_bar
import os
from multiprocessing import Pool
import sys
import logging
from offline_preprocessing.process_wikidata_dump import build_wikidata_lookups
from offline_preprocessing.process_wiki import build_redirects
from offline_preprocessing.generate_pem import build_pem_lookup
from offline_preprocessing.run_span_detection import run, add_spans_to_existing_datasets
from training.train_md_standalone import train_md_model
from doc_preprocessing.dataclasses import NER_TAG_TO_IX
from utilities.lookup_utils import load_pem, load_labels
from utilities.lookup_utils import load_pem, load_labels, load_instance_of
from tqdm.auto import tqdm
import json

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
LOG = logging.getLogger(__name__)

keep_all_entities = True

OUTPUT_PATH = 'data'
# OUTPUT_PATH = '/tom_data/data_2020_feb'

# do not overwrite
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Wikidata configuration
WIKIDATA_DUMP_URL = 'https://dumps.wikimedia.your.org/wikidatawiki/entities/latest-all.json.bz2'
# WIKIDATA_DUMP_URL = 'https://web.archive.org/web/20191010114721/https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2'
WIKIDATA_DUMP_FILE = 'wikidata.json.bz2'

# September 2019 entities Wikidata
# https://web.archive.org/web/20190930113616/https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2
# maybe second one to match all ids
# https://web.archive.org/web/20191010114721/https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2

# September 2019
# redirects
# https://web.archive.org/web/20191010114721/https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-redirect.sql.gz

# page ids
# https://web.archive.org/web/20191218013206/https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-page.sql.gz
# https://web.archive.org/web/20191010114721/https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-page.sql.gz

# page articles
# https://web.archive.org/web/20191010114721/https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# fallback
# Wikipedia BLINK
# http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2

# Wikipedia configuration
WIKIPEDIA_REDIRECTS_URL = 'https://dumps.wikimedia.your.org/enwiki/latest/enwiki-latest-redirect.sql.gz'
# WIKIPEDIA_REDIRECTS_URL = 'https://web.archive.org/web/20191010114721/https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-redirect.sql.gz'

WIKIPEDIA_REDIRECTS_FILE = 'wikipedia_redirects.sql.gz'

WIKIPEDIA_PAGE_IDS_URL = 'https://dumps.wikimedia.your.org/enwiki/latest/enwiki-latest-page.sql.gz'
# WIKIPEDIA_PAGE_IDS_URL = 'https://web.archive.org/web/20191010114721/https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-page.sql.gz'

WIKIPEDIA_PAGE_IDS_FILE = 'wikipedia_page_ids.sql.gz'


WIKIPEDIA_ARTICLES_URL = 'https://dumps.wikimedia.your.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2'
# WIKIPEDIA_ARTICLES_URL = 'https://web.archive.org/web/20191010114721/https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2'
WIKIPEDIA_ARTICLES_FILE = 'wikipedia_articles.xml.bz2'

AIDA_MEANS_URL = 'http://resources.mpi-inf.mpg.de/yago-naga/aida/download/aida_means.tsv.bz2'
AIDA_MEANS_FILE = 'aida_means.tsv.bz2'


def download_dumps():
    """
    Concurrently download Wikidata text, Wikipedia redirects, Wikipedia page ids, Wikidata dumps
    """
    LOG.info('Downloading Wikidata and Wikipedia dumps.')
    with Pool(processes=4) as pool:
        jobs = [
            pool.apply_async(download_url_with_progress_bar,
                             (WIKIDATA_DUMP_URL, os.path.join(OUTPUT_PATH, WIKIDATA_DUMP_FILE))),
            pool.apply_async(download_url_with_progress_bar,
                             (WIKIPEDIA_PAGE_IDS_URL, os.path.join(OUTPUT_PATH, WIKIPEDIA_PAGE_IDS_FILE))),
            pool.apply_async(download_url_with_progress_bar,
                             (WIKIPEDIA_REDIRECTS_URL, os.path.join(OUTPUT_PATH, WIKIPEDIA_REDIRECTS_FILE))),
            pool.apply_async(download_url_with_progress_bar,
                             (WIKIPEDIA_ARTICLES_URL, os.path.join(OUTPUT_PATH, WIKIPEDIA_ARTICLES_FILE))),
        ]
        for job in jobs:
            job.get()
            job.wait()


def create_qcode_to_idx(pem_file: str, is_test: bool) -> Dict[str, int]:
    all_qcodes = set()
    with open(pem_file, 'r') as f:
        for i, line in tqdm(enumerate(f), total=10e+6, desc='create_qcode_to_idx'):
            line = json.loads(line)
            all_qcodes.update(set(line['qcode_probs'].keys()))
            if is_test and i > 100000:
                break
    return {qcode: qcode_idx + 1 for qcode_idx, qcode in enumerate(list(all_qcodes))}


def build_entity_index(pem_filename: str, output_path: str):
    pem = load_pem(pem_filename)
    all_qcodes: Set[str] = set()
    for qcode_probs in tqdm(pem.values(), total=18749702):
        all_qcodes.update(set(qcode_probs.keys()))
    qcode_to_index = {qcode: qcode_idx + 1 for qcode_idx, qcode in enumerate(list(all_qcodes))}

    with open(os.path.join(output_path, 'qcode_to_idx.json.part'), 'w') as fout:
        for k, v in qcode_to_index.items():
            fout.write(json.dumps({'qcode': k, 'index': v}) + '\n')
    os.rename(os.path.join(output_path, 'qcode_to_idx.json.part'), os.path.join(output_path, 'qcode_to_idx.json'))


def build_class_labels(resources_dir: str):
    with open(os.path.join(resources_dir, 'chosen_classes.txt'), 'r') as f:
        chosen_classes = {l.rstrip('\n') for l in f.readlines()}

    labels = load_labels(os.path.join(resources_dir, 'label.json'), False)
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
    with open(f'{resources_dir}/class_to_label.json', 'w') as f:
        json.dump(cls_to_label, f)

    logging.info('Written class to label')


def main():
    parser = ArgumentParser()
    parser.add_argument('--debug',  type=str,
                        default="n",
                        help="y or n",)
    parser.add_argument('--additional_entities_file', type=str)
    cli_args = parser.parse_args()
    debug = cli_args.debug.lower() == 'y'

    LOG.info('Step 1) Downloading the raw data for Wikidata and Wikipedia.')
    if not os.path.exists(os.path.join(OUTPUT_PATH, 'wikidata.json.bz2')):
        download_dumps()

    LOG.info('Step 2) Processing Wikidata dump to build lookups and sets.')
    args = {'dump_file_path': os.path.join(OUTPUT_PATH, WIKIDATA_DUMP_FILE),
            'output_dir': OUTPUT_PATH, 'overwrite_output_dir': True, 'test': False}
    if not os.path.exists(os.path.join(OUTPUT_PATH, 'sitelinks_cnt.json')):
        build_wikidata_lookups(args_override=args)

    LOG.info('Step 3) Processing Wikipedia redirects dump.')
    args = {'page_sql_gz_filepath': os.path.join(OUTPUT_PATH, WIKIPEDIA_PAGE_IDS_FILE),
            'redirect_sql_gz_filepath': os.path.join(OUTPUT_PATH, WIKIPEDIA_REDIRECTS_FILE),
            'output_dir': OUTPUT_PATH,
            'overwrite_output_dir': True,
            'test': False}
    if not os.path.exists(os.path.join(OUTPUT_PATH, 'redirects.json')):
        build_redirects(args=args)

    LOG.info('Step 4) Extract text from Wikipedia dump.')
    if not os.path.exists(os.path.join(OUTPUT_PATH, 'wikipedia_links_aligned.json')):
        preprocess_wikipedia(dump_path=os.path.join(OUTPUT_PATH, WIKIPEDIA_ARTICLES_FILE),
                             save_path=os.path.join(OUTPUT_PATH, 'preprocessed_wikipedia'))
        merge_files_and_extract_links(input_dir=os.path.join(OUTPUT_PATH, 'preprocessed_wikipedia'),
                                      resources_dir=OUTPUT_PATH, output_dir=OUTPUT_PATH)

    LOG.info('Step 5) Building PEM lookup.')
    # additional entity set file
    # {label: "label",
    # alias: ["alias_1", "alias2"],
    # entity_type: ["qcode_1", "qcode_2"],
    # entity_description: "english description"
    # }
    # A1, A2 instead of Q1, Q2
    additional_entities: List[AdditionalEntity] = []

    if cli_args.additional_entities_file is not None:
        print('Adding additional entities')
        with open(cli_args.additional_entities_file, 'r') as f:
            for line in tqdm(f, desc='Loading additional entities'):
                line = json.loads(line)
                additional_entities.append(AdditionalEntity(**line))

    # add extra human and fictional humans to human qcodes
    # TODO add fictional human to original human qcodes as well
    if additional_entities is not None and len(additional_entities) > 0:
        instance_of = load_instance_of(os.path.join(OUTPUT_PATH, 'instance_of_p31.json'), is_test=debug)
        human_qcodes: Set[str] = set()
        for qcode, classes in tqdm(instance_of.items(), desc='Adding human qcodes from instance_of'):
            if 'Q5' in classes or 'Q15632617' in classes:
                human_qcodes.add(qcode)

        for additional_entity in tqdm(additional_entities, desc='Adding human qcodes from additional entities'):
            if 'Q5' in additional_entity.entity_types or 'Q15632617' in additional_entity.entity_types:
                human_qcodes.add(additional_entity.entity_id)

        with open(os.path.join(OUTPUT_PATH, 'human_qcodes.json'), 'w') as f_out:
            f_out.write('\n'.join(human_qcodes))
        del instance_of

    if not os.path.exists(os.path.join(OUTPUT_PATH, AIDA_MEANS_FILE)):
        download_url_with_progress_bar(url=AIDA_MEANS_URL, output_path=os.path.join(OUTPUT_PATH, AIDA_MEANS_FILE))
    if not os.path.exists(os.path.join(OUTPUT_PATH, 'wiki_pem.json')):
        build_pem_lookup(aligned_wiki_file=os.path.join(OUTPUT_PATH, 'wikipedia_links_aligned.json'),
                         output_dir=OUTPUT_PATH, resources_dir=OUTPUT_PATH, keep_all_entities=keep_all_entities,
                         additional_entities=additional_entities,
                         is_test=debug)

    LOG.info('Step 6) Building entity index from PEM.')
    if not os.path.exists(os.path.join(OUTPUT_PATH, 'qcode_to_idx.json')):
        build_entity_index(os.path.join(OUTPUT_PATH, 'wiki_pem.json'), OUTPUT_PATH)

    # # build descriptions (include labels without descriptions, maybe some alts as well should keep it short)
    # LOG.info('Step 7) Building descriptions tensor.')
    # if not os.path.exists(os.path.join(OUTPUT_PATH, 'descriptions_tns.pt')):
    #     create_description_tensor(output_path=OUTPUT_PATH,
    #                               qcode_to_idx_filename=os.path.join(OUTPUT_PATH, 'qcode_to_idx.json'),
    #                               desc_filename=os.path.join(OUTPUT_PATH, 'desc.json'),
    #                               label_filename=os.path.join(OUTPUT_PATH, 'label.json'),
    #                               wiki_to_qcode=os.path.join(OUTPUT_PATH, 'enwiki.json'),
    #                               keep_all_entities=keep_all_entities)
    #
    # # build classes and selection
    # LOG.info('Step 8) Selecting classes tensor.')
    # if not os.path.exists(os.path.join(OUTPUT_PATH, 'chosen_classes.txt')):
    #     select_classes(resources_dir=OUTPUT_PATH, is_test=False)

    # build descriptions (include labels without descriptions, maybe some alts as well should keep it short)
    LOG.info('Step 7) Building descriptions tensor.')
    if not os.path.exists(os.path.join(OUTPUT_PATH, 'descriptions_tns.pt')):
        create_description_tensor(output_path=OUTPUT_PATH,
                                  qcode_to_idx_filename=os.path.join(OUTPUT_PATH, 'qcode_to_idx.json'),
                                  desc_filename=os.path.join(OUTPUT_PATH, 'desc.json'),
                                  label_filename=os.path.join(OUTPUT_PATH, 'label.json'),
                                  wiki_to_qcode=os.path.join(OUTPUT_PATH, 'enwiki.json'),
                                  additional_entities=additional_entities,
                                  keep_all_entities=keep_all_entities,
                                  is_test=debug)

    LOG.info('Step 8) Selecting classes tensor.')
    if not os.path.exists(os.path.join(OUTPUT_PATH, 'chosen_classes.txt')):
        select_classes(resources_dir=OUTPUT_PATH, is_test=debug)

    LOG.info('Step 9) Creating tensors.')
    if not os.path.exists(os.path.join(OUTPUT_PATH, 'class_to_idx.json')):
        create_tensors(resources_dir=OUTPUT_PATH, additional_entities=additional_entities, is_test=debug)

    LOG.info('Step 10) Creating class labels lookup')
    if not os.path.exists(os.path.join(OUTPUT_PATH, 'class_to_label.json')):
        build_class_labels(OUTPUT_PATH)


    LOG.info('(Step 11) Training MD model for ontonotes numeric spans (date, cardinal, percent etc.)')

    model_dir_prefix = 'onto-onto-article-onto-lower-epoch-10'
    # build classes and selection

    if False:
        LOG.info('(Step 11) Training MD model for DATE spans only')
        model_dir_prefix = 'onto-onto-article-onto-lower-epoch-4'

    NER_TAG_TO_NUM_MD = copy.deepcopy(NER_TAG_TO_IX)
    del NER_TAG_TO_NUM_MD["B-MENTION"]
    del NER_TAG_TO_NUM_MD["I-MENTION"]

    # train_md_model(resources_dir=OUTPUT_PATH, datasets=['onto', 'onto-article', 'onto-lower'],
    #                device='cuda:0', max_seq=300, batch_size=16, bio_only=False, max_articles=None,
    #                ner_tag_to_num=NER_TAG_TO_NUM_MD, num_epochs=10, filter_types=set())


    # LOG.info('Step 10) Relabelling CONLL dataset using DATE MD model')
    # model_dir = [x[0] for x in list(os.walk(OUTPUT_PATH)) if model_dir_prefix in x[0]][0]
    # model_dir = os.path.join(OUTPUT_PATH, "onto-onto-article-onto-lower-epoch-9-mf1-0.8608866832401668")
    # add_spans_to_existing_datasets(dataset_names=["conll"], dataset_dir=os.path.join(OUTPUT_PATH, "datasets"),
    #                                model_dir=model_dir, file_extension="_plus_dates", ner_types_to_add={"DATE",
    #                                                                                                     "CARDINAL",
    #                                                                                                     "MONEY",
    #                                                                                                     "PERCENT",
    #                                                                                                     "TIME",
    #                                                                                                     "ORDINAL",
    #                                                                                                     "QUANTITY"},
    #                                device="cuda:0")

    model_dir_prefix = 'onto-onto-article-onto-lower-onto-article-lower-conll-conll-lower-conll-article-conll-article' \
                       '-lower-webqsp-epoch-9'
    # # TODO: change "data" to OUTPUT_PATH
    # if any(x[0] in model_dir_prefix for x in list(os.walk("data"))):

    datasets = ['onto', 'onto-article', 'onto-lower', 'onto-article-lower',
                                                        'conll', 'conll-lower', 'conll-article',
                                                        'conll-article-lower', 'webqsp']

    # train_md_model(resources_dir=OUTPUT_PATH, datasets=datasets,
    #                device="cuda:0", max_seq=300, batch_size=16, bio_only=False,
    #                ner_tag_to_num=NER_TAG_TO_IX,
    #                additional_filenames={'conll': '_plus_dates', 'conll-lower': '_plus_dates',
    #                                      'conll-article': '_plus_dates', 'conll-article-lower': '_plus_dates'},
    #                use_mention_tag=True,
    #                convert_types={"webqsp": {"DURATION": "TIME", "NUMBER": "CARDINAL"}},
    #                filter_types=set())

    LOG.info('Step 12) Running MD model over Wikipedia.')
    # TODO: change "data" to OUTPUT_PATH
    if not os.path.exists(os.path.join(OUTPUT_PATH, 'wikipedia_links_aligned_spans.json')):
        model_dir = [x[0] for x in list(os.walk("data")) if model_dir_prefix in x[0]][0]
        n_gpu = 2
        run(aligned_wiki_file=os.path.join(OUTPUT_PATH, 'wikipedia_links_aligned.json'),
            n_gpu=n_gpu, resources_dir=OUTPUT_PATH, model_dir=model_dir)
        # command = 'cat '
        # for part_num in range(n_gpu):
        #     command += os.path.join(OUTPUT_PATH, f'wikipedia_links_aligned.json_spans_{part_num} ')
        # command += f"> {os.path.join(OUTPUT_PATH, 'wikipedia_links_aligned_spans.json')}"

        command = 'cat '
        for part_num in range(n_gpu):
            # wikipedia_links_aligned.json_spans_1.json
            command += os.path.abspath(os.path.join(OUTPUT_PATH, f'wikipedia_links_aligned.json_spans_{part_num}.json '))
        f_out = open(os.path.abspath(os.path.join(OUTPUT_PATH, 'wikipedia_links_aligned_spans.json')), 'w')
        process = subprocess.Popen(command.split(), stdout=f_out)
        output, error = process.communicate()
        f_out.close()

    # LOG.info('Step 12) Creating tensors.')
    # if not os.path.exists(os.path.join(OUTPUT_PATH, 'class_to_idx.json')):
    #     create_tensors(resources_dir=OUTPUT_PATH)
    #
    # LOG.info('Step 13) Creating class labels lookup')
    # if not os.path.exists(os.path.join(OUTPUT_PATH, 'class_to_label.json')):
    #     build_class_labels(OUTPUT_PATH)
    #
    # # do not hard code 4 gpus use call to see how many are free
    # # TODO: how about Evi tkids?


    # do not hard code 4 gpus use call to see how many are free
    # TODO: how about Evi tkids?


if __name__ == '__main__':
    main()

# ~/anaconda3/envs/pytorch_p36/bin/python
