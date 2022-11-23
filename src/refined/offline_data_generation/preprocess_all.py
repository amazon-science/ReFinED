# This is a script that does all the data pre-processing necessary to generate the data needed to
# train a new ReFinED ER model from scratch.
# Data files are written for intermediate steps so work will resume if the script is restarted.

import copy
import json
import logging
import os
import subprocess
import sys
from argparse import ArgumentParser
from multiprocessing.pool import ThreadPool
from typing import Set, Dict, List

from tqdm.auto import tqdm

from refined.model_components.config import NER_TAG_TO_IX
from refined.offline_data_generation.build_lmdb_dicts import build_lmdb_dicts
from refined.offline_data_generation.class_selection import select_classes
from refined.offline_data_generation.clean_wikipedia import preprocess_wikipedia
from refined.offline_data_generation.dataclasses_for_preprocessing import AdditionalEntity
from refined.offline_data_generation.generate_descriptions_tensor import create_description_tensor
from refined.offline_data_generation.generate_pem import build_pem_lookup
from refined.offline_data_generation.generate_qcode_to_type_indices import create_tensors
from refined.offline_data_generation.merge_files_and_extract_links import merge_files_and_extract_links
from refined.offline_data_generation.preprocessing_utils import download_url_with_progress_bar
from refined.offline_data_generation.process_wiki import build_redirects
from refined.offline_data_generation.process_wikidata_dump import build_wikidata_lookups
from refined.offline_data_generation.run_span_detection import run, add_spans_to_existing_datasets
from refined.resource_management.aws import S3Manager
from refined.resource_management.loaders import load_pem, load_labels, load_instance_of
from refined.resource_management.resource_manager import ResourceManager
from refined.training.train.train_md_standalone import train_md_model
from refined.utilities.general_utils import get_logger

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# LOG = logging.getLogger(__name__)

LOG = get_logger(__name__)

# keep_all_entities=True means keep Wikidata entities even if they do not have a Wikipedia page
# keep_all_entities=False means only keep Wikidata entities that have a Wikipedia page
keep_all_entities = True

OUTPUT_PATH = 'data'

os.makedirs(OUTPUT_PATH, exist_ok=True)

# NOTE that these dump urls may need to be updated if mirrors no longer exist.
# Also note that dump mirrors have different download speeds.
# See https://wikimedia.mirror.us.dev/mirrors.html for a list.

# Wikidata configuration
WIKIDATA_DUMP_URL = 'https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2'
WIKIDATA_DUMP_FILE = 'wikidata.json.bz2'

# Wikipedia configuration
WIKIPEDIA_REDIRECTS_URL = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-redirect.sql.gz'
WIKIPEDIA_REDIRECTS_FILE = 'wikipedia_redirects.sql.gz'
WIKIPEDIA_PAGE_IDS_URL = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-page.sql.gz'
WIKIPEDIA_PAGE_IDS_FILE = 'wikipedia_page_ids.sql.gz'
WIKIPEDIA_ARTICLES_URL = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2'
WIKIPEDIA_ARTICLES_FILE = 'wikipedia_articles.xml.bz2'
AIDA_MEANS_URL = 'http://resources.mpi-inf.mpg.de/yago-naga/aida/download/aida_means.tsv.bz2'
AIDA_MEANS_FILE = 'aida_means.tsv.bz2'


def download_dumps_in_parallel():
    """
    Concurrently download Wikidata text, Wikipedia redirects, Wikipedia page ids, Wikidata dumps
    """
    LOG.info('Downloading Wikidata and Wikipedia dumps.')
    with ThreadPool(processes=4) as pool:
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


def download_dumps():
    """
    Concurrently download Wikidata text, Wikipedia redirects, Wikipedia page ids, Wikidata dumps
    """
    LOG.info('Downloading Wikidata and Wikipedia dumps.')
    download_url_with_progress_bar(WIKIPEDIA_REDIRECTS_URL, os.path.join(OUTPUT_PATH, WIKIPEDIA_REDIRECTS_FILE))
    download_url_with_progress_bar(WIKIPEDIA_PAGE_IDS_URL, os.path.join(OUTPUT_PATH, WIKIPEDIA_PAGE_IDS_FILE))
    download_url_with_progress_bar(WIKIDATA_DUMP_URL, os.path.join(OUTPUT_PATH, WIKIDATA_DUMP_FILE))
    download_url_with_progress_bar(WIKIPEDIA_ARTICLES_URL, os.path.join(OUTPUT_PATH, WIKIPEDIA_ARTICLES_FILE))


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

    labels = load_labels(os.path.join(resources_dir, 'qcode_to_label.json'), False)
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
    parser.add_argument('--debug', type=str,
                        default="n",
                        help="y or n", )
    parser.add_argument('--additional_entities_file', type=str)
    cli_args = parser.parse_args()
    debug = cli_args.debug.lower() == 'y'

    LOG.info('Step 1) Downloading the raw data for Wikidata and Wikipedia.')
    if not os.path.exists(os.path.join(OUTPUT_PATH, 'wikipedia_articles.xml.bz2')):
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

    # build descriptions (include labels without descriptions, maybe some alts as well should keep it short)
    LOG.info('Step 7) Building descriptions tensor.')
    if not os.path.exists(os.path.join(OUTPUT_PATH, 'descriptions_tns.pt')):
        create_description_tensor(output_path=OUTPUT_PATH,
                                  qcode_to_idx_filename=os.path.join(OUTPUT_PATH, 'qcode_to_idx.json'),
                                  desc_filename=os.path.join(OUTPUT_PATH, 'desc.json'),
                                  label_filename=os.path.join(OUTPUT_PATH, 'qcode_to_label.json'),
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

    LOG.info('(Step 11) Training MD model for ontonotes numeric/date spans (date, cardinal, percent etc.)')
    # check if model exists
    model_dir_prefix = 'onto-onto-article-onto-lower-epoch-4'
    if len([x[0] for x in list(os.walk(OUTPUT_PATH)) if model_dir_prefix in x[0]]) == 0:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        resource_manager = ResourceManager(S3Manager(),
                                           data_dir=OUTPUT_PATH,
                                           entity_set=None,
                                           model_name=None
                                           )
        resource_manager.download_datasets_if_needed()
        NER_TAG_TO_NUM_MD = copy.deepcopy(NER_TAG_TO_IX)
        del NER_TAG_TO_NUM_MD["B-MENTION"]
        del NER_TAG_TO_NUM_MD["I-MENTION"]
        train_md_model(resources_dir=OUTPUT_PATH, datasets=['onto', 'onto-article', 'onto-lower'],
                       device='cuda:0', max_seq=500, batch_size=16, bio_only=False, max_articles=None,
                       ner_tag_to_num=NER_TAG_TO_NUM_MD, num_epochs=10, filter_types=set())
    else:
        LOG.info('Model already trained so skipping')

    LOG.info('Step 12) Relabelling CONLL dataset using numeric/date MD model')
    if not os.path.exists(os.path.join(OUTPUT_PATH, "datasets", "conll_train_plus_dates.txt")):
        model_dir = [x[0] for x in list(os.walk(OUTPUT_PATH)) if model_dir_prefix in x[0]][0]  # or -1
        add_spans_to_existing_datasets(dataset_names=["conll"], dataset_dir=os.path.join(OUTPUT_PATH, "datasets"),
                                       model_dir=model_dir, file_extension="_plus_dates", ner_types_to_add={"DATE",
                                                                                                            "CARDINAL",
                                                                                                            "MONEY",
                                                                                                            "PERCENT",
                                                                                                            "TIME",
                                                                                                            "ORDINAL",
                                                                                                            "QUANTITY"},
                                       device="cuda:0")
    else:
        LOG.info('Already relabelled CONLL dataset so skipping')

    LOG.info('Step 13) Train MD model on augmented MD datasets')
    model_dir_prefix = 'onto-onto-article-onto-lower-onto-article-lower-conll-conll-lower-conll-article-conll-article' \
                       '-lower-webqsp-epoch-9'
    if len([x[0] for x in list(os.walk(OUTPUT_PATH)) if model_dir_prefix in x[0]]) == 0:
        datasets = ['onto', 'onto-article', 'onto-lower', 'onto-article-lower',
                    'conll', 'conll-lower', 'conll-article',
                    'conll-article-lower', 'webqsp']
        train_md_model(resources_dir=OUTPUT_PATH, datasets=datasets,
                       device="cuda:0", max_seq=510, batch_size=16, bio_only=False,
                       ner_tag_to_num=NER_TAG_TO_IX,
                       additional_filenames={'conll': '_plus_dates', 'conll-lower': '_plus_dates',
                                             'conll-article': '_plus_dates', 'conll-article-lower': '_plus_dates'},
                       use_mention_tag=True,
                       convert_types={"webqsp": {"DURATION": "TIME", "NUMBER": "CARDINAL"}},
                       filter_types=set())
    else:
        LOG.info("Found an MD model already trained on augmented MD datasets, so skipping")

    LOG.info('Step 14) Running MD model over Wikipedia.')
    if not os.path.exists(os.path.join(OUTPUT_PATH, 'wikipedia_links_aligned_spans.json')):
        model_dir = [x[0] for x in list(os.walk(OUTPUT_PATH)) if model_dir_prefix in x[0]][0]
        n_gpu = 1  # can change this to speed it up if more GPUs are available
        run(aligned_wiki_file=os.path.join(OUTPUT_PATH, 'wikipedia_links_aligned.json'),
            n_gpu=n_gpu, resources_dir=OUTPUT_PATH, model_dir=model_dir)
        command = 'cat '
        for part_num in range(n_gpu):
            command += os.path.abspath(
                os.path.join(OUTPUT_PATH, f'wikipedia_links_aligned.json_spans_{part_num}.json '))
        f_out = open(os.path.abspath(os.path.join(OUTPUT_PATH, 'wikipedia_links_aligned_spans.json')), 'w')
        process = subprocess.Popen(command.split(), stdout=f_out)
        output, error = process.communicate()
        print(error)
        f_out.close()

    LOG.info('Step 15) Building LMDB dictionaries and storing files in the expected file structures.')
    build_lmdb_dicts(preprocess_all_data_dir=OUTPUT_PATH, keep_all_entities=keep_all_entities)

    LOG.info("The preprocess_all.py script is done. You can now use the newly generated/updated data files "
             "for your trained model or train a model from scratch on the newly generated Wikipedia dataset.")
    LOG.info(f"The data_dir is the relative path: {OUTPUT_PATH}/organised_data_dir.")
    LOG.info(f"You can train a model with the new data using `train.py --download_files n "
             f"--data_dir {OUTPUT_PATH}/organised_data_dir` . Ensure --download_files n to avoid overwriting.")
    LOG.info(f"You can use an existing model with the updated data files (e.g. includes recently added entities) "
             f"without retraining the model (zero-shot entities) by replacing the data files stored in an existing "
             f"data_dir. Note that qcode_to_class_tns will need to be renamed in the resource_constants file "
             f"and download should be se to False to avoid downloading a different file.")
    LOG.info("Done.")
    # example_usage = """
    # from refined.doc_preprocessing.preprocessor import PreprocessorInferenceOnly
    # from refined.model_components.config import NER_TAG_TO_IX
    # from refined.resource_management.aws import S3Manager
    # from refined.resource_management.resource_manager import ResourceManager
    #
    # data_dir = "/home/ubuntu/refined/data/organised_data_dir"
    # resource_manager = ResourceManager(S3Manager(),
    #                                    data_dir=data_dir,
    #                                    entity_set="wikidata",
    #                                    load_qcode_to_title=True,
    #                                    load_descriptions_tns=True,
    #                                    model_name=None,
    #                                    )
    #
    # preprocessor = PreprocessorInferenceOnly(
    #     data_dir=data_dir,
    #     max_candidates=30,
    #     transformer_name="roberta-base",
    #     ner_tag_to_ix=NER_TAG_TO_IX,  # for now include default ner_to_tag_ix can make configurable in future
    #     entity_set="wikidata",
    #     use_precomputed_description_embeddings=False
    # )
    # """


if __name__ == '__main__':
    main()
