import argparse
import logging
import os
import subprocess
import sys
from multiprocessing import Process
from typing import Optional, List, Set

import torch
import ujson as json
from tqdm.auto import tqdm

from refined.dataset_reading.mention_detection.conll_reader import CoNLLNER
from refined.dataset_reading.mention_detection.ontonotes_reader import OntoNotesNER
from refined.dataset_reading.mention_detection.webqsp_reader import WebQSPNER
from refined.inference.standalone_md import MentionDetector

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
LOG = logging.getLogger(__name__)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def main():
    parser = argparse.ArgumentParser(description='Process cleaned Wikipedia, extract links, merge files.')
    parser.add_argument(
        "--aligned_wiki_file",
        type=str,
        default='/big_data/wikidata/output_aligned/wikitext_aligned_wd_shuf.json',
        help="File path for cleaned wikipedia text with links extracted and mapped to Wikidata."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='/big_data/wikidata/output_aligned',
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
        "--device",
        type=str,
        default='cuda:0',
    )
    parser.add_argument(
        "--start_line",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end_line",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.rstrip('/')
    # run()


def run(aligned_wiki_file: str, n_gpu: int, resources_dir: str, model_dir: str, end_line: int = -1):
    if end_line == -1:
        end_line = wc(aligned_wiki_file)

    if n_gpu > 1:
        lines_per_p = end_line // n_gpu
        p_start = 0
        p_end = p_start + lines_per_p
        processes = []
        for i in range(n_gpu):
            p = Process(target=add_spans, args=(aligned_wiki_file, resources_dir, f'cuda:{i}', p_start,
                                                p_end, model_dir, i))
            p.start()
            processes.append(p)
            p_start = p_end + 1
            p_end += lines_per_p
        for p in processes:
            LOG.info('A process completed.')
            p.join()
    else:
        device = str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        add_spans(aligned_wiki_file=aligned_wiki_file, output_dir=resources_dir,
                  device=device, start_line=0, end_line=end_line, model_dir=model_dir)


def wc(filename: str) -> int:
    return int(subprocess.check_output(['wc', '-l', filename]).split()[0])


def add_spans(aligned_wiki_file: str, output_dir: str, device: str, start_line: int, end_line: int, model_dir: str,
              part_num: Optional[int] = None):
    if part_num is None:
        part_num = 0
    output_dir = output_dir.rstrip('/')
    device = "cpu" if not torch.cuda.is_available() else device   # fallback to using CPU
    md = MentionDetector.init_from_pretrained(model_dir=model_dir, device=device)

    output_filename = f'{output_dir}/wikipedia_links_aligned.json_spans_{part_num}.json'
    with open(output_filename, 'w') as output_file:
        with open(aligned_wiki_file, 'r') as f:
            for line_num, line in tqdm(enumerate(f), total=end_line - start_line, desc=f'Process num: {part_num}',
                                       disable=part_num is not None and part_num > 0):
                if start_line <= line_num <= end_line:
                    line = json.loads(line)
                    spans = md.process_text(line['text'])
                    line['predicted_spans'] = spans
                    output_file.write(json.dumps(line) + '\n')
                if line_num > end_line:
                    break


def add_spans_to_existing_datasets(dataset_names: List[str], dataset_dir: str, model_dir: str,
                                   file_extension: str, ner_types_to_add: Set[str], device: str = "cpu"):

    name_to_dataset = {
        "onto": OntoNotesNER,
        "conll": CoNLLNER,
        "webqsp": WebQSPNER
    }

    mention_detector = MentionDetector.init_from_pretrained(model_dir=model_dir, device=device)

    for dataset_name in dataset_names:
        dataset_class = name_to_dataset[dataset_name]

        for split in ["train", "dev", "test"]:
            dataset = dataset_class(
                data_split=split,
                data_dir=dataset_dir,
                transformer_name=mention_detector.transformer_name,
                max_seq=mention_detector.max_seq,
                ner_tag_to_num=mention_detector.ner_tag_to_num,
                random_lower_case_prob=0.0,
                random_replace_question_mark=0.0,
                sentence_level=False,
                lower=False,
                bio_only=False,
            )
            LOG.info(f"Adding additional labels to {split} split of {dataset_name} dataset...")
            dataset.relabel_dataset(additional_filename=file_extension, mention_detector=mention_detector,
                                    ner_types_to_add=ner_types_to_add)


if __name__ == '__main__':
    main()
