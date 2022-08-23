import argparse
import gzip
import json
import os
import subprocess
import sys
from multiprocessing import Process
from typing import Optional

from tqdm.auto import tqdm

import ujson as json

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # ensures first 4 GPUs are visible
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # disable synchronizing, can enable for debugging
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # disable synchronizing, can enable for debugging
os.environ["TOKENIZERS_PARALLELISM"] = "false"


sys.path.append(".")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser(
        description="Process cleaned Wikipedia, extract links, merge files."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/big_data/ads_dataset.json",
        help="Input file of documents (each line has attribute 'text')",
    )
    parser.add_argument(
        "--output_file", type=str, default="/big_data/refined_ads_new.json", help="Output file"
    )
    parser.add_argument("--data_dir", type=str, default="/data/tayoola/2021_data", help="data dir")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/data/tayoola/2021_code/TomED/best_model",
        help="model dir",
    )
    parser.add_argument("--code_dir", type=str, default="/tom_data/2021_code", help="code dir")
    parser.add_argument(
        "--test", action="store_true", help="mode for testing (only processes first 500 lines)"
    )
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--number_of_parts",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--parts_to_process",
        type=str,
        help="A json list for the parts for each gpu to process. [0, 1, 2, 3] means cuda:0 will process part 0.",
        default="[0, 1, 2, 3]",
    )
    args = parser.parse_args()
    sys.path.append(args.code_dir)

    if args.n_gpu > 1:
        parts_to_process = json.loads(args.parts_to_process)
        processes = []
        for i in range(args.n_gpu):
            p = Process(
                target=add_spans,
                args=(
                    args.input_file,
                    args.output_file,
                    f"cuda:{i}",
                    parts_to_process[i],
                    args.number_of_parts,
                    args.data_dir,
                    args.model_dir,
                    i,
                    args.test,
                    args.n_gpu,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            print("A process completed.")
            p.join()
    else:
        pass
        # add_spans(args.input_file, args.output_file, 'cuda:0', args.start_line, args.end_line, args.test)


def wc(filename: str) -> int:
    return int(subprocess.check_output(["wc", "-l", filename]).split()[0])


def add_spans(
    input_file: str,
    output_file: str,
    device: str,
    process_part: int,
    total_parts: int,
    data_dir: str,
    model_dir: str,
    part_num: Optional[int] = 0,
    debug: bool = False,
    max_parts: int = 1,
):
    print(
        f"[Process {part_num}] processing processing part {process_part}, total parts {total_parts}"
    )

    from refined.processor import Refined

    refined = Refined(
        device=device,
        model_dir=model_dir,
        data_dir=data_dir,
        debug=debug,
        requires_redirects_and_disambig=True,
        backward_coref=True,
    )
    refined.preprocessor.zero_features = False
    refined.preprocessor.zero_string_features = True
    refined.model.use_precomputed_descriptions = True
    refined.model.use_kg_layer = False

    with open(f"{output_file}_{process_part}-{total_parts}", "w") as output:
        file_open_func = gzip.open if ".gz" in input_file else open
        progress_bar = tqdm(
            total=6000000 // total_parts, desc=f"Process part {process_part}", disable=part_num != 0
        )
        with file_open_func(f"{input_file}", "r") as f:
            for line_number, line in enumerate(f):
                if line_number % total_parts == process_part:
                    progress_bar.update(n=1)
                    line = json.loads(line)
                    ents = refined.process_text(
                        text=line["text"],
                        max_batch_size=16,
                        prune_ner_types=True,
                        apply_class_check=True,
                        use_final_ed_score=False,
                        ds_format=True,
                    )

                    # delete old information from DS dataset
                    if "mentions" in line:
                        del line["mentions"]

                    if "sentences" in line:
                        for sent in line["sentences"]:
                            if "triples" in sent:
                                sent["triples"] = None
                    line["mentions"] = ents
                    output.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
