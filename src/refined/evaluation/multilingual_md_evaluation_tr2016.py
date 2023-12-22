from argparse import ArgumentParser
from refined.inference.standalone_md import MentionDetector
from glob import glob
import json
from typing import Iterable
from refined.data_types.doc_types import Doc
from refined.data_types.base_types import Entity, Span
from refined.doc_preprocessing.preprocessor import Preprocessor
from refined.resource_management.resource_manager import ResourceManager
from refined.doc_preprocessing.wikidata_mapper import WikidataMapper
from refined.resource_management.loaders import normalize_surface_form
import os
import pandas as pd
from tqdm import tqdm
import torch
languages = ['de','es','fr','it']
device = "cpu" if not torch.cuda.is_available() else "cuda:0"   # fallback to using CPU
parser = ArgumentParser()
parser.add_argument('--model', type=str,
                        help="model path", required=True)
cli_args = parser.parse_args()
all_md_models = [cli_args.model]

class Datasets:
    def __init__(self,
                 datasets_path: str):
        self.datasets_path = datasets_path
        
    def get_tr2016_docs(self, filename: str, lang: str, include_spans: bool = True, include_gold_label: bool = False,
                             filter_not_in_kb: bool = True):
        doc_files = glob(filename+'/*.txt')
        mention_files = glob(filename+'/*.mentions.new')

        doc_files_dict = self.list2dict(doc_files)
        mention_files_dict = self.list2dict(mention_files)
        all_files = list(mention_files_dict.keys())

        doc_all = {}
        entity_span_all = {}
        md_span_all = {}
        final_doc_list = []
        for file in all_files[:]:
            doc = open(doc_files_dict[file]).read()
            mention_label = pd.read_csv(mention_files_dict[file], sep="\t")
            all_titles = mention_label[(mention_label["is_hard"]==1) & (mention_label["q_id"] != 0)][["start","end","non_en_title","q_id"]].values.tolist()
            if len(all_titles) == 0:
                continue
            md_spans = []
            spans = []
            for information in all_titles:
                start_idx = information[0]
                end_idx = information[1]
                non_eng_title = str(information[2]).replace("_", " ")
                q_id = information[3]
                md_spans.append(Span(text = doc[start_idx:end_idx], start = start_idx, ln = end_idx-start_idx, coarse_type="MENTION"))
                spans.append(Span(text = doc[start_idx:end_idx], start = start_idx, ln = end_idx-start_idx, gold_entity = Entity(wikidata_entity_id=q_id, wikipedia_entity_title=non_eng_title), coarse_type="MENTION"))

            if md_spans != []:
                entity_span_all.update({
                    file : spans
                })
                md_span_all.update({
                    file : md_spans
                })
                doc_all.update({
                    file : doc
                })
                final_doc_list.append(file)
            
        for key in final_doc_list[:]:
            text = doc_all[key]
            spans = entity_span_all[key] 
            md_spans = md_span_all[key]
            
            yield {'text':text,'spans':spans,'md_spans':md_spans,'doc_name':key}

    def list2dict(self, list_item):
        new_dict = {}
        for x in list_item:
            new_dict.update({
                x.split('/')[-1].replace(".mentions","").replace(".txt","").replace(".new","").replace(".removed_incorrect_labels",""):x
            })
        return new_dict
    
def main():
    for model_dir in all_md_models:
        print(f"Model:{model_dir}")
        md = MentionDetector.init_from_pretrained(model_dir=model_dir, device=device)

        recall_all = 0
        for lang in languages:
            datasets_dir = f'../tr2016/{lang}/test'

            datasets = Datasets(datasets_path=datasets_dir)
            dataset_docs = datasets.get_tr2016_docs(filename=datasets_dir, lang=lang)

            tp_md_all = 0
            fn_md_all = 0
            for doc in tqdm(dataset_docs):
                spans = md.process_text(doc['text'])

                pred_spans_md = {(normalize_surface_form(span[2]), span[0]) for span in spans}
                gold_spans_md = {(normalize_surface_form(span.text), span.start) for span in doc['md_spans']}

                tp_md_all += len(pred_spans_md & gold_spans_md)
                fn_md_all += len(gold_spans_md - pred_spans_md)

            recall = (tp_md_all / (tp_md_all + fn_md_all + 1e-8 * 1.0))*100
            recall_all+=recall
            print(f"Language:{lang}, Recall:{recall}")
        print(f"Total recall:{(recall_all/len(languages)):.2f}")

if __name__ == '__main__':
    main()