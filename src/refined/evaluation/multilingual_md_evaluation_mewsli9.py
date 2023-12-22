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
languages = ['ar','de','en','es','fa','ja','sr','ta','tr']
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

    def get_mewsli_docs(self, filename: str, include_spans: bool = True, include_gold_label: bool = False,
                             filter_not_in_kb: bool = True):
        mentions_file = os.path.join(filename, 'mentions.tsv') 
        docs_file = os.path.join(filename, 'docs.tsv')
        mentions_data = pd.read_csv(mentions_file,sep="\t", header = 0).rename(columns = {'url': 'url_mentions'}, inplace = False)
        docs_data = pd.read_csv(docs_file,  sep="\t", header=0).rename(columns = {'url': 'url_docs'}, inplace = False)
        text_dict = self._get_text_files(os.path.join(filename, 'text'))
        docs_data_with_text = self._add_wiki_text(docs_data, text_dict)
        merged_data = pd.merge(docs_data_with_text, mentions_data,  on =['docid'])
        doc_spans, doc_md_spans = self._process_spans(merged_data)
        for key in doc_spans.keys():
            text = text_dict[key]
            spans = doc_spans[key] 
            md_spans = doc_md_spans[key]
            yield {'text':text,'spans':spans,'md_spans':md_spans,'doc_name':key}
    
    def _process_spans(self, merged_data, include_gold_label: bool = True):
        doc_id_span_dict = {}
        doc_id_md_span_dict = {}
        last_doc_id = ''
        spans = []
        md_spans = []
        for i,row in merged_data.iterrows():
            docid = row['docid']
            if last_doc_id == '' or docid != last_doc_id:
                if last_doc_id != ''  and docid != last_doc_id:
                    doc_id_span_dict[last_doc_id] = spans
                    doc_id_md_span_dict[last_doc_id] = md_spans
                last_doc_id = docid
                spans = []
                md_spans=[]
                
            md_spans.append(Span(text = row['mention'], start = row['position'], ln = row['length'], coarse_type="X", coarse_mention_type="MENTION"))
            if include_gold_label:
                spans.append(Span(text = row['mention'], start = row['position'], ln = row['length'], gold_entity = Entity(wikidata_entity_id=row['qid'], wikipedia_entity_title=row['title']), coarse_type="MENTION"))
            else:
                spans.append(Span(text = row['mention'], start = row['position'], ln = row['length'], coarse_type="X", coarse_mention_type="MENTION"))
    
        if last_doc_id not in doc_id_span_dict.keys() and not spans:
            doc_id_span_dict[last_doc_id] = spans
            doc_id_md_span_dict[last_doc_id] = md_spans
        return doc_id_span_dict, doc_id_md_span_dict
    
    def _get_text_files(self, text_dir):
        text_dict = {}
        for file in os.listdir(text_dir):
            text = open(os.path.join(text_dir,file), "r").read()
            text_dict[file] = text
        return text_dict

    def _add_wiki_text(self, docs_data, text_dict):
        text = []
        for docid in docs_data['docid']:
            text.append(text_dict[docid])
        docs_data['text'] = text
        return docs_data

def main():
    for model_dir in all_md_models:
        print(f"Model:{model_dir}")
        md = MentionDetector.init_from_pretrained(model_dir=model_dir, device=device)

        recall_all = 0
        for lang in languages:
            datasets_dir = f"../mewsli_9_el_datasets/{lang}"

            datasets = Datasets(datasets_path=datasets_dir)
            dataset_docs = datasets.get_mewsli_docs(filename=datasets_dir)

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