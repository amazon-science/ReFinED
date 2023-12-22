import json
from typing import Iterable

from refined.data_types.doc_types import Doc
from refined.data_types.base_types import Entity, Span
from refined.doc_preprocessing.preprocessor import Preprocessor
from refined.resource_management.resource_manager import ResourceManager
from refined.doc_preprocessing.wikidata_mapper import WikidataMapper
import os
import pandas as pd
from glob import glob
from typing import Dict, Set, Iterator, Tuple, List, Optional

class Datasets:
    def __init__(self,
                 preprocessor: Preprocessor,
                 datasets_path: str):
        self.preprocessor = preprocessor
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
        for file in all_files:
            with open(doc_files_dict[file]) as document:
                doc = document.read()
            mention_label = pd.read_csv(mention_files_dict[file], sep="\t")
            all_titles = mention_label[(mention_label["is_hard"]==1) & (mention_label["q_id"] != 0)][["start","end","non_en_title","q_id"]].values.tolist()
            if not all_titles: # no hard mention
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

            if spans is None:
                yield Doc.from_text(
                    text=text,
                    preprocessor=self.preprocessor
                )
                
            else:
                yield Doc.from_text_with_spans(
                    text=text, spans=spans, md_spans=md_spans, preprocessor=self.preprocessor
                )
    
    

    def list2dict(self, list_item: List[str]):
        new_dict = {}
        for x in list_item:
            new_dict.update({
                x.split('/')[-1].replace(".mentions","").replace(".txt","").replace(".new",""):x # file names example: A1.mentions and A1.txt (mentions and text are separated)
            })
        return new_dict

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

            if spans is None:
                yield Doc.from_text(
                    text=text,
                    preprocessor=self.preprocessor
                )
                
            else:
                yield Doc.from_text_with_spans(
                    text=text, spans=spans, md_spans=md_spans, preprocessor=self.preprocessor
                )
    
    def _process_spans(self, merged_data, include_gold_label: bool = True):
        doc_id_span_dict = {}
        doc_id_md_span_dict = {}
        last_doc_id = ''
        spans = []
        md_spans = []
        for i,row in merged_data.iterrows():
            docid = row['docid']
            if last_doc_id == '' or docid != last_doc_id: # if doc_id has been changed (found new doc), we will save the value in line 124-125
                if last_doc_id != ''  and docid != last_doc_id: # Is this not the first document and is this the new document? 
                    doc_id_span_dict[last_doc_id] = spans # save value
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
            with open(os.path.join(text_dir,file), "r") as text_file:
                text = text_file.read()
            text_dict[file] = text
        return text_dict

    def _add_wiki_text(self, docs_data, text_dict):
        text = []
        for docid in docs_data['docid']:
            text.append(text_dict[docid])
        docs_data['text'] = text
        return docs_data
