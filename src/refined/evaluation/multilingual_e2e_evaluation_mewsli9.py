from argparse import ArgumentParser
from refined.inference.processor import Refined
from refined.evaluation.evaluation import evaluate_on_docs
from refined.data_types.base_types import Span
from refined.dataset_reading.entity_linking.dataset_reader_multilingual import Datasets
from refined.resource_management.loaders import normalize_surface_form
from refined.resource_management.lmdb_wrapper import LmdbImmutableDict
import os
import pickle

def main():
    parser = ArgumentParser()
    parser.add_argument('--lang_title2wikidata', type=str,
                        default="../data_combine_11_languages_wikidata_all_eng_label_desc/additional_data",
                        help="path of lang_title2wikidataID-normalized_with_redirect.pkl", required=False)
    parser.add_argument('--mention2wikidata', type=str,
                        default="../data_combine_11_languages_wikidata_all_eng_label_desc/additional_data",
                        help="path of mention2wikidataID.lmdb", required=False)
    parser.add_argument('--model', type=str,
                        default="../finetune_models/mReFinED_Recall_9343",
                        help="model path", required=False)
    parser.add_argument('--wikidata', type=bool,
                        default=True,
                        help="true or false", required=False)
    parser.add_argument('--data', type=str,
                        default="../data_combine_11_languages_wikidata_all_eng_label_desc", required=False)
    cli_args = parser.parse_args()

    print(f"Loading mReFinED components....")
    languages = ['ar','de','en','es','fa','ja','sr','ta','tr']
    with open(os.path.join(cli_args.lang_title2wikidata,"lang_title2wikidataID-normalized_with_redirect.pkl"), "rb") as f:
        lang_title2wikidataID = pickle.load(f)
    mention2wikidataID = LmdbImmutableDict(os.path.join(cli_args.lang_title2wikidata,"mention2wikidataID.lmdb"))                  
    DATA_DIR = cli_args.data
    model = cli_args.model
    entity_set = 'wikidata' if cli_args.wikidata else 'wikipedia'
     
    refined = Refined.from_pretrained(model_name=model,
                                    entity_set=entity_set,
                                    use_precomputed_descriptions=False,
                                    data_dir=DATA_DIR,
                                    download_files=False)

    refined.preprocessor.candidate_generator.mention2wikidataID = mention2wikidataID
    refined.preprocessor.candidate_generator.lang_title2wikidataID = lang_title2wikidataID
    refined.model.ed_2.temperature_scaling = 0.02
    all_recall = 0
    all_tp = 0
    all_fn = 0

    for lang in languages:
        refined.preprocessor.lookups.pem = LmdbImmutableDict(os.path.join(DATA_DIR,f'wikidata_data/pem_{lang}.lmdb'))
        refined.preprocessor.candidate_generator.pem = LmdbImmutableDict(os.path.join(DATA_DIR,f'wikidata_data/pem_{lang}.lmdb'))
        refined.preprocessor.candidate_generator.language = lang
        
        datasets_dir = f"../mewsli_9_el_datasets/{lang}"
        dataset_name = f'mewsli-9-{lang}'
        
        datasets = Datasets(preprocessor=refined.preprocessor, datasets_path=datasets_dir)
        dataset_docs = datasets.get_mewsli_docs(filename=datasets_dir)
        metrics = evaluate_on_docs(refined=refined, docs=dataset_docs, dataset_name=dataset_name, 
                                    el=True,
                                    ed_threshold = 0.0,topk_eval=True,top_k=3)
        print('*****************************\n\n')
        print(f'Dataset name: {dataset_name}')
        print(metrics.get_summary())
        print('*****************************\n\n')
        all_recall+=metrics.get_recall()
        all_tp+=metrics.tp
        all_fn+=metrics.fn

    print(f"Average recall:{all_recall/len(languages)}")
    micro_avg = (all_tp/(all_tp+all_fn))*100
    print(f"Micro-avg:{micro_avg:.2f}")

if __name__ == '__main__':
    main()