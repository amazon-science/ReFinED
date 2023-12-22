from typing import Dict, Any
from refined.training.train.training_args import parse_training_args

s3_resource_bucket = "refined.public"
s3_resource_prefix = "2022_oct/"
s3_datasets_prefix = "2022_oct/datasets/"

model = 'bert-base-multilingual-cased'

DATA_FILES_WIKIDATA = {
    "roberta_base_model": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + f"wikidata_data/{model}/pytorch_model.bin",
        "local_filename": f"{model}/pytorch_model.bin",
        "needed_for_inference": True
    },
    "roberta_base_model_config": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + f"wikidata_data/{model}/config.json",
        "local_filename": f"{model}/config.json",
        "needed_for_inference": True
    },
    "roberta_base_vocab": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + f"wikidata_data/{model}/vocab.json",
        "local_filename": f"{model}/vocab.json",
        "needed_for_inference": True
    },
    "roberta_base_tokenizer_merges": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + f"wikidata_data/{model}/merges.txt",
        "local_filename": f"{model}/merges.txt",
        "needed_for_inference": True
    },
    "wiki_pem": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/pem.lmdb",
        "local_filename": "wikidata_data/pem.lmdb",
        "needed_for_inference": True
    },
    "class_to_label": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/class_to_label.json",
        "local_filename": "wikidata_data/class_to_label.json",
        "needed_for_inference": True 
    },
    "human_qcodes": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/human_qcodes.json",
        "local_filename": "wikidata_data/human_qcodes.json", # used for person name co-reference
        "needed_for_inference": True 
    },
    "class_to_idx": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/class_to_idx.json",
        "local_filename": "wikidata_data/class_to_idx.json",
        "needed_for_inference": True
    },
    "qcode_to_idx": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/qcode_to_idx.lmdb",
        "local_filename": "wikidata_data/qcode_to_idx.lmdb",
        "needed_for_inference": True
    },
    "qcode_idx_to_class_idx": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/qcode_to_class_tns_38520582-122.np",
        "local_filename": "wikidata_data/qcode_to_class_tns_38520582-122.np",  
        "needed_for_inference": True
    },

    "subclasses": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/subclasses.lmdb",
        "local_filename": "wikidata_data/subclasses.lmdb",
        "needed_for_inference": True  # used when apply_class_check=True
    },
    "qcode_to_label": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/qcode_to_label.json",
        "local_filename": "wikidata_data/qcode_to_label.json",
        "needed_for_inference": False
    },
    "descriptions_tns": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/descriptions_tns.pt",
        "local_filename": "wikidata_data/descriptions_tns.pt",
        "needed_for_inference": False  # only needed if use_precomputed_desc_embedding=False
    },
    "qcode_to_wiki": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/qcode_to_wiki.lmdb",
        "local_filename": "wikidata_data/qcode_to_wiki.lmdb",
        "needed_for_inference": False  # only needed if map to Wikipedia titles
    },
    "nltk_sentence_splitter_english": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/nltk_sentence_splitter_english.pickle",
        "local_filename": "wikidata_data/nltk_sentence_splitter_english.pickle",
        "needed_for_inference": True  # only needed if map to Wikipedia titles
    }
}
DATA_FILES_WIKIPEDIA = {
    "roberta_base_model": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + f"wikipedia_data/{model}/pytorch_model.bin",
        "local_filename": f"{model}/pytorch_model.bin",
        "needed_for_inference": True
    },
    "roberta_base_model_config": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + f"wikipedia_data/{model}/config.json",
        "local_filename": f"{model}/config.json",
        "needed_for_inference": True
    },
    "roberta_base_vocab": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + f"wikipedia_data/{model}/vocab.json",
        "local_filename": f"{model}/vocab.json",
        "needed_for_inference": True
    },
    "roberta_base_tokenizer_merges": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + f"wikipedia_data/{model}/merges.txt",
        "local_filename": f"{model}/merges.txt",
        "needed_for_inference": True
    },
    "wiki_pem": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/pem.lmdb",
        "local_filename": "wikipedia_data/pem.lmdb", 
        "needed_for_inference": True
    },
    "class_to_label": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/class_to_label.json",
        "local_filename": "wikipedia_data/class_to_label.json",
        "needed_for_inference": True
    },
    "human_qcodes": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/human_qcodes.json",
        "local_filename": "wikipedia_data/human_qcodes.json",
        "needed_for_inference": True
    },
    "class_to_idx": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/class_to_idx.json",
        "local_filename": "wikipedia_data/class_to_idx.json",
        "needed_for_inference": True
    },
    "qcode_to_idx": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/qcode_to_idx.lmdb",
        "local_filename": "wikipedia_data/qcode_to_idx.lmdb",
        "needed_for_inference": True
    },
    "qcode_idx_to_class_idx": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/qcode_to_class_tns_9050844-117.np",
        "local_filename": "wikipedia_data/qcode_to_class_tns_9050844-117.np",
        "needed_for_inference": True
    },
    "subclasses": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/subclasses.lmdb",
        "local_filename": "wikipedia_data/subclasses.lmdb",
        "needed_for_inference": True
    },
    "qcode_to_label": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/qcode_to_label.json",
        "local_filename": "wikipedia_data/qcode_to_label.json",
        "needed_for_inference": False
    },
    "descriptions_tns": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/descriptions_tns.pt",
        "local_filename": "wikipedia_data/descriptions_tns.pt",
        "needed_for_inference": False
    },
    "qcode_to_wiki": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/qcode_to_wiki.lmdb",
        "local_filename": "wikipedia_data/qcode_to_wiki.lmdb",
        "needed_for_inference": False  # only needed if map to Wikipedia titles
    },
    "nltk_sentence_splitter_english": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/nltk_sentence_splitter_english.pickle",
        "local_filename": "wikipedia_data/nltk_sentence_splitter_english.pickle",
        "needed_for_inference": True  # only needed if map to Wikipedia titles
    }
}
WIKIPEDIA_MODEL = {
    # pretrained models
    "model": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_model/model.pt",
        "local_filename": "wikipedia_model/model.pt",
    },
    "model_config": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_model/config.json",
        "local_filename": "wikipedia_model/config.json",
    },
    "model_description_embeddings": {
        "wikipedia": {
            "s3_bucket": s3_resource_bucket,
            "s3_key": s3_resource_prefix +
                      "wikipedia_model/precomputed_entity_descriptions_emb_wikipedia_6269457-300.np",
            "local_filename": "wikipedia_model/precomputed_entity_descriptions_emb_wikipedia_6269457-300.np",
        },
        "wikidata": {
            "s3_bucket": s3_resource_bucket,
            "s3_key": s3_resource_prefix +
                      "wikipedia_model/precomputed_entity_descriptions_emb_wikidata_33831487-300.np",
            "local_filename": "wikipedia_model/precomputed_entity_descriptions_emb_wikidata_33831487-300.np"
        }
    }
}
WIKIPEDIA_MODEL_WITH_NUMBERS = {
    # pretrained models
    "model": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_model_with_numbers/model.pt",
        "local_filename": "wikipedia_model_with_numbers/model.pt",
    },
    "model_config": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_model_with_numbers/config.json",
        "local_filename": "wikipedia_model_with_numbers/config.json",
    },
    "model_description_embeddings": {
        "wikipedia": {
            "s3_bucket": s3_resource_bucket,
            "s3_key": s3_resource_prefix +
                      "wikipedia_model_with_numbers/precomputed_entity_descriptions_emb_wikipedia_6269457-300.np",
            "local_filename": "wikipedia_model_with_numbers/precomputed_entity_descriptions_emb_wikipedia_6269457-300"
                              ".np",
        },
        "wikidata": {
            "s3_bucket": s3_resource_bucket,
            "s3_key": s3_resource_prefix +
                      "wikipedia_model_with_numbers/precomputed_entity_descriptions_emb_wikidata_33831487-300.np",
            "local_filename": "wikipedia_model_with_numbers/precomputed_entity_descriptions_emb_wikidata_33831487-300"
                              ".np",
        }
    }
}

AIDA_MODEL = {
    # pretrained models
    "model": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "fine_tuned_models/aida_fine_tuned_el/model.pt",
        "local_filename": "fine_tuned_models/aida_fine_tuned_el/model.pt",
    },
    "model_config": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "fine_tuned_models/aida_fine_tuned_el/config.json",
        "local_filename": "fine_tuned_models/aida_fine_tuned_el/config.json",
    },
    "model_description_embeddings": {
        "wikipedia": {
            "s3_bucket": s3_resource_bucket,
            "s3_key": s3_resource_prefix +
                      "fine_tuned_models/aida_fine_tuned_el/precomputed_entity_descriptions_emb_wikipedia_6269457-300.np",
            "local_filename": "fine_tuned_models/aida_fine_tuned_el/precomputed_entity_descriptions_emb_wikipedia_6269457-300"
                              ".np",
        },
        "wikidata": {
            "s3_bucket": s3_resource_bucket,
            "s3_key": s3_resource_prefix +
                      "fine_tuned_models/aida_fine_tuned_el/precomputed_entity_descriptions_emb_wikidata_33831487-300.np",
            "local_filename": "fine_tuned_models/aida_fine_tuned_el/precomputed_entity_descriptions_emb_wikidata_33831487-300"
                              ".np",
        }
    }
}

QUESTIONS_MODEL = {
    # pretrained models
    "model": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "fine_tuned_models/web_qsp_el/model.pt",
        "local_filename": "fine_tuned_models/web_qsp_el/model.pt",
    },
    "model_config": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "fine_tuned_models/web_qsp_el/config.json",
        "local_filename": "fine_tuned_models/web_qsp_el/config.json",
    },
    "model_description_embeddings": {
        "wikipedia": {
            "s3_bucket": s3_resource_bucket,
            "s3_key": s3_resource_prefix +
                      "fine_tuned_models/web_qsp_el/precomputed_entity_descriptions_emb_wikipedia_6269457-300.np",
            "local_filename": "fine_tuned_models/web_qsp_el/precomputed_entity_descriptions_emb_wikipedia_6269457-300"
                              ".np",
        },
        "wikidata": {
            "s3_bucket": s3_resource_bucket,
            "s3_key": s3_resource_prefix +
                      "fine_tuned_models/web_qsp_el/precomputed_entity_descriptions_emb_wikidata_33831487-300.np",
            "local_filename": "fine_tuned_models/web_qsp_el/precomputed_entity_descriptions_emb_wikidata_33831487-300"
                              ".np",
        }
    }
}

model_name_to_files: Dict[str, Dict[str, Dict[str, Any]]] = {
    "wikipedia_model": WIKIPEDIA_MODEL,
    "wikipedia_model_with_numbers": WIKIPEDIA_MODEL_WITH_NUMBERS,
    "aida_model": AIDA_MODEL,
    "questions_model": QUESTIONS_MODEL
}

DATASET_DATA_FILES = {
    "aida_test": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "aida_test.json",
        "local_filename": "aida_test.json",
    },
    "aida_dev": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "aida_dev.json",
        "local_filename": "aida_dev.json",
    },
    "aida_train": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "aida_train.json",
        "local_filename": "aida_train.json",
    },
    "ace2004": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "ace2004_parsed.json",
        "local_filename": "ace2004_parsed.json",
    },
    "aquaint": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "aquaint_parsed.json",
        "local_filename": "aquaint_parsed.json",
    },
    "clueweb": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "clueweb_parsed.json",
        "local_filename": "clueweb_parsed.json",
    },
    "msnbc": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "msnbc_parsed.json",
        "local_filename": "msnbc_parsed.json",
    },
    "wikipedia": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "wikipedia_parsed.json",
        "local_filename": "wikipedia_parsed.json",
    },

    # MD datasets for training a standalone MD model
    "ontonotes_development_articles": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "ontonotes_development_articles.json",
        "local_filename": "ontonotes_development_articles.json",
    },
    "ontonotes_train_articles": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "ontonotes_train_articles.json",
        "local_filename": "ontonotes_train_articles.json",
    },
    "ontonotes_test_articles": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "ontonotes_test_articles.json",
        "local_filename": "ontonotes_test_articles.json",
    },

    "conll_dev": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "conll_dev.txt",
        "local_filename": "conll_dev.txt",
    },
    "conll_test": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "conll_test.txt",
        "local_filename": "conll_test.txt",
    },
    "conll_train": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "conll_train.txt",
        "local_filename": "conll_train.txt",
    },

    "webqsp_test_data_ner": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "webqsp_test_data_ner.json",
        "local_filename": "webqsp_test_data_ner.json",
    },
    "webqsp_training_data_ner": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "webqsp_training_data_ner.json",
        "local_filename": "webqsp_training_data_ner.json",
    },

    # TODO add shadowlink datasets
    # shadowlink_shadow_with_spacy_mentions.json
    # shadowlink_shadow_with_spacy_mentions_fulltext.json
    # shadowlink_shadow_with_spacy_mentions_subset.json
    # shadowlink_tail_with_spacy_mentions.json
    # shadowlink_top_with_spacy_mentions.json
    # shadowlink_top_with_spacy_mentions_fulltext.json
    # shadowlink_top_with_spacy_mentions_subset.json

    # WebQSP EL
    "webqsp_train_data_el": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "WebQSP_EL/train.jsonl",
        "local_filename": "webqsp_training_data_el.json",
    },
    "webqsp_dev_data_el": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "WebQSP_EL/dev.jsonl",
        "local_filename": "webqsp_dev_data_el.json",
    },
    "webqsp_test_data_el": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "WebQSP_EL/test.jsonl",
        "local_filename": "webqsp_test_data_el.json",
    }
}

ADDITIONAL_DATA_FILES = {
    "redirects": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/redirects.lmdb",
        "local_filename": "redirects.lmdb"
    },
    "disambiguation_qcodes": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/disambiguation_qcodes.txt",
        "local_filename": "disambiguation_qcodes.txt"
    },
    "wiki_to_qcode": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/wiki_to_qcode.lmdb",
        "local_filename": "wiki_to_qcode.lmdb"
    },
    "qcode_to_label": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/qcode_to_label.lmdb",
        "local_filename": "qcode_to_label.lmdb",
    }
}

TRAINING_DATA_FILES = {
    "wikipedia_training_dataset": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "training_data/wikipedia_links_aligned_spans.json",
        "local_filename": "wikipedia_links_aligned_spans.json"
    }
}
