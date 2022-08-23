import os
from typing import Iterable, Optional, Union, Dict, Tuple

import smart_open

from utilities.aws import S3Manager

s3_resource_bucket = "refined.public"
s3_resource_prefix = "2022_aug/"
s3_datasets_prefix = "2022_aug/datasets/"

DATA_FILES_WIKIDATA = {
    "wiki_pem": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/wiki_pem.json",
        "local_filename": "wikidata_data/wiki_pem.json",
    },
    "class_to_label": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/class_to_label.json",
        "local_filename": "wikidata_data/class_to_label.json",
    },
    "human_qcodes": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/human_qcodes.json",
        "local_filename": "wikidata_data/human_qcodes.json",
    },
    "class_to_idx": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/class_to_idx.json",
        "local_filename": "wikidata_data/class_to_idx.json",
    },
    "qcode_to_idx": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/qcode_to_idx.json",
        "local_filename": "wikidata_data/qcode_to_idx.json",
    },
    "qcode_idx_to_class_idx": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/qcode_to_class_tns_33831487-200.np",
        "local_filename": "wikidata_data/qcode_to_class_tns_33831487-200.np",
    },
    "subclasses": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/subclass_p279.json",
        "local_filename": "wikidata_data/subclass_p279.json",
    },
    "disambiguation_qcodes": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/disambiguation_qcodes.txt",
        "local_filename": "wikidata_data/disambiguation_qcodes.txt",
    },
    "redirects": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/redirects.json",
        "local_filename": "wikidata_data/redirects.json",
    },
    "qcode_to_label": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/qcode_to_label.json",
        "local_filename": "wikidata_data/qcode_to_label.json",
    },
    "descriptions_tns": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/descriptions_tns.pt",
        "local_filename": "wikidata_data/descriptions_tns.pt",
    },
    "wiki_to_qcode": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikidata_data/enwiki.json",
        "local_filename": "wikidata_data/enwiki.json",
    }
}

DATA_FILES_WIKIPEDIA = {
    "wiki_pem": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/wiki_pem.json",
        "local_filename": "wikipedia_data/wiki_pem.json",
    },
    "class_to_label": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/class_to_label.json",
        "local_filename": "wikipedia_data/class_to_label.json",
    },
    "human_qcodes": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/human_qcodes.json",
        "local_filename": "wikipedia_data/human_qcodes.json",
    },
    "class_to_idx": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/class_to_idx.json",
        "local_filename": "wikipedia_data/class_to_idx.json",
    },
    "qcode_to_idx": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/qcode_to_idx.json",
        "local_filename": "wikipedia_data/qcode_to_idx.json",
    },
    "qcode_idx_to_class_idx": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/qcode_to_class_tns_6269457-138.np",
        "local_filename": "wikipedia_data/qcode_to_class_tns_6269457-138.np",
    },
    "subclasses": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/subclass_p279.json",
        "local_filename": "wikipedia_data/subclass_p279.json",
    },
    "disambiguation_qcodes": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/disambiguation_qcodes.txt",
        "local_filename": "wikipedia_data/disambiguation_qcodes.txt",
    },
    "redirects": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/redirects.json",
        "local_filename": "wikipedia_data/redirects.json",
    },
    "qcode_to_label": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/qcode_to_label.json",
        "local_filename": "wikipedia_data/qcode_to_label.json",
    },
    "descriptions_tns": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/descriptions_tns.pt",
        "local_filename": "wikipedia_data/descriptions_tns.pt",
    },
    "wiki_to_qcode": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix + "wikipedia_data/enwiki.json",
        "local_filename": "wikipedia_data/enwiki.json",
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
    "model_description_embeddings_wikidata": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix
                  + "wikipedia_model/precomputed_entity_descriptions_emb_wikidata_33831487-300.np",
        "local_filename": "wikipedia_model/precomputed_entity_descriptions_emb_wikidata_33831487-300.np",
    },
    "model_description_embeddings_wikipedia": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix
                  + "wikipedia_model/precomputed_entity_descriptions_emb_wikipedia_6269457-300.np",
        "local_filename": "wikipedia_model/precomputed_entity_descriptions_emb_wikipedia_6269457-300.np",
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
    "model_description_embeddings_wikidata": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix
        + "wikipedia_model_with_numbers/precomputed_entity_descriptions_emb_wikidata_33831487-300.np",
        "local_filename": "wikipedia_model_with_numbers/precomputed_entity_descriptions_emb_wikidata_33831487-300.np",
    },
    "model_description_embeddings_wikipedia": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_resource_prefix
                  + "wikipedia_model_with_numbers/precomputed_entity_descriptions_emb_wikipedia_6269457-300.np",
        "local_filename": "wikipedia_model_with_numbers/precomputed_entity_descriptions_emb_wikipedia_6269457-300.np",
    }
}


DATASET_DATA_FILES = {
    "aida_test": {
        "s3_bucket": s3_resource_bucket,
        "s3_key": s3_datasets_prefix + "aida_test.json",
        "local_filename": "aida_test.json",
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
}


def get_data_files(entity_set: str) -> Dict:

    entity_set_to_files = {
        "wikidata": DATA_FILES_WIKIDATA,
        "wikipedia": DATA_FILES_WIKIPEDIA
    }

    assert entity_set in entity_set_to_files, f"entity_set should be one of {entity_set_to_files.keys()} but " \
                                                    f"is {entity_set}"

    return entity_set_to_files[entity_set]


def get_model_files(model_name: str) -> Dict:

    model_name_to_files = {
        "wikipedia_model": WIKIPEDIA_MODEL,
        "wikipedia_model_with_numbers": WIKIPEDIA_MODEL_WITH_NUMBERS
    }

    assert model_name in model_name_to_files, f"model_name should be one of {model_name_to_files.keys()} but " \
                                                    f"is {model_name}"

    return model_name_to_files[model_name]


def get_mmap_shape(file_path: str) -> Tuple[int, int]:
    shape = file_path.split("_")[-1][:-3]
    shape = [int(i) for i in shape.split("-")]
    return tuple(shape)


def get_resource(
    resource_name: str, mode: str, path: Optional[str] = None
) -> Union[bytearray, Iterable[str]]:
    assert mode in {"r", "rb"}, "mode must be r or rb."

    file = smart_open.smart_open(path, mode=mode)
    if mode == "rb":
        # return all bytes for binary files
        data = file.read()
        file.close()
    else:
        # return a generator for each line of the file when mode is 'r'
        data = file.readlines()
        file.close()
    return data


class ResourceManager:
    def __init__(self, s3_manager: S3Manager, data_dir: str):
        # by default will download to ~/.cache/refined/
        # if data_dir with files is provided it will not download
        self.s3_manager = s3_manager
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir

    def download_data_if_needed(self, entity_set: str, load_descriptions_tns: bool = False):

        data_files = get_data_files(entity_set=entity_set)

        for resource_name, data_file in data_files.items():
            if resource_name == 'descriptions_tns' and not load_descriptions_tns:
                continue
            print(resource_name, data_file)
            self.s3_manager.download_file_if_needed(
                s3_bucket=data_file["s3_bucket"],
                s3_key=data_file["s3_key"],
                output_file_path=os.path.join(self.data_dir, data_file["local_filename"]),
            )

    def download_models_if_needed(self, model_name: str):

        model_files = get_model_files(model_name=model_name)

        for resource_name, model_file in model_files.items():

            print(resource_name, model_file)
            self.s3_manager.download_file_if_needed(
                s3_bucket=model_file["s3_bucket"],
                s3_key=model_file["s3_key"],
                output_file_path=os.path.join(self.data_dir, model_file["local_filename"]),
            )

    def download_datasets(self):
        for resource_name, data_file in DATASET_DATA_FILES.items():
            self.s3_manager.download_file_if_needed(
                s3_bucket=data_file["s3_bucket"],
                s3_key=data_file["s3_key"],
                output_file_path=os.path.join(self.data_dir, data_file["local_filename"]),
            )
