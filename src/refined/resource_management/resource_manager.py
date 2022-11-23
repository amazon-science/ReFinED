import os
from copy import deepcopy
from typing import Dict, Tuple, Optional

from refined.constants.resources_constants import DATA_FILES_WIKIDATA, DATA_FILES_WIKIPEDIA, DATASET_DATA_FILES, \
    ADDITIONAL_DATA_FILES, model_name_to_files, TRAINING_DATA_FILES
from refined.resource_management.aws import S3Manager


def get_mmap_shape(file_path: str) -> Tuple[int, int]:
    shape = file_path.split("_")[-1][:-3]
    shape = [int(i) for i in shape.split("-")]
    return tuple(shape)


class ResourceManager:
    def __init__(self,
                 s3_manager: S3Manager,
                 entity_set: Optional[str],
                 model_name: Optional[str],
                 inference_ony: bool = True,
                 data_dir: Optional[str] = None,
                 datasets_dir: Optional[str] = None,
                 additional_data_dir: Optional[str] = None,
                 load_descriptions_tns: bool = False,
                 load_qcode_to_title: bool = False):
        self.entity_set = entity_set
        self.model_name = model_name
        self.inference_ony = inference_ony
        self.load_descriptions_tns = load_descriptions_tns
        self.load_qcode_to_title = load_qcode_to_title
        self.data_dir = os.path.join(os.path.expanduser('~'), '.cache', 'refined') if data_dir is None else data_dir
        self.datasets_dir = os.path.join(data_dir, 'datasets') if datasets_dir is None else datasets_dir
        self.additional_data_dir = os.path.join(data_dir, 'additional_data') if \
            additional_data_dir is None else additional_data_dir
        self.s3_manager = s3_manager

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.additional_data_dir, exist_ok=True)

    def download_data_if_needed(self):
        for resource_name, data_file in self.get_data_files_info().items():
            self.s3_manager.download_file_if_needed(
                s3_bucket=data_file["s3_bucket"],
                s3_key=data_file["s3_key"],
                output_file_path=data_file["local_filename"],
            )

    def download_models_if_needed(self):
        for resource_name, model_file in self.get_model_files_info().items():
            self.s3_manager.download_file_if_needed(
                s3_bucket=model_file["s3_bucket"],
                s3_key=model_file["s3_key"],
                output_file_path=model_file["local_filename"],
            )

    def download_datasets_if_needed(self):
        for resource_name, data_file in self.get_dataset_files_info().items():
            self.s3_manager.download_file_if_needed(
                s3_bucket=data_file["s3_bucket"],
                s3_key=data_file["s3_key"],
                output_file_path=data_file["local_filename"],
            )

    def download_additional_files_if_needed(self):
        """
        Additional files which includes files needed to run evaluation on datasets with Wikipedia titles.
        """
        for resource_name, data_file in self.get_additional_data_files_info().items():
            self.s3_manager.download_file_if_needed(
                s3_bucket=data_file["s3_bucket"],
                s3_key=data_file["s3_key"],
                output_file_path=data_file["local_filename"],
            )

    def download_training_files_if_needed(self):
        """
        Additional files which includes files needed to run evaluation on datasets with Wikipedia titles.
        """
        for resource_name, data_file in self.get_training_data_files_info().items():
            self.s3_manager.download_file_if_needed(
                s3_bucket=data_file["s3_bucket"],
                s3_key=data_file["s3_key"],
                output_file_path=data_file["local_filename"],
            )

    def get_model_files_info(self) -> Dict[str, Dict[str, str]]:
        assert self.model_name in model_name_to_files, f"model_name should be " \
                                                       f"one of {model_name_to_files.keys()} but " \
                                                       f"is {self.model_name}"

        assert self.entity_set in model_name_to_files[self.model_name]['model_description_embeddings'], \
            f"the model {self.model_name} does not have" \
            f"precomputed descriptions for the" \
            f"entity set {self.entity_set}. use one of" \
            f"the following" \
            f"{model_name_to_files[self.model_name]['model_description_embeddings'].keys()}"

        model_files = deepcopy(model_name_to_files[self.model_name])
        model_files['model_description_embeddings'] = model_files['model_description_embeddings'][self.entity_set]
        for resource_name, resource_locations in model_files.items():
            resource_locations['local_filename'] = os.path.join(
                self.data_dir, resource_locations["local_filename"]
            )
        if self.load_descriptions_tns and 'model_description_embeddings' in model_files:
            del model_files['model_description_embeddings']
        return model_files

    def get_data_files_info(self) -> Dict[str, Dict[str, str]]:
        entity_set_to_files: Dict[str, Dict[str, Dict[str, str]]] = {
            "wikidata": DATA_FILES_WIKIDATA,
            "wikipedia": DATA_FILES_WIKIPEDIA
        }

        assert self.entity_set in entity_set_to_files, f"entity_set should be one of" \
                                                       f" {entity_set_to_files.keys()} but " \
                                                       f"is {self.entity_set}"
        data_files = deepcopy(entity_set_to_files[self.entity_set])
        if self.inference_ony:
            data_files = {
                resource_name: data_file for resource_name, data_file
                in data_files.items() if data_file['needed_for_inference']
                                         or ((resource_name == 'descriptions_tns' and self.load_descriptions_tns)
                                             or (resource_name == 'qcode_to_wiki' and self.load_qcode_to_title))
            }
        for resource_name, resource_locations in data_files.items():
            resource_locations['local_filename'] = os.path.join(
                self.data_dir, resource_locations["local_filename"]
            )
        return data_files

    def get_additional_data_files_info(self) -> Dict[str, Dict[str, str]]:
        resource_to_file_path: Dict[str, Dict[str, str]] = deepcopy(ADDITIONAL_DATA_FILES)
        for resource_name, resource_locations in resource_to_file_path.items():
            resource_locations['local_filename'] = os.path.join(
                self.additional_data_dir, resource_locations["local_filename"]
            )
        return resource_to_file_path

    def get_training_data_files_info(self) -> Dict[str, Dict[str, str]]:
        resource_to_file_path: Dict[str, Dict[str, str]] = deepcopy(TRAINING_DATA_FILES)
        for resource_name, resource_locations in resource_to_file_path.items():
            resource_locations['local_filename'] = os.path.join(
                self.datasets_dir, resource_locations["local_filename"]
            )
        return resource_to_file_path

    def get_dataset_files_info(self) -> Dict[str, Dict[str, str]]:
        resource_to_file_path: Dict[str, Dict[str, str]] = deepcopy(DATASET_DATA_FILES)
        for resource_name, resource_locations in resource_to_file_path.items():
            resource_locations['local_filename'] = os.path.join(
                self.datasets_dir, resource_locations["local_filename"]
            )
        return resource_to_file_path

    def get_additional_data_files(self) -> Dict[str, str]:
        resource_to_file_path: Dict[str, Dict[str, str]] = self.get_additional_data_files_info()
        return {k: v['local_filename'] for k, v in resource_to_file_path.items()}

    def get_training_data_files(self) -> Dict[str, str]:
        resource_to_file_path: Dict[str, Dict[str, str]] = self.get_training_data_files_info()
        return {k: v['local_filename'] for k, v in resource_to_file_path.items()}

    def get_dataset_files(self) -> Dict[str, str]:
        resource_to_file_path: Dict[str, Dict[str, str]] = self.get_dataset_files_info()
        return {k: v['local_filename'] for k, v in resource_to_file_path.items()}

    def get_data_files(self) -> Dict[str, str]:
        resource_to_file_path: Dict[str, Dict[str, str]] = self.get_data_files_info()
        return {k: v['local_filename'] for k, v in resource_to_file_path.items()}

    def get_model_files(self) -> Dict[str, str]:
        resource_to_file_path: Dict[str, Dict[str, str]] = self.get_model_files_info()
        return {k: v['local_filename'] for k, v in resource_to_file_path.items()}
