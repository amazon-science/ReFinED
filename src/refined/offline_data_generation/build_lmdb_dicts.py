import os
import shutil

from refined.resource_management.aws import S3Manager
from refined.resource_management.lmdb_wrapper import LmdbImmutableDict
from refined.resource_management.loaders import load_pem, load_qcode_to_idx, load_subclasses, load_redirects, \
    load_wikipedia_to_qcode, load_labels
from refined.resource_management.resource_manager import ResourceManager


def build_lmdb_dicts(preprocess_all_data_dir: str, keep_all_entities: bool):
    # This file intentionally duplicates files to avoid risk of losing files by overwritten them by mistake
    # with the resource_manager class.
    # Replace shutil.copy(src, dst) to os.rename(src, dst) to avoid duplicating files.
    # keep_all_entities=True means Wikidata, False means Wikipedia
    # converts the output of preprocess_all.py steps into lmdb dictionaries and organises the file structure
    # based on resource_constants.py and the resource_manager.
    # nest the output in the preprocess_all_data_dir data_dir
    # wikipedia or wikidata

    new_data_dir = os.path.join(preprocess_all_data_dir, "organised_data_dir")
    os.makedirs(new_data_dir, exist_ok=True)

    entity_set = "wikidata" if keep_all_entities else "wikipedia"
    os.makedirs(os.path.join(new_data_dir, f"{entity_set}_data"), exist_ok=True)
    os.makedirs(os.path.join(new_data_dir, "additional_data"), exist_ok=True)

    resource_manager = ResourceManager(entity_set=entity_set,
                                       data_dir=new_data_dir,
                                       model_name=None,
                                       s3_manager=S3Manager(),
                                       load_descriptions_tns=True,
                                       load_qcode_to_title=True,
                                       inference_ony=False
                                       )
    # get output file paths for each file
    # TODO: resource manager should return class instead of dictionary to make the code more maintainable
    data_files = resource_manager.get_data_files()
    training_data_files = resource_manager.get_training_data_files()
    additional_data_files = resource_manager.get_additional_data_files()

    # TODO what about "roberta_base_tokenizer_merges"

    # data files
    # Set max_cands to 30 to save space
    pem = load_pem(pem_file=os.path.join(preprocess_all_data_dir, "wiki_pem.json"), max_cands=None)
    LmdbImmutableDict.from_dict(pem, output_file_path=data_files["wiki_pem"])
    del pem
    shutil.copy(os.path.join(preprocess_all_data_dir, "class_to_label.json"), data_files["class_to_label"])
    shutil.copy(os.path.join(preprocess_all_data_dir, "human_qcodes.json"), data_files["human_qcodes"])
    shutil.copy(os.path.join(preprocess_all_data_dir, "class_to_idx.json"), data_files["class_to_idx"])
    qcode_to_idx = load_qcode_to_idx(filename=os.path.join(preprocess_all_data_dir, "qcode_to_idx.json"))
    LmdbImmutableDict.from_dict(qcode_to_idx, output_file_path=data_files["qcode_to_idx"])
    del qcode_to_idx

    qcode_to_class_tns_filename = [x for x in os.listdir(preprocess_all_data_dir) if "qcode_to_class_tns_" in x][0]
    # data_files["qcode_idx_to_class_idx"] cannot be used at the moment as the size is encoded in the filename
    shutil.copy(os.path.join(preprocess_all_data_dir, qcode_to_class_tns_filename),
                os.path.join(new_data_dir, f"{entity_set}_data", qcode_to_class_tns_filename))
    subclasses, _ = load_subclasses(os.path.join(preprocess_all_data_dir, "subclass_p279.json"))
    subclasses_with_lists = dict()
    for k, v in subclasses.items():
        subclasses_with_lists[k] = list(v)
    LmdbImmutableDict.from_dict(subclasses_with_lists, output_file_path=data_files["subclasses"])
    shutil.copy(os.path.join(preprocess_all_data_dir, "descriptions_tns.pt"),
                data_files["descriptions_tns"])
    qcode_to_wiki = load_qcode_to_idx(os.path.join(preprocess_all_data_dir, "qcode_to_idx.json"))
    LmdbImmutableDict.from_dict(qcode_to_wiki, output_file_path=data_files["qcode_to_wiki"])

    # training data files
    shutil.copy(os.path.join(preprocess_all_data_dir, "wikipedia_links_aligned_spans.json"),
                training_data_files["wikipedia_training_dataset"])

    # additional_data_files
    redirects = load_redirects(os.path.join(preprocess_all_data_dir, "redirects.json"))
    LmdbImmutableDict.from_dict(redirects, output_file_path=additional_data_files["redirects"])
    shutil.copy(os.path.join(preprocess_all_data_dir, "disambiguation_qcodes.txt"),
                additional_data_files["disambiguation_qcodes"])
    wiki_to_qcode = load_wikipedia_to_qcode(os.path.join(preprocess_all_data_dir, 'enwiki.json'))
    LmdbImmutableDict.from_dict(wiki_to_qcode, output_file_path=additional_data_files["wiki_to_qcode"])
    qcode_to_label = load_labels(os.path.join(preprocess_all_data_dir, 'qcode_to_label.json'))
    LmdbImmutableDict.from_dict(qcode_to_label, output_file_path=additional_data_files["qcode_to_label"])

    # add other expected files in the data_dir
    for resource_name, data_file in resource_manager.get_data_files_info().items():
        if resource_name in {
            "roberta_base_model",
            "roberta_base_model_config",
            "roberta_base_vocab",
            "roberta_base_tokenizer_merges",
            "nltk_sentence_splitter_english",
        }:
            resource_manager.s3_manager.download_file_if_needed(
                s3_bucket=data_file["s3_bucket"],
                s3_key=data_file["s3_key"],
                output_file_path=data_file["local_filename"],
            )

    print(f"Data is now contained in {preprocess_all_data_dir}/organised_data_dir/ which can be used by the "
          f"`PreprocessorInferenceOnly` class and `ResourceManager` class")
    print(f"Adjust the `resource_constants.py` file to 'qcode_to_class_tns_33831487-200.np' to the match the name of"
          f"the file in your {preprocess_all_data_dir}/organised_data_dir/wikidata_data or "
          f"{preprocess_all_data_dir}/organised_data_dir/wikipedia_data.")
    print("To use this with a model copy the model files into the directory and update the resource_constants.py.")
    print("Ensure download=False when using this new data_dir to avoid overwriting files.")

    # train.py can now be run with {preprocess_all_data_dir}/{organised_data_dir}
