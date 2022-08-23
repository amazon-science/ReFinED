import os
import json

from typing import Optional, Dict, Set, List
from dataclasses import dataclass
from collections import defaultdict


@dataclass(frozen=True)
class Key:
    sentence: str
    source_uri: Optional[str] = None
    wikipedia_title: Optional[str] = None


@dataclass(frozen=True)
class Value:
    entity_resolution_error: str
    entity_span_error: str
    start: int
    end: int
    tkid: str
    text: str


def dm_item_to_value(dm_item, side: str):
    fact = dm_item["value"]["fountRelation"]

    entity_resolution_error = dm_item["value"][f"{side}EntityError"]["ResolutionError"]
    entity_span_error = dm_item["value"][f"{side}EntityError"]["Undercomplte/Overcomplete"]
    start = fact[f"{side}Entity"]["startIndex"]
    end = fact[f"{side}Entity"]["endIndex"]
    text = fact[f"{side}Entity"]["text"]
    tkid = fact[f"{side}Entity"]["id"].replace("evi_id:", "")

    return Value(
        start=start,
        end=end,
        text=text,
        tkid=tkid,
        entity_resolution_error=entity_resolution_error,
        entity_span_error=entity_span_error,
    )


def fact_to_key(fact):
    sentence = fact["sentence"]
    return Key(sentence)


def read_articles(file_path: str):
    with open(file_path, "r") as f:
        for line in f:
            article = json.loads(line)
            yield article


def load_wiki_to_article(ads_test_dir: str):
    """
    Read ads test datasets into memory - test datasets saved at 's3://fount.experiments/data/'
    If include all of [ads_test_dataset.json, ads_test_dataset_08-10-2020.json, ads_test_dataset_v2.json, non_wikipedia_ads_test_dataset.json]
    will be about 90k articles in total
    """
    ads_test_datasets = ["ads_test_dataset.json", "ads_test_dataset_08-10-2020.json",
                         "ads_test_dataset_v2.json", "non_wikipedia_ads_test_dataset.json"]

    wiki_title_to_article = {}

    for dset in ads_test_datasets:

        print(f"Reading {dset} into memory...")

        dset_path = os.path.join(ads_test_dir, dset)
        for article in read_articles(dset_path):
            wiki_title_to_article[article["title"]] = article["text"]

    return wiki_title_to_article


def read_annotations(dirpath: str, exclude_japanese: bool = True) -> List[Dict]:
    japanese_experiments = {"Wikipedia_new_test_split_Class_Checked_0.9-xwenduan"}

    for dirpath, _, filenames in os.walk(dirpath):
        for f in filenames:
            file_path = os.path.join(dirpath, f)
            annotation = json.load(open(file_path, "r"))

            # Filter out annotation in japanese
            value = annotation["value"]
            if exclude_japanese and "locale" in value and value["locale"] == "ja_JP":
                continue

            if exclude_japanese and "experiment_name" in value and value["experiment_name"] in japanese_experiments:
                continue

            if "source_uri" not in value:
                wikipedia_title = file_path.split("/")[5]
                wikipedia_title = wikipedia_title.replace("+", " ")
                source_uri = None
            else:
                source_uri = value["source_uri"]
                wikipedia_title = None

            yield source_uri, wikipedia_title, annotation


def read_webdata_annotations(dirpath: str, max_annotations: Optional[int] = None):
    count = 0
    for source_uri, wikipedia_title, annotation in read_annotations(dirpath=dirpath):
        if source_uri == "no_source_uri":
            yield source_uri, wikipedia_title, annotation
            count += 1

            if max_annotations is not None and count == max_annotations:
                break


def group_results_by_sentence(facts):
    grouped_results: Dict[Key, Set[Value]] = defaultdict(set)

    i = 0
    num_annotations = 0

    for source_uri, wiki_title, fact in facts:
        key = Key(sentence=fact["value"]["fountRelation"]["sentence"],
                  source_uri=source_uri,
                  wikipedia_title=wiki_title)

        for side in ["left", "right"]:
            value = dm_item_to_value(fact, side=side)

            if value not in grouped_results[key]:
                num_annotations += 1

            grouped_results[key].add(value)

    print(f"Number of sentences {len(grouped_results)}")
    print(f"Number of distinct annotations {num_annotations}")

    return grouped_results


def entity_correct(label: str):
    correct_labels = {"Yes", "Correctly Resolved", ""}
    incorrect_labels = {"No", "Incorrectly Resolved"}

    if label in correct_labels:
        return True

    if label in incorrect_labels:
        return False

    raise Exception(f"Don\'t recognize resolution label {label}")


def write_dataset_to_disk(dset: List[Dict], fpath: str):
    with open(fpath, "w") as wf:
        for article in dset:
            wf.write(json.dumps(article) + "\n")


def add_nonambiguous_annotations(dset: List[Dict]) -> List[Dict]:
    for article in dset:
        for mention in article["mentions"]:
            annotation = mention["annotation"]

            # No annotations for this mention
            if annotation is None:
                mention["er_annotation"] = None
                continue

            # Amgiguous cases where different annotators have annotated differently
            if len(set(annotation["entity_correct"])) > 1:
                mention["er_annotation"] = None
                continue

            mention["er_annotation"] = annotation["entity_correct"][0]

    return dset


def articles_with_a_false_mention(dset):
    negative_dset = []

    for article in dset:
        has_negative = False
        for mention in article["mentions"]:

            annotation = mention["er_annotation"]

            if annotation == False:
                has_negative = True
                break

        if has_negative:
            negative_dset.append(article)

    return negative_dset






