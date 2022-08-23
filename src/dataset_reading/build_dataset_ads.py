import json
from collections import defaultdict
from typing import Dict

from dataclasses import asdict, dataclass
from tqdm import tqdm

from dataset_reading.ads_dataset_creation.utils import fact_to_key, Key, Value, dm_item_to_value


grouped_results: Dict[Key, Dict[Value, int]] = defaultdict(dict)

i = 0
num_annotations = 0
with open("all.json", "r") as f:
    for line in tqdm(f, total=500000):
        line = json.loads(line)
        key = fact_to_key(line["value"]["fountRelation"])
        value = dm_item_to_value(line, left=True)
        if value not in grouped_results[key]:
            num_annotations += 1
            grouped_results[key][value] = 0
        grouped_results[key][value] += 1

        value = dm_item_to_value(line, left=False)
        if value not in grouped_results[key]:
            num_annotations += 1
            grouped_results[key][value] = 0
        grouped_results[key][value] += 1

print(f"Number of sentences {len(grouped_results)}")
print(f"Number of distinct annotations {num_annotations}")


with open("grouped_er_annotations.json", "w") as f:
    for k, v in grouped_results.items():
        f.write(json.dumps({k.sentence: [(asdict(value), cnt) for value, cnt in v.items()]}) + "\n")
