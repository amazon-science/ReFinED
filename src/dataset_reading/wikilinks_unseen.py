import json
import os

from tqdm.auto import tqdm


def get_wikilinks_ned(split: str, dataset_path: str = "/Users/tayoola/Downloads/unseen_mentions"):

    assert split in {"train", "dev", "test"}
    split_to_filenames = {
        "train": [os.path.join(dataset_path, "train", f"train_{i}.json") for i in range(6)],
        "dev": [os.path.join(dataset_path, "dev.json")],
        "test": [os.path.join(dataset_path, "test.json")],
    }
    filenames = split_to_filenames[split]
    for filename in filenames:
        with open(filename, "r") as f:
            for line in tqdm(f):
                line = json.loads(line)
                left_context = " ".join(line["left_context"])
                right_context = " ".join(line["right_context"])
                mention = " ".join(line["mention_as_list"])
                text = left_context + " " + mention + " " + right_context
                entity_start = len(left_context) + 1
                entity_length = len(mention)
                wikipedia_title = line["y_title"]
                mention_text = text[entity_start : entity_start + entity_length]
                yield {
                    "text": text,
                    "title": wikipedia_title,
                    "start": entity_start,
                    "length": entity_length,
                    "mention_text": mention_text,
                }


train_titles = set()
test_titles = set()
for x in get_wikilinks_ned("train"):
    train_titles.add(x["title"])
    # break

for x in get_wikilinks_ned("test"):
    test_titles.add(x["title"])
    # break

print(f"Train titles: {len(train_titles)}, test titles: {len(test_titles)}")
print(f"Zero-shot titles: {len(test_titles - train_titles)}")
