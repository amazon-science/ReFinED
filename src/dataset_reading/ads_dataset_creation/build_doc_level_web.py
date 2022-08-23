import json
import gzip

from typing import Optional, Dict, List
from tqdm import tqdm

from dataset_reading.ads_dataset_creation.utils import entity_correct, write_dataset_to_disk, read_webdata_annotations, \
    group_results_by_sentence, add_nonambiguous_annotations, articles_with_a_false_mention


class ADSWebArticles:

    def __init__(self, fpath: str, max_articles: Optional[int] = None):
        """
        Loads articles from ADS WEB dataset at:
         s3://fount.resources.dev/multi_instance/datasets/RE/2021-08-31-FOUNT-DS-WEB/
        """
        self.fpath = fpath

        sentence_to_article_hash, article_hash_to_article = self.read_ads_web_dataset(self.fpath,
                                                                                      max_articles=max_articles)

        self.sentence_to_article_hash: Dict[str, int] = sentence_to_article_hash
        self.article_hash_to_article: Dict[int: Dict] = article_hash_to_article

    def map_sentence_to_article(self, sentence: str) -> Optional[Dict]:

        if sentence not in self.sentence_to_article_hash:
            return None

        article_hash = self.sentence_to_article_hash[sentence]
        article = self.article_hash_to_article[article_hash]

        return article

    @staticmethod
    def read_ads_web_dataset(fpath: str, max_articles: Optional[int] = None):

        total = max_articles if max_articles is not None else 885228

        sentence_to_article_hash = {}
        article_hash_to_article = {}

        with gzip.open(fpath, "rb") as f:
            for ix, line in enumerate(tqdm(f, total=total)):
                article = json.loads(line)
                text_hash = hash(article["text"])
                article_hash_to_article[text_hash] = article

                for sentence in article["sentences"]:
                    sentence_text = article["text"][sentence["start_ix"]:sentence["end_ix"]]
                    sentence_to_article_hash[sentence_text] = text_hash

                if max_articles is not None and (ix + 1) == max_articles:
                    break

        return sentence_to_article_hash, article_hash_to_article


def add_annotations_to_web_articles(web_sentences: List, dset_fpath: str, max_articles: Optional[int] = None):
    ads_web = ADSWebArticles(fpath=dset_fpath, max_articles=max_articles)

    articles_with_sentences = {}

    for sentence, annotations in web_sentences.items():

        article = ads_web.map_sentence_to_article(sentence.sentence)

        if article is None:
            continue

        article_hash = hash(article["text"])

        if article_hash not in articles_with_sentences:
            articles_with_sentences[article_hash] = article

        sentence_to_start_ix = {article["text"][s["start_ix"]:s["end_ix"]]: s["start_ix"] for s in article["sentences"]}
        mention_ixs_to_mention = {(m["start_ix"], m["end_ix"]): ix for ix, m in enumerate(article["mentions"])}

        # Character start index of the sentence in the article
        sentence_start_ix = sentence_to_start_ix[sentence.sentence]

        for annotation in annotations:
            start_ix = sentence_start_ix + annotation.start
            end_ix = sentence_start_ix + annotation.end

            mention_key = (start_ix, end_ix)

            if mention_key in mention_ixs_to_mention:
                mention_ix = mention_ixs_to_mention[mention_key]
                mention = article["mentions"][mention_ix]

                if mention["annotation"] is None:
                    mention["annotation"] = {"span_annotation": [], "entity_correct": []}

                mention["annotation"]["span_annotation"].append(annotation.entity_span_error)
                mention["annotation"]["entity_correct"].append(entity_correct(annotation.entity_resolution_error))

    return list(articles_with_sentences.values())


if __name__ == "__main__":

    # Read annotations that are most likely from web data
    web_annotations = read_webdata_annotations("/home/fshjos/ads_annotations", max_annotations=None)
    web_sentences = group_results_by_sentence(web_annotations)

    # Read ads web dataset into memory and match annotations to articles
    web_dset = add_annotations_to_web_articles(web_sentences, "/home/fshjos/ads_test_datasets/ads_test.json.gz",
                                               max_articles=None)

    web_dset = add_nonambiguous_annotations(web_dset)
    web_dset_neg = articles_with_a_false_mention(web_dset)

    write_dataset_to_disk(web_dset,
                          fpath="/home/fshjos/python_modules/refined/ReFinED/data/datasets/ads/doc_level_web.json")

    write_dataset_to_disk(web_dset_neg,
                          fpath="/home/fshjos/python_modules/refined/ReFinED/data/datasets/ads/doc_level_web_neg.json")