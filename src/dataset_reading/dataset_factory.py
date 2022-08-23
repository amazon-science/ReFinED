# general ER reading module or class
# yields Doc object from any dataset
# additional arguments determine, max seq length, max candidates, max mentions, same doc only option
# filter disambiguation pages,
import json
import os
from typing import Iterable, Optional

from doc_preprocessing.dataclasses import Doc, Span
from doc_preprocessing.doc_preprocessor import DocumentPreprocessorMemoryBased


class Datasets:
    def __init__(self, preprocessor: DocumentPreprocessorMemoryBased, datasets_path: str):
        self.preprocessor = preprocessor
        self.datasets_path = datasets_path

    def get_aida_docs(
        self,
        split: str,
        include_spans: bool = True,
        include_gold_label: bool = True,
        filter_not_in_kb: bool = True,
        include_mentions_for_nil: bool = True,
    ) -> Iterable[Doc]:
        split_to_filename = {
            "train": "aida_training_gerbil_fixed.json",
            "dev": "aida_dev_gerbil_fixed.json",
            "test": "aida_test.json",
        }
        assert split in split_to_filename, "split must be in {train, dev, test}"
        filename = os.path.join(self.datasets_path, split_to_filename[split])
        with open(filename, "r") as f:
            for line_idx, line in enumerate(f):
                line = json.loads(line)
                text = line["text"]
                spans = None
                md_spans = None
                if include_spans:
                    spans = []
                    md_spans = []
                    for span in line["spans"]:
                        if include_mentions_for_nil:
                            md_spans.append(
                                Span(
                                    start=span["start"],
                                    ln=span["length"],
                                    text=text[span["start"] : span["start"] + span["length"]],
                                    coarse_type="MENTION"
                                )
                            )

                        titles = [
                            uri.replace("http://en.wikipedia.org/wiki/", "")
                            for uri in span["uris"]
                            if "http://en.wikipedia.org/wiki/" in uri
                        ]

                        if len(titles) == 0:
                            continue

                        title = titles[0]
                        qcode = self.preprocessor.map_title_to_qcode(title)

                        if filter_not_in_kb and (
                            qcode is None or qcode in self.preprocessor.disambiguation_qcodes
                        ):
                            continue

                        if not filter_not_in_kb and qcode is None:
                            qcode = "Q0"

                        if not include_mentions_for_nil:
                            md_spans.append(
                                Span(
                                    start=span["start"],
                                    ln=span["length"],
                                    text=text[span["start"] : span["start"] + span["length"]],
                                    coarse_type="MENTION"
                                )
                            )

                        if include_gold_label:
                            spans.append(
                                Span(
                                    start=span["start"],
                                    ln=span["length"],
                                    text=text[span["start"] : span["start"] + span["length"]],
                                    gold_entity_id=qcode,
                                    coarse_type="MENTION"
                                )
                            )
                        else:
                            spans.append(
                                Span(
                                    start=span["start"],
                                    ln=span["length"],
                                    text=text[span["start"] : span["start"] + span["length"]],
                                    coarse_type="MENTION"
                                )
                            )

                if spans is None:
                    yield Doc.from_text(
                        text=text,
                        data_dir=self.preprocessor.data_dir,
                        transformer_name=self.preprocessor.transformer_name,
                    )
                else:
                    yield Doc.from_text_with_spans(
                        text=text, spans=spans, md_spans=md_spans, preprocessor=self.preprocessor
                    )

    def _read_standard_format(
        self,
        filename: str,
        include_spans: bool = True,
        include_gold_label: bool = True,
        filter_not_in_kb: bool = True,
    ) -> Iterable[Doc]:
        with open(filename, "r") as f:
            for line_idx, line in enumerate(f):
                line = json.loads(line)
                text = line["text"]
                spans = None
                md_spans = None
                if include_spans:
                    spans = []
                    md_spans = []
                    for span in line["mentions"]:
                        title = span["wiki_name"]
                        md_spans.append(
                            Span(
                                start=span["start"],
                                ln=span["length"],
                                text=text[span["start"] : span["start"] + span["length"]],
                                coarse_type="MENTION"
                            )
                        )

                        if title is None or title == "NIL":
                            continue

                        title = title.replace(" ", "_")
                        qcode = self.preprocessor.map_title_to_qcode(title)

                        if filter_not_in_kb and (
                            qcode is None or qcode in self.preprocessor.disambiguation_qcodes
                        ):
                            continue

                        if not filter_not_in_kb and qcode is None:
                            qcode = "Q0"

                        if include_gold_label:
                            spans.append(
                                Span(
                                    start=span["start"],
                                    ln=span["length"],
                                    text=text[span["start"] : span["start"] + span["length"]],
                                    gold_entity_id=qcode,
                                    coarse_type="MENTION"
                                )
                            )
                        else:
                            spans.append(
                                Span(
                                    start=span["start"],
                                    ln=span["length"],
                                    text=text[span["start"] : span["start"] + span["length"]],
                                    coarse_type="MENTION"
                                )
                            )
                if spans is None:
                    yield Doc.from_text(
                        text=text,
                        data_dir=self.preprocessor.data_dir,
                        transformer_name=self.preprocessor.transformer_name,
                    )
                else:
                    yield Doc.from_text_with_spans(
                        text=text, spans=spans, md_spans=md_spans, preprocessor=self.preprocessor
                    )

    def get_msnbc_docs(
        self,
        split: str,
        include_spans: bool = True,
        include_gold_label: bool = True,
        filter_not_in_kb: bool = True,
    ) -> Iterable[Doc]:
        assert split == "test", "MSNBC only has a test dataset"
        return self._read_standard_format(
            filename=os.path.join(self.datasets_path, "msnbc_parsed.json"),
            include_spans=include_spans,
            include_gold_label=include_gold_label,
            filter_not_in_kb=filter_not_in_kb,
        )

    def get_aquaint_docs(
        self,
        split: str,
        include_spans: bool = True,
        include_gold_label: bool = True,
        filter_not_in_kb: bool = True,
    ) -> Iterable[Doc]:
        assert split == "test", "aquaint only has a test dataset"
        return self._read_standard_format(
            filename=os.path.join(self.datasets_path, "aquaint_parsed.json"),
            include_spans=include_spans,
            include_gold_label=include_gold_label,
            filter_not_in_kb=filter_not_in_kb,
        )

    def get_ace2004_docs(
        self,
        split: str,
        include_spans: bool = True,
        include_gold_label: bool = True,
        filter_not_in_kb: bool = True,
    ) -> Iterable[Doc]:
        assert split == "test", "ace2004 only has a test dataset"
        return self._read_standard_format(
            filename=os.path.join(self.datasets_path, "ace2004_parsed.json"),
            include_spans=include_spans,
            include_gold_label=include_gold_label,
            filter_not_in_kb=filter_not_in_kb,
        )

    def get_cweb_docs(
        self,
        split: str,
        include_spans: bool = True,
        include_gold_label: bool = False,
        filter_not_in_kb: bool = True,
    ) -> Iterable[Doc]:
        assert split == "test", "cweb only has a test dataset"
        return self._read_standard_format(
            filename=os.path.join(self.datasets_path, "clueweb_parsed.json"),
            include_spans=include_spans,
            include_gold_label=include_gold_label,
            filter_not_in_kb=filter_not_in_kb,
        )

    def get_wiki_docs(
        self,
        split: str,
        include_spans: bool = True,
        include_gold_label: bool = False,
        filter_not_in_kb: bool = True,
    ) -> Iterable[Doc]:
        assert split == "test", "wiki only has a test dataset"
        return self._read_standard_format(
            filename=os.path.join(self.datasets_path, "wikipedia_parsed.json"),
            include_spans=include_spans,
            include_gold_label=include_gold_label,
            filter_not_in_kb=filter_not_in_kb,
        )

    def _get_elq_format_docs(
        self,
        filename: str,
        include_spans: bool = True,
        include_gold_label: bool = False,
        filter_not_in_kb: bool = True,
    ):
        with open(filename, "r") as f:
            for line in f:
                line = json.loads(line)
                text = line["text"]
                assert len(line["mentions"]) == len(line["wikidata_id"])
                md_spans = None
                spans = None
                if include_spans:
                    md_spans = []
                    spans = []
                    for mention, qcode in zip(line["mentions"], line["wikidata_id"]):
                        if filter_not_in_kb and (
                            qcode is None or qcode in self.preprocessor.disambiguation_qcodes
                        ):
                            continue
                        if not filter_not_in_kb and qcode is None:
                            qcode = "Q0"
                        start, end = mention
                        md_spans.append(Span(start=start, ln=end - start, text=text[start:end], coarse_type="MENTION"))
                        if include_gold_label:
                            spans.append(
                                Span(
                                    start=start,
                                    ln=end - start,
                                    text=text[start:end],
                                    gold_entity_id=qcode,
                                    coarse_type="MENTION"
                                )
                            )
                        else:
                            spans.append(Span(start=start, ln=end - start, text=text[start:end], coarse_type="MENTION"))

                if spans is None:
                    yield Doc.from_text(
                        text=text,
                        data_dir=self.preprocessor.data_dir,
                        transformer_name=self.preprocessor.transformer_name,
                    )
                else:
                    yield Doc.from_text_with_spans(
                        text=text, spans=spans, md_spans=md_spans, preprocessor=self.preprocessor
                    )

    def get_web_qsp_docs(
        self,
        split: str,
        include_spans: bool = True,
        include_gold_label: bool = False,
        filter_not_in_kb: bool = True,
    ):
        split_to_filename = {
            "train": "train.jsonl",
            "dev": "dev.jsonl",
            "test": "test.jsonl",
        }
        assert split in {"train", "dev", "test"}, "split must be in {train, dev, test}"
        filename = os.path.join(self.datasets_path, "WebQSP_EL", split_to_filename[split])
        return self._get_elq_format_docs(
            filename=filename,
            include_spans=include_spans,
            include_gold_label=include_gold_label,
            filter_not_in_kb=filter_not_in_kb,
        )

    def get_graph_questions_docs(
        self,
        split: str,
        include_spans: bool = True,
        include_gold_label: bool = False,
        filter_not_in_kb: bool = True,
    ):
        split_to_filename = {
            "train": "train.jsonl",
            "dev": "dev.jsonl",
            "test": "test.jsonl",
        }
        assert split in {"train", "dev", "test"}, "split must be in {train, dev, test}"
        filename = os.path.join(self.datasets_path, "graphquestions_EL", split_to_filename[split])
        return self._get_elq_format_docs(
            filename=filename,
            include_spans=include_spans,
            include_gold_label=include_gold_label,
            filter_not_in_kb=filter_not_in_kb,
        )

    def get_aida_ppr_docs(
        self,
        split: str,
        include_spans: bool = True,
        include_gold_label: bool = False,
        filter_not_in_kb: bool = True,
    ) -> Iterable[Doc]:
        split_to_filename = {
            "train": "aida_train_ppr.json",
            "dev": "aida_dev_ppr.json",
            "test": "aida_test_ppr.json",
        }
        assert split in split_to_filename, "split must be in {train, dev, test}"
        filename = os.path.join(self.datasets_path, split_to_filename[split])
        with open(filename, "r") as f:
            for line_idx, line in enumerate(f):
                line = json.loads(line)
                text = line["doc_text"]
                spans = None
                md_spans = None
                if include_spans:
                    spans = []
                    md_spans = []
                    for span in line["spans"]:
                        start = span["start"]
                        ln = span["end"] - span["start"]
                        md_spans.append(Span(start=start, ln=ln, text=span["text"], coarse_type="MENTION"))
                        gold_qcode = self.preprocessor.map_title_to_qcode(span["gold_titles"][0])
                        candidates_qcodes = {
                            self.preprocessor.map_title_to_qcode(c) for c in span["candidates"]
                        }
                        candidates_qcodes -= {None}
                        if filter_not_in_kb and (
                            gold_qcode is None
                            or gold_qcode in self.preprocessor.disambiguation_qcodes
                        ):
                            continue

                        if not filter_not_in_kb and gold_qcode is None:
                            gold_qcode = "Q0"

                        if include_gold_label:
                            spans.append(
                                Span(
                                    start=start,
                                    ln=ln,
                                    text=span["text"],
                                    pruned_candidates=candidates_qcodes,
                                    gold_entity_id=gold_qcode,
                                    coarse_type="MENTION"
                                )
                            )
                        else:
                            spans.append(
                                Span(
                                    start=start,
                                    ln=ln,
                                    text=span["text"],
                                    pruned_candidates=candidates_qcodes,
                                    coarse_type="MENTION"
                                )
                            )
                if spans is None:
                    yield Doc.from_text(
                        text=text,
                        data_dir=self.preprocessor.data_dir,
                        transformer_name=self.preprocessor.transformer_name,
                    )
                else:
                    yield Doc.from_text_with_spans(
                        text=text, spans=spans, md_spans=md_spans, preprocessor=self.preprocessor
                    )

    def get_wikilinks_ned_docs(
        self,
        split: str,
        include_spans: bool = True,
        include_gold_label: bool = False,
        filter_not_in_kb: bool = True,
        sample_k_candidates: Optional[int] = None,
    ) -> Iterable[Doc]:

        assert split in {"train", "dev", "test"}
        split_to_filenames = {
            "train": [
                os.path.join(self.datasets_path, "wikilinks_ned", "train", f"train_{i}.json")
                for i in range(6)
            ],
            "dev": [os.path.join(self.datasets_path, "wikilinks_ned", "dev.json")],
            "test": [os.path.join(self.datasets_path, "wikilinks_ned", "test.json")],
        }
        filenames = split_to_filenames[split]
        for filename in filenames:
            with open(filename, "r") as f:
                for line in f:
                    line = json.loads(line)
                    left_context = " ".join(line["left_context"])
                    right_context = " ".join(line["right_context"])
                    mention = " ".join(line["mention_as_list"])
                    text = left_context + " " + mention + " " + right_context
                    entity_start = len(left_context) + 1
                    entity_length = len(mention)
                    wikipedia_title = line["y_title"]
                    mention_text = text[entity_start : entity_start + entity_length]

                    title = wikipedia_title.replace(" ", "_")
                    qcode = self.preprocessor.map_title_to_qcode(title)

                    if filter_not_in_kb and (
                        qcode is None or qcode in self.preprocessor.disambiguation_qcodes
                    ):
                        continue

                    if not filter_not_in_kb and qcode is None:
                        qcode = "Q0"

                    if include_gold_label:
                        spans = [
                            Span(
                                start=entity_start,
                                ln=entity_length,
                                text=mention_text,
                                gold_entity_id=qcode,
                                coarse_type="MENTION"
                            )
                        ]
                    else:
                        spans = [Span(start=entity_start, ln=entity_length, text=mention_text, coarse_type="MENTION")]

                    if spans is None:
                        yield Doc.from_text(
                            text=text,
                            data_dir=self.preprocessor.data_dir,
                            transformer_name=self.preprocessor.transformer_name,
                        )
                    else:
                        yield Doc.from_text_with_spans(
                            text=text,
                            spans=spans,
                            preprocessor=self.preprocessor,
                            sample_k_candidates=sample_k_candidates,
                        )

    def get_shadowlink_docs(self,
                             split: str,
                             include_spans: bool = True,
                             include_mentions_for_nil: bool = True,
                             include_gold_label: bool = True,
                             filter_not_in_kb: bool = False):

        split_to_filename = {
            "tail": "shadowlink_tail_with_spacy_mentions.json",
            "top": "shadowlink_top_with_spacy_mentions.json",
            "shadow": "shadowlink_shadow_with_spacy_mentions.json"
        }

        assert split in split_to_filename, f"split must be in {split_to_filename.keys()}"

        filename = os.path.join(self.datasets_path, split_to_filename[split])

        lines = json.load(open(filename, "r"))

        for line_idx, line in enumerate(lines):
            text = line["example"]
            spans = None
            md_spans = None
            if include_spans:
                spans = []
                md_spans = []
                for span in line["spans"]:
                    if include_mentions_for_nil:
                        md_spans.append(
                            Span(
                                start=span["start"],
                                ln=span["length"],
                                text=text[span["start"]: span["start"] + span["length"]],
                                coarse_type="MENTION"
                            )
                        )

                    title = span["wikipedia_title"] if "wikipedia_title" in span else None

                    qcode = self.preprocessor.map_title_to_qcode(title) if title is not None else None

                    if filter_not_in_kb and (
                            qcode is None or qcode in self.preprocessor.disambiguation_qcodes
                    ):
                        continue

                    if not include_mentions_for_nil:
                        md_spans.append(
                            Span(
                                start=span["start"],
                                ln=span["length"],
                                text=text[span["start"]: span["start"] + span["length"]],
                                coarse_type="MENTION"
                            )
                        )

                    if include_gold_label:
                        spans.append(
                            Span(
                                start=span["start"],
                                ln=span["length"],
                                text=text[span["start"]: span["start"] + span["length"]],
                                gold_entity_id=qcode,
                                coarse_type="MENTION"
                            )
                        )
                    else:
                        spans.append(
                            Span(
                                start=span["start"],
                                ln=span["length"],
                                text=text[span["start"]: span["start"] + span["length"]],
                                coarse_type="MENTION"
                            )
                        )

            if spans is None:
                yield Doc.from_text(
                    text=text,
                    data_dir=self.preprocessor.data_dir,
                    transformer_name=self.preprocessor.transformer_name,
                )
            else:
                yield Doc.from_text_with_spans(
                    text=text, spans=spans, md_spans=md_spans, preprocessor=self.preprocessor
                )

    def get_wikidata_dataset(
        self,
        split: str,
        include_spans: bool = True,
        include_gold_label: bool = False,
        filter_not_in_kb: bool = True,
    ):
        assert split in {"train", "test"}
        split_to_filename = {
            "train": os.path.join(self.datasets_path, "opentapioca_datasets", "istex_train.ttl"),
            "test": os.path.join(self.datasets_path, "opentapioca_datasets", "istex_test.ttl"),
        }
        filename = split_to_filename[split]
        from pynif import NIFCollection

        nif = NIFCollection.load(filename)
        for context in nif.contexts:
            text = context.mention
            spans = None
            if include_spans:
                spans = []
                for phrase in context.phrases:
                    start = phrase.beginIndex
                    end = phrase.endIndex
                    ln = end - start
                    mention_text = text[start:end]
                    gold_qcode = phrase.taIdentRef
                    if "http://www.wikidata.org/entity/" not in gold_qcode:
                        continue

                    if filter_not_in_kb and (
                        gold_qcode is None or gold_qcode in self.preprocessor.disambiguation_qcodes
                    ):
                        continue

                    if not filter_not_in_kb and gold_qcode is None:
                        gold_qcode = "Q0"

                    gold_qcode = gold_qcode.replace("http://www.wikidata.org/entity/", "")

                    if include_gold_label:
                        spans.append(
                            Span(start=start, ln=ln, text=mention_text, gold_entity_id=gold_qcode, coarse_type="MENTION")
                        )
                    else:
                        spans.append(Span(start=start, ln=ln, text=mention_text, coarse_type="MENTION"))
            if spans is None:
                yield Doc.from_text(
                    text=text,
                    data_dir=self.preprocessor.data_dir,
                    transformer_name=self.preprocessor.transformer_name,
                )
            else:
                yield Doc.from_text_with_spans(
                    text=text, spans=spans, preprocessor=self.preprocessor
                )

    def get_ads_docs(self,
                     split: str,
                     include_spans: bool = True,
                     include_mentions_for_nil: bool = True,
                     include_gold_label: bool = True,
                     filter_not_in_kb: bool = False):

        split_to_filename = {
            "web": "ads_doc_level_web.json",
            "web_negatives": "ads_doc_level_web_neg.json",
            "web_negatives_annotated": "ads_doc_level_annotated.json"
        }

        assert split in split_to_filename, f"split must be in {split_to_filename.keys()}"

        filename = os.path.join(self.datasets_path, split_to_filename[split])

        annotated_dset = split in {"web_negatives_annotated"}

        for line_idx, line in enumerate(open(filename, "r")):

            line = json.loads(line)

            text = line["text"]
            spans = None
            md_spans = None
            if include_spans:
                spans = []
                md_spans = []
                for span in line["mentions"]:

                    if annotated_dset:
                        # Only add qids for spans that were re-annotated with the correct entity
                        span["qid"] = span["qid"] if span["difficult_entity"] else None

                    if include_mentions_for_nil:
                        md_spans.append(
                            Span(
                                start=span["start_ix"],
                                ln=span["end_ix"] - span["start_ix"],
                                text=span["text"],
                                coarse_type="MENTION",
                                entity_annotation=span["er_annotation"]
                            )
                        )

                    if filter_not_in_kb and (
                            span["qid"] is None or span["qid"] in self.preprocessor.disambiguation_qcodes
                    ):
                        continue

                    if not include_mentions_for_nil:
                        md_spans.append(
                            Span(
                                start=span["start_ix"],
                                ln=span["end_ix"] - span["start_ix"],
                                text=span["text"],
                                coarse_type="MENTION",
                                entity_annotation=span["er_annotation"]
                            )
                        )

                    if include_gold_label:
                        spans.append(
                            Span(
                                start=span["start_ix"],
                                ln=span["end_ix"] - span["start_ix"],
                                text=span["text"],
                                gold_entity_id=span["qid"],
                                coarse_type="MENTION",
                                entity_annotation=span["er_annotation"]
                            )
                        )
                    else:
                        spans.append(

                            Span(
                                start=span["start_ix"],
                                ln=span["end_ix"] - span["start_ix"],
                                text=span["text"],
                                coarse_type="MENTION",
                                entity_annotation=span["er_annotation"]
                            )
                        )

            if spans is None:
                yield Doc.from_text(
                    text=text,
                    data_dir=self.preprocessor.data_dir,
                    transformer_name=self.preprocessor.transformer_name,
                )
            else:
                yield Doc.from_text_with_spans(
                    text=text, spans=spans, md_spans=md_spans, preprocessor=self.preprocessor
                )