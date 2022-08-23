from typing import Iterable, Optional

from dataset_reading.dataset_factory import Datasets
from doc_preprocessing.dataclasses import Doc
from doc_preprocessing.resource_manager import ResourceManager
from evaluation.metrics import Metrics
from utilities.aws import S3Manager
from tqdm.auto import tqdm
from refined.processor import Refined


def process_annotated_document(
    refined: Refined,
    doc,
    el: bool = False,
    ed_threshold: float = 0.0,
    force_prediction: bool = False,
    apply_class_check: bool = False,
    filter_nil: bool = False,
):
    if force_prediction:
        assert ed_threshold == 0.0, "ed_threshold must be set to 0 to force predictions"
    gold_spans = set()
    gold_entity_in_cands = 0
    for span in doc.spans:
        if filter_nil and (span.gold_entity_id is None or span.gold_entity_id == "Q0"):
            continue
        gold_spans.add((span.text, span.start, span.gold_entity_id))
        if span.gold_entity_id in {qcode for qcode, _ in span.candidate_entity_ids}:
            gold_entity_in_cands += 1

    # optionally filter NIL gold spans
    nil_spans = set()
    if doc.spans is not None:
        for span in doc.spans:
            if span.gold_entity_id is None:
                nil_spans.add((span.text, span.start))

    predicted_spans = refined.process_text(
        text=doc.text,
        spans=doc.spans if not el else None,
        apply_class_check=apply_class_check,
        prune_ner_types=False,
        return_special_spans=False
    )
    pred_spans = set()
    for span in predicted_spans:
        if (
            span.pred_entity_id is None
            or "wikidata_qcode" not in span.pred_entity_id[0]
            or span.pred_entity_id[1] < ed_threshold
        ):
            qcode = "Q0"
        else:
            qcode = span.pred_entity_id[0]["wikidata_qcode"]
        if force_prediction and qcode == "Q0":
            if len(span.pred_ranked_entity_ids) >= 2:
                qcode = span.pred_ranked_entity_ids[1][0].get("wikidata_qcode", "Q0")
        pred_spans.add((span.text, span.start, qcode))

    pred_spans = {(text, start, qcode) for text, start, qcode in pred_spans if qcode != "Q0"}
    if filter_nil:
        pred_spans = {
            (text, start, qcode)
            for text, start, qcode in pred_spans
            if (text, start) not in nil_spans
        }

    num_gold_spans = len(gold_spans)
    tp = len(pred_spans & gold_spans)
    fp = len(pred_spans - gold_spans)
    fn = len(gold_spans - pred_spans)

    metrics = Metrics(
        num_gold_spans=num_gold_spans,
        tp=tp,
        fp=fp,
        fn=fn,
        gold_entity_in_cand=gold_entity_in_cands,
        num_docs=1,
    )
    return metrics


def evaluate_on_docs(
    refined,
    docs: Iterable[Doc],
    progress_bar: bool = True,
    dataset_name: str = "dataset",
    ed_threshold: float = 0.0,
    apply_class_check: bool = False,
    use_final_ed_score: bool = False,
    el: bool = False,
    sample_size: Optional[int] = None,
    filter_nil: bool = False,
):
    overall_metrics = Metrics.zeros()
    for doc_idx, doc in tqdm(
        enumerate(list(docs)), disable=not progress_bar, desc=f"Evaluating on {dataset_name}"
    ):
        doc_metrics = process_annotated_document(
            refined=refined,
            doc=doc,
            force_prediction=False,
            ed_threshold=ed_threshold,
            apply_class_check=apply_class_check,
            el=el,
            filter_nil=filter_nil,
        )
        overall_metrics += doc_metrics
        if sample_size is not None and doc_idx > sample_size:
            break
    return overall_metrics


def eval_all(
    refined,
    datasets_dir: str,
    include_spans: bool = True,
    filter_not_in_kb: bool = True,
    ed_threshold: float = 0.0,
    print_results: bool = True,
    el: bool = False,
    download: bool = True,
    apply_class_check: bool = False,
):
    if download:
        resource_manager = ResourceManager(S3Manager(), data_dir=datasets_dir)
        resource_manager.download_datasets()

    datasets = Datasets(preprocessor=refined.preprocessor, datasets_path=datasets_dir)
    if not el:
        dataset_name_to_docs = {
            "AIDA": datasets.get_aida_docs(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
            "MSNBC": datasets.get_msnbc_docs(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
            "AQUAINT": datasets.get_aquaint_docs(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
            "ACE2004": datasets.get_ace2004_docs(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
            "CWEB": datasets.get_cweb_docs(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
            "WIKI": datasets.get_wiki_docs(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
        }
    else:
        dataset_name_to_docs = {
            "AIDA": datasets.get_aida_docs(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
            "MSNBC": datasets.get_msnbc_docs(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
        }

    dataset_name_to_metrics = dict()
    for dataset_name, dataset_docs in dataset_name_to_docs.items():
        metrics = evaluate_on_docs(
            refined=refined,
            docs=dataset_docs,
            dataset_name=dataset_name,
            ed_threshold=ed_threshold,
            el=el,
            apply_class_check=apply_class_check,
        )
        dataset_name_to_metrics[dataset_name] = metrics
        if print_results:
            print("*****************************\n\n")
            print(f"Dataset name: {dataset_name}")
            print(metrics.get_summary())
            print("*****************************\n\n")

    return dataset_name_to_metrics
