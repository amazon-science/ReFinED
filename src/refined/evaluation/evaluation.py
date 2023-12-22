import os
from pprint import pprint
from typing import Iterable, Optional, Dict

from refined.dataset_reading.entity_linking.dataset_factory import Datasets
from refined.data_types.doc_types import Doc
from refined.doc_preprocessing.preprocessor import Preprocessor
from refined.resource_management.resource_manager import ResourceManager
from refined.evaluation.metrics import Metrics
from refined.resource_management.aws import S3Manager
from tqdm.auto import tqdm
from refined.inference.processor import Refined
from refined.doc_preprocessing.wikidata_mapper import WikidataMapper
from refined.utilities.general_utils import get_logger
from refined.resource_management.loaders import normalize_surface_form
LOG = get_logger(__name__)


def process_annotated_document(
        refined: Refined,
        doc: Doc,
        el: bool = False,
        ed_threshold: float = 0.0,
        force_prediction: bool = False,
        apply_class_check: bool = False,
        filter_nil: bool = False,  # filter_nil is False for our papers results as this is consistent with previous
        # work. But `filter_nil=False` unfairly penalises predicting new (or unlabelled) entities.
        return_special_spans: bool = False, # only set to True if the dataset has special spans (e.g. dates),
        topk_eval: bool = False,
        top_k: int = 0,
) -> Metrics:
    if force_prediction:
        assert ed_threshold == 0.0, "ed_threshold must be set to 0 to force predictions"
    gold_spans = set()
    gold_entity_in_cands = 0
    for idx,span in enumerate(doc.spans):
        if (span.gold_entity is None or span.gold_entity.wikidata_entity_id is None
            # only include entity spans that have been annotated as an entity in a KB
                or span.gold_entity.wikidata_entity_id == "Q0"):
            continue
        gold_spans.add((normalize_surface_form(span.text), span.start, span.gold_entity.wikidata_entity_id))
        if span.gold_entity.wikidata_entity_id in {qcode for qcode, _ in span.candidate_entities}:
            gold_entity_in_cands += 1

    # optionally filter NIL gold spans
    # nil_spans is a set of mention spans that are annotated as mentions in the dataset but are not linked to a KB
    # many nil_spans in public datasets should have been linked to an entity but due to the annotation creation
    # method many entity were missed. Furthermore, when some datasets were built the correct entity
    # did not exist in the KB at the time but do exist now. This means models are unfairly penalized for predicting
    # entities for nil_spans.
    nil_spans = set()
    if doc.md_spans is not None:
        for span in doc.md_spans:
            # gold_entity id will be added to md_spans when md_spans overlaps withs spans in merge_spans() method
            if span.gold_entity is None or span.gold_entity.wikidata_entity_id is None:
                nil_spans.add((span.text, span.start))
    
    predicted_spans = refined.process_text(
        text=doc.text,
        spans=doc.spans if not el else None,
        apply_class_check=apply_class_check,
        prune_ner_types=False,
        return_special_spans=return_special_spans  # only set to True if the dataset has special spans (e.g. dates)
    )

    pred_spans = set()
    pred_spans_topk = set()
    for span in predicted_spans:
        if (
                span.predicted_entity.wikidata_entity_id is None
                or span.entity_linking_model_confidence_score < ed_threshold
                or span.predicted_entity.wikidata_entity_id == 'Q-1'
        ):
            qcode = "Q0"
        else:
            qcode = span.predicted_entity.wikidata_entity_id
        if force_prediction and qcode == "Q0":
            if len(span.top_k_predicted_entities) >= 2:
                qcode = span.top_k_predicted_entities[1][0].wikidata_entity_id
        if len(span.top_k_predicted_entities)>1 and topk_eval:
            max_len = min(top_k,len(span.top_k_predicted_entities))
            for s in span.top_k_predicted_entities[1:max_len]:
                qcode_top_k = s[0].wikidata_entity_id
                pred_spans_topk.add((span.text, span.start, qcode_top_k))
        pred_spans.add((span.text, span.start, qcode))
        if topk_eval:
            pred_spans_topk.add((span.text, span.start, qcode))

    pred_spans = {(normalize_surface_form(text), start, qcode) for text, start, qcode in pred_spans if qcode != "Q0"}
    pred_spans_topk = {(normalize_surface_form(text), start, qcode) for text, start, qcode in pred_spans_topk if qcode != "Q0"}
    if filter_nil:
        # filters model predictions that align with NIL spans in the dataset. See above for more information.
        # Note that this `Doc.md_spans` must include spans with wikidata_entity_id set to None,
        # so the data reader must not filter them out for this argument to work.
        pred_spans = {
            (text, start, qcode)
            for text, start, qcode in pred_spans
            if (text, start) not in nil_spans
        }

    num_gold_spans = len(gold_spans)
    tp = len(pred_spans & gold_spans)
    fp = len(pred_spans - gold_spans)
    fn = len(gold_spans - pred_spans)
    
    if topk_eval:
        tp_k = len(pred_spans_topk & gold_spans)
        fp_k = len(pred_spans_topk - gold_spans)
        fn_k = len(gold_spans - pred_spans_topk)
    else:
        tp_k = 0
        fp_k = 0
        fn_k = 0
    
    # ignore which entity is linked to (consider just the mention detection (NER) prediction)
    pred_spans_md = {(normalize_surface_form(span.text), span.start) for span in predicted_spans}
    gold_spans_md = {(normalize_surface_form(span.text), span.start) for span in doc.md_spans}
    
    
    tp_md = len(pred_spans_md & gold_spans_md)
    fp_md = len(pred_spans_md - gold_spans_md)
    fn_md = len(gold_spans_md - pred_spans_md)

    fp_errors = sorted(list(pred_spans - gold_spans), key=lambda x: x[1])[:5]
    fn_errors = sorted(list(gold_spans - pred_spans), key=lambda x: x[1])[:5]

    fp_errors_md = sorted(list(pred_spans_md - gold_spans_md), key=lambda x: x[1])[:5]
    fn_errors_md = sorted(list(gold_spans_md - pred_spans_md), key=lambda x: x[1])[:5]
    metrics = Metrics(
        el=el,
        num_gold_spans=num_gold_spans,
        tp=tp,
        fp=fp,
        fn=fn,
        tp_md=tp_md,
        fp_md=fp_md,
        fn_md=fn_md,
        tp_k=tp_k,
        fp_k=fp_k,
        fn_k=fn_k,
        gold_entity_in_cand=gold_entity_in_cands,
        num_docs=1,
        example_errors=[{'doc_title': doc.text[:20], 'fp_errors': fp_errors, 'fn_errors': fn_errors}],
        example_errors_md=[{'doc_title': doc.text[:20], 'fp_errors_md': fp_errors_md, 'fn_errors_md': fn_errors_md}]
    )
    return metrics


def evaluate_on_docs(
        refined,
        docs: Iterable[Doc],
        progress_bar: bool = True,
        dataset_name: str = "dataset",
        ed_threshold: float = 0.0,
        apply_class_check: bool = False,
        el: bool = False,
        sample_size: Optional[int] = None,
        filter_nil_spans: bool = False,
        return_special_spans: bool = False,
        topk_eval: bool = False,
        top_k: int = 10,
):
    if topk_eval and top_k < 2: # for top-k evaluation, we need to set k at least 2 . For the case k==1, we will calculate it as the default evaluation. 
        top_k = 2
        
    overall_metrics = Metrics.zeros(el=el)
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
            filter_nil=filter_nil_spans,
            return_special_spans=return_special_spans,
            topk_eval = topk_eval,
            top_k = top_k
        )
        overall_metrics += doc_metrics
        if sample_size is not None and doc_idx > sample_size:
            break
    return overall_metrics


def eval_all(
        refined,
        data_dir: Optional[str] = None,
        datasets_dir: Optional[str] = None,
        additional_data_dir: Optional[str] = None,
        include_spans: bool = True,
        filter_not_in_kb: bool = True,
        ed_threshold: float = 0.15,
        el: bool = False,
        download: bool = True,
        apply_class_check: bool = False,
        filter_nil_spans: bool = False
):
    datasets = get_datasets_obj(preprocessor=refined.preprocessor,
                                data_dir=data_dir,
                                datasets_dir=datasets_dir,
                                additional_data_dir=additional_data_dir,
                                download=download)
    dataset_name_to_docs = get_standard_datasets(datasets, el, filter_not_in_kb, include_spans)
    return evaluate_on_datasets(refined=refined,
                                dataset_name_to_docs=dataset_name_to_docs,
                                el=el,
                                apply_class_check=apply_class_check,
                                ed_threshold=ed_threshold,
                                filter_nil_spans=filter_nil_spans
                                )


def get_standard_datasets(datasets: Datasets,
                          el: bool,
                          filter_not_in_kb: bool = True,
                          include_spans: bool = True) -> Dict[str, Iterable[Doc]]:
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
    return dataset_name_to_docs


def evaluate_on_datasets(refined: Refined,
                         dataset_name_to_docs: Dict[str, Iterable[Doc]],
                         el: bool,
                         apply_class_check: bool = False,
                         ed_threshold: float = 0.15,
                         return_special_spans: bool = False,  # only set to True if the dataset has special spans (
                         # e.g. dates)
                         filter_nil_spans: bool = False
                         ):
    dataset_name_to_metrics = dict()
    for dataset_name, dataset_docs in dataset_name_to_docs.items():
        metrics = evaluate_on_docs(
            refined=refined,
            docs=dataset_docs,
            dataset_name=dataset_name,
            ed_threshold=ed_threshold,
            el=el,
            apply_class_check=apply_class_check,
            filter_nil_spans=filter_nil_spans,  # filter model predictions that align with md_spans that have no
            # gold_entity_id but are annotated/labelled as mentions in the dataset.
            return_special_spans=return_special_spans,
        )
        dataset_name_to_metrics[dataset_name] = metrics
        print("*****************************\n\n")
        print(f"Dataset name: {dataset_name}")
        print(metrics.get_summary())
        print("*****************************\n\n")
    return dataset_name_to_metrics


def get_datasets_obj(preprocessor: Preprocessor,
                     download: bool = True,
                     data_dir: Optional[str] = None,
                     datasets_dir: Optional[str] = None,
                     additional_data_dir: Optional[str] = None,
                     ) -> Datasets:
    if data_dir is None:
        data_dir = os.path.join(os.path.expanduser('~'), '.cache', 'refined')
    if datasets_dir is None:
        datasets_dir = os.path.join(data_dir, 'datasets')
    if additional_data_dir is None:
        additional_data_dir = os.path.join(data_dir, 'additional_data')

    resource_manager = ResourceManager(S3Manager(),
                                       data_dir=datasets_dir,
                                       datasets_dir=datasets_dir,
                                       additional_data_dir=additional_data_dir,
                                       entity_set=None,
                                       model_name=None
                                       )
    if download:
        resource_manager.download_datasets_if_needed()
        resource_manager.download_additional_files_if_needed()

    wikidata_mapper = WikidataMapper(resource_manager=resource_manager)
    return Datasets(preprocessor=preprocessor,
                    resource_manager=resource_manager,
                    wikidata_mapper=wikidata_mapper)


def evaluate(evaluation_dataset_name_to_docs: Dict[str, Iterable[Doc]],
             refined: Refined,
             ed_threshold: float = 0.15,
             el: bool = True,
             ed: bool = True,
             print_errors: bool = True,
             return_special_spans: bool = True) -> Dict[str, Metrics]:
    dataset_name_to_metrics = dict()
    if el:
        LOG.info("Running entity linking evaluation")
        el_results = evaluate_on_datasets(
            refined=refined,
            dataset_name_to_docs=evaluation_dataset_name_to_docs,
            el=True,
            ed_threshold=ed_threshold,
            return_special_spans=return_special_spans,
            filter_nil_spans=True  # makes EL evaluation more fair
        )
        for dataset_name, metrics in el_results.items():
            dataset_name_to_metrics[f"{dataset_name}-EL"] = metrics
            if print_errors:
                LOG.info("Printing EL errors")
                pprint(metrics.example_errors[:5])
                LOG.info("Printing MD errors")
                pprint(metrics.example_errors_md[:5])

    if ed:
        LOG.info("Running entity disambiguation evaluation")
        ed_results = evaluate_on_datasets(
            refined=refined,
            dataset_name_to_docs=evaluation_dataset_name_to_docs,
            el=False,
            ed_threshold=ed_threshold,
            return_special_spans=False
        )
        for dataset_name, metrics in ed_results.items():
            dataset_name_to_metrics[f"{dataset_name}-ED"] = metrics
            if print_errors:
                LOG.info("Printing ED errors")
                pprint(metrics.example_errors[:5])

    return dataset_name_to_metrics