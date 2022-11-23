from refined.evaluation.evaluation import get_datasets_obj
from refined.inference.processor import Refined
from refined.training.fine_tune.fine_tune import fine_tune_on_docs

refined = Refined.from_pretrained(model_name='wikipedia_model',
                                  entity_set='wikipedia',
                                  use_precomputed_descriptions=False)

dataset = get_datasets_obj(preprocessor=refined.preprocessor)
fine_tune_on_docs(refined=refined,
                  train_docs=dataset.get_aida_docs("train", include_gold_label=True),
                  eval_docs=dataset.get_aida_docs("dev", include_gold_label=True))
# To fine-tune on EL on the WebQSP (questions) dataset use `dataset.get_webqsp_docs("train", include_gold_label=True)`.
