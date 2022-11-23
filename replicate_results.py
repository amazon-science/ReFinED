from refined.evaluation.evaluation import eval_all
from refined.inference.processor import Refined

refined = Refined.from_pretrained(model_name='aida_model',
                                  entity_set='wikipedia',
                                  use_precomputed_descriptions=True)
print('EL results (with model fine-tuned on AIDA)')
eval_all(refined=refined, el=True, filter_nil_spans=False)

refined = Refined.from_pretrained(model_name='wikipedia_model',
                                  entity_set='wikipedia',
                                  use_precomputed_descriptions=True)
print('ED results (with model not fine-tuned on AIDA)')
eval_all(refined=refined, el=False)
