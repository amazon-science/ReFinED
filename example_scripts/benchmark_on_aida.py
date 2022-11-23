import time

import torch.cuda

from refined.evaluation.evaluation import get_datasets_obj
from refined.inference.processor import Refined

refined = Refined.from_pretrained(model_name='aida_model',
                                  entity_set='wikipedia',
                                  use_precomputed_descriptions=True)
dataset = get_datasets_obj(preprocessor=refined.preprocessor)
eval_docs = list(dataset.get_aida_docs("dev"))
texts = [doc.text for doc in eval_docs]
if torch.cuda.is_available():
    torch.cuda.synchronize()
start_time = time.time()
refined.process_text_batch(texts=texts, prune_ner_types=False,
                           return_special_spans=False, max_batch_size=8)
end_time = time.time()
print(f"Time to EL process AIDA dev is {end_time - start_time:.2f}s")
