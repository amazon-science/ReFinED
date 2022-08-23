import os

from evaluation.evaluation import eval_all
from refined.processor import Refined

refined = Refined.from_pretrained(model_name='wikipedia_model',
                                  entity_set='wikipedia',
                                  data_dir=os.path.join(os.path.expanduser('~'), '.cache/refined/'),
                                  download_files=True,
                                  debug=False,
                                  use_precomputed_descriptions=True,
                                  requires_redirects_and_disambig=True)

eval_all(refined=refined,
         datasets_dir=os.path.join(os.path.expanduser('~'), '.cache/refined/', 'datasets'),
         filter_not_in_kb=True,
         download=True,
         el=False)
