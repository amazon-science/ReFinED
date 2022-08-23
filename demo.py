import os

from doc_preprocessing.dataclasses import Span
from refined.processor import Refined

refined = Refined.from_pretrained(model_name='wikipedia_model',
                                  entity_set='wikipedia',
                                  data_dir=os.path.join(os.path.expanduser('~'), '.cache/refined/'),
                                  download_files=True,
                                  debug=False,
                                  use_precomputed_descriptions=True,
                                  requires_redirects_and_disambig=True)

ents = refined.process_text("England won the FIFA World Cup in 1966.")
print([(ent.text, ent.pred_entity_id, ent.coarse_type, ent.coarse_mention_type, ent.entity_label,
        ent.failed_class_check) for ent in ents])

ents = refined.process_text("The population of England is 120,000")
print([(ent.text, ent.pred_entity_id, ent.coarse_type, ent.coarse_mention_type) for ent in ents])

ents = refined.process_text("The net worth of Jeff Bezos is $120B")
print([(ent.text, ent.pred_entity_id, ent.coarse_type, ent.coarse_mention_type) for ent in ents])

ents = refined.process_text("It takes 1 hour to cook a potato")
print([(ent.text, ent.pred_entity_id, ent.coarse_type, ent.coarse_mention_type) for ent in ents])

ents = refined.process_text("The first book in the Harry Potter series is Harry Potter and the Philosopher's Stone")
print([(ent.text, ent.pred_entity_id, ent.coarse_type, ent.coarse_mention_type) for ent in ents])

ents = refined.process_text("Donald Trump got married at the age of 40 years old.")
print([(ent.text, ent.pred_entity_id, ent.coarse_type, ent.coarse_mention_type) for ent in ents])

ents = refined.process_text("95% of people agree that something is true.")
print([(ent.text, ent.pred_entity_id, ent.coarse_type, ent.coarse_mention_type) for ent in ents])

ents = refined.process_text("Joe Biden is 1.80m.", spans=[Span(text='Joe Biden', start=0, ln=10)])
print([(ent.text, ent.pred_entity_id, ent.coarse_type, ent.coarse_mention_type) for ent in ents])

ents = refined.process_text("The death toll was 100,000.")
print([(ent.text, ent.pred_entity_id, ent.coarse_type, ent.coarse_mention_type) for ent in ents])
