import random
import string

from refined.data_types.base_types import Span
from refined.inference.processor import Refined
from refined.torch_overrides.data_parallel_refined import DataParallelReFinED

refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                  entity_set='wikipedia',
                                  use_precomputed_descriptions=False)

refined.model = DataParallelReFinED(refined.model, device_ids=list(range(8)), output_device='cuda:0')
refined.model = refined.model.to('cuda:0')

# Difficult disambiguation example with long random string before it to ensure the document is split into chunks.
text = ''.join(random.choices(string.ascii_uppercase, k=512 * 128)) + \
       '.\n\n Michael Jordan is a Professor of Computer Science at UC Berkeley.'
spans = refined.process_text(text)
print('\n' + '****' * 10 + '\n')
print("Michael Jordan is a Professor of Computer Science at UC Berkeley.")
print(spans)


# Difficult disambiguation example
text = 'Michael Jordan is a Professor of Computer Science at UC Berkeley.'
spans = refined.process_text(text)
print('\n' + '****' * 10 + '\n')
print("Michael Jordan is a Professor of Computer Science at UC Berkeley.")
print(spans)
print('\n' + '****' * 10 + '\n')

# Example where entity mention spans are provided
text = "Joe Biden was born in Scranton. " + ''.join(random.choices(string.ascii_uppercase, k=512 * 128))
spans = refined.process_text(text, spans=[Span(text='Joe Biden', start=0, ln=10),
                                          Span(text='Scranton', start=22, ln=8)])
print(text)
print(spans)
print('\n' + '****' * 10 + '\n')
