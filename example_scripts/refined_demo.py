from refined.data_types.base_types import Span
from refined.inference.processor import Refined

refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                  entity_set='wikipedia',
                                  use_precomputed_descriptions=False)

# Difficult disambiguation example
text = 'Michael Jordan is a Professor of Computer Science at UC Berkeley.'
spans = refined.process_text(text)
print('\n' + '****' * 10 + '\n')
print(text)
print(spans)
print('\n' + '****' * 10 + '\n')

# Example where entity mention spans are provided
text = "Joe Biden was born in Scranton."
spans = refined.process_text(text, spans=[Span(text='Joe Biden', start=0, ln=10),
                                          Span(text='Scranton', start=22, ln=8)])
print(text)
print(spans)
print('\n' + '****' * 10 + '\n')

# Example with numeric value
text = 'The population of England is 55,000,000.'
spans = refined.process_text(text)
print(text)
print(spans)
print('\n' + '****' * 10 + '\n')

# Example with currency
text = "The net worth of Elon Musk is $200B."
spans = refined.process_text(text)
print(text)
print(spans)
print('\n' + '****' * 10 + '\n')

# Example with time
text = "It takes 60 minutes bake a potato."
spans = refined.process_text(text)
print(text)
print(spans)
print('\n' + '****' * 10 + '\n')

# Example with an ordinal
text = "The first book in the Harry Potter series is Harry Potter and the Philosopher's Stone."
spans = refined.process_text(text)
print(text)
print(spans)
print('\n' + '****' * 10 + '\n')

# Example with age
text = "Barack Obama was 48 years old when he became president of the United States."
spans = refined.process_text(text)
print(text)
print(spans)
print('\n' + '****' * 10 + '\n')

# Example with percentage
text = "The rural population of England was 10% in 2020."
spans = refined.process_text(text)
print(text)
print(spans)
print('\n' + '****' * 10 + '\n')

# Example with height (quantity)
text = "Joe Biden is 1.82m tall."
spans = refined.process_text(text)
print(text)
print(spans)
print('\n' + '****' * 10 + '\n')

# Example with Wikidata entity that is not in Wikipedia
text = "Andreas Hecht is a professor."
spans = refined.process_text(text)
print(text)
print(spans)
print('\n' + '****' * 10 + '\n')

# Batched example
texts = ["Andreas Hecht is a professor.", "Michael Jordan is a Professor of Computer Science at UC Berkeley."]
docs = refined.process_text_batch(texts)
for doc in docs:
    print(f'Document: {doc.text}, spans: {doc.spans}')
print('\n' + '****' * 10 + '\n\n')

# Batched example with spans
texts = ["Joe Biden was born in Scranton."] * 2
# deep copy the Spans otherwise in-place modifications can cause issues
spanss = [[Span(text='Joe Biden', start=0, ln=10), Span(text='Scranton', start=22, ln=8)] for _ in range(2)]
docs = refined.process_text_batch(texts=texts, spanss=spanss)
for doc in docs:
    print(f'Document: {doc.text}, spans: {doc.spans}')
print('\n' + '****' * 10 + '\n')
