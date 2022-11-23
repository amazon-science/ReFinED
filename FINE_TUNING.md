# Fine-tuning the ReFinED model
## Why fine-tune?
ReFinED was trained to perform entity linking (EL) on the entirety on English Wikipedia. This means the model will perform very well when the document text is similar to the kind of text that appears on Wikipedia.
However, performance can be improved on other domains by fine-tuning the model on a relevant dataset. The same is true if the kind of entities to detect/link differs from the kind of entities hyperlinked on Wikipedia.

## Fine-tuning instructions
1. Ensure the ReFinED source directory is in your Python path:
```
export PYTHONPATH=$PYTHONPATH:src
```
2. Run the fine_tune.py script (to list the arguments and help, run `fine_tune.py -h`):
```
python3 src/refined/training/fine_tune.py --experiment_name test
```

The `fine_tune.py` script will automatically download the training and development split for the CoNLL-AIDA dataset and 
use this for fine-tuning and evaluation. The default arguments are the ones used to produce the results reported in the ReFinED (NAACL 2022) paper.

## Using a fine-tuned model
To use a fine-tuned model provide the file path to the directory containing the fine-tuned model (requires `model.pt` and `config.json` files) to the refined.from_pretrained(...) method as follows:
```python
refined = Refined.from_pretrained(model_name='<absolute_file_path_to_directory_containing_fine_tuned_model>',
                                  entity_set='wikipedia',
                                  use_precomputed_descriptions=False)
```
Then use the `refined` object as usual.

### Speeding up the model by pre-computing description embeddings
The steps above will not use precomputed entity description embeddings for inference. This means the embeddings will be computed on-the-fly, which doubles the inference time.
To generate precomputed description embeddings, run the `precompute_description_embeddings.py` script (or run `refined.precompute_description_embeddings()`) and then copy the embeddings file into the directory with the fine-tuned model. Lastly, ensure you have `use_precomputed_descriptions=True` when you call `Refined.from_pretrained(...)`. 


## Fine-tuning on custom datasets
To add a custom dataset:
1. Add a method to the `Datasets` class (`dataset_factory.py`) such as `get_custom_dataset_name_docs(...)` and return an iterable of `Doc` objects.
2. The `Doc` objects returned should be created using the following method:
```
Doc.from_spans_with_text(text='insert_document_text', spans=[Span(...), ...], md_spans=[Span(...), ...])
```
Where:
- `text` is the full text for the document.
- `spans` is a list of `Span` (used for entity disambiguation and typing training) objects where each span has a `gold_entity` set to the correct (annotated) entity (using the wikidata ID) and `coarse_type="MENTION"` .
- `md_spans` a list of `Span` (used for mention detection training) objects where do each span does not have a `gold_entity` and `coarse_type` can be set to any of the types ("MENTION", """DATE", "CARDINAL", "MONEY", "PERCENT", "TIME", "ORDINAL", "QUANTITY").
3. Then modify `fine_tune.py` to read your custom dataset:
```
training_dataset = DocDataset(
        docs=list(datasets.get_custom_dataset_docs(split="train", ...),
        preprocessor=refined.preprocessor,
    )
evaluation_dataset_name_to_docs = {
        "CUSTOM": list(datasets.get_custom_dataset_docs(
            split="dev",...
        ))
    }
```
4. Then run the `fine_tune.py` script the same as before.
5. If your dataset contains numbers and dates ensure you set the CLI arg `model_name=wikipedia_model_with_numbers`, if not use then it is fine to use the default `model_name=wikipedia_model`. Similarly, if your dataset contains entities that are not in Wikipedia (but are in Wikidata) set `entity_set='wikidata'` instead of the default `entity_set='wikipedia'`.

## Fine-tuning programmatically, without modifying source code
Alternatively, fine-tuning can be done programmatically as follows (example code in `fine_tuning_example.py`):
```
from refined.evaluation.evaluation import get_datasets_obj
from refined.inference.processor import Refined
from refined.training.fine_tune.fine_tune import fine_tune_on_docs

refined = Refined.from_pretrained(model_name='wikipedia_model'
                                  entity_set='wikipedia',
                                  use_precomputed_descriptions=False)
                                  
train_docs: Iterable[Doc] = ... # any method that returns Docs with Wikidata entity ids (qcodes) (spans used for ED + ET, and md_spans used for MD)
eval_docs: Iterable[Doc] = ... # any method that returns Docs with Wikidata entity ids (qcodes)
fine_tune_on_docs(refined=refined, train_docs=train_docs, eval_docs=eval_docs)

```
This method has the same functionality as the `fine_tune.py` script, and the same arguments can be provided (see the `fine_tune.py -h` for more information).
