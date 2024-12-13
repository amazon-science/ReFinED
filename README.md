# ReFinED
## Quickstart
```commandline
pip install https://github.com/amazon-science/ReFinED/archive/refs/tags/V1.zip
```
```python
from refined.inference.processor import Refined
refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                  entity_set="wikipedia")
spans = refined.process_text("<add_text_here>")
```

## Overview
ReFinED is an entity linking (EL) system which links entity mentions in documents to their corresponding entities in Wikipedia or Wikidata (over 30M entities).
The combination of accuracy, speed, and scalability of ReFinED means the system is capable of being deployed to extract entities from web-scale datasets with higher accuracy and an order of magnitude lower cost than existing approaches.

### News
- **(November 2022)**
  - Code refactoring 🔨
  - Increased inference speed by 2x (replicates results from our paper) 💨
  - Released `aida_model` (trained on news articles) and `questions_model` (trained on questions) to replicate the results from our paper ✅
  - New features 🚀
    - Entity linking evaluation code 
    - Fine-tuning script (allows use of custom datasets)
    - Training script
    - Data generation script (includes adding additional entities).


### Hardware Requirements
ReFinED has a low hardware requirement. For fast inference speed, a GPU should be used, but this is not a strict requirement. 


### Model Architecture
In summary, ReFinED uses a Transformer model to perform mention detection, entity typing, and entity disambiguation for all mentions in a document in a single forward pass. The model is trained on a dataset we generated dataset using Wikipedia hyperlinks, which consists of over 150M entity mentions. The model uses entity descriptions and fine-grained entity types to perform linking. Therefore, new entities can be added to the system without retraining.

#### ReFinED Paper
The ReFinED model architecture is described in the paper below (https://arxiv.org/abs/2207.04108):
```bibtex
@inproceedings{ayoola-etal-2022-refined,
    title = "{R}e{F}in{ED}: An Efficient Zero-shot-capable Approach to End-to-End Entity Linking",
    author = "Tom Ayoola, Shubhi Tyagi, Joseph Fisher, Christos Christodoulopoulos, Andrea Pierleoni",
    booktitle = "NAACL",
    year = "2022"
}
```
 

#### Incorporating Knowledge Base Information Paper
The following paper is an extension of ReFinED which incorporates Knowledge Base (KB) information into the ED model in a fully differentiable and scalable manner (https://arxiv.org/abs/2207.04106):
```bibtex
@inproceedings{ayoola-etal-2022-improving,
    title = "Improving Entity Disambiguation by Reasoning over a Knowledge Base",
    author = "Tom Ayoola, Joseph Fisher, Andrea Pierleoni",
    booktitle = "NAACL",
    year = "2022"
}
```

### Examples
While classical NER systems, such as widely used spaCy, classify entities to high-level classes (e.g. PERSON, LOCATION, NUMBER, ...; 26 in total for spaCy), ReFinED supports over 1k low-level classes (e.g. Human, Football Team, Politician, Screenwriter, Association Football Player, Guitarist, ...). As an example, for the sentence "England qualified for the 1970 FIFA World Cup in Mexico as reigning champions.", ReFinED predicts "England" → {national football team} and "Mexico" → {country}; while spaCy maps both "England" and "Mexico" → {GPE - country}. Using fine-grained classes, the model is able to probabilistically narrow-down the set of possible candidates for "England" leading to correct disambiguation of the entity. Additionally, ReFinED uses textual descriptions of entities to perform disambiguation.

## Library

### Getting Started
The setup for ReFinED is very simple because the data files and datasets are downloaded automatically.
1. Install the dependencies using the command below:
```commandline
pip install -r requirments.txt
```
If the command above fails (which currently, happens on a Mac), run the commands below instead:
```commandline
conda create -n refined38 -y python=3.8 && conda activate refined38
conda install -c conda-forge python-lmdb -y
pip install -r requirments.txt
```

2. Add the `src` folder to your Python path. One way to do this is by running this command:
```commandline
export PYTHONPATH=$PYTHONPATH:src
```
3. Now you can use ReFinED is your code as follows:
```python
from refined.inference.processor import Refined
refined = Refined.from_pretrained(...)
```

### Importing ReFinED as a library
To import the ReFinED model into your existing code run the commands below (note that the conda commands are only needed on a Mac):
```commandline
pip install https://github.com/amazon-science/ReFinED/archive/refs/tags/V1.zip
```

Alternatively, if the command above does not work, try the commands below which will install some dependencies using conda.
```commandline
conda create -n refined38 -y python=3.8 && conda activate refined38
conda install -c conda-forge python-lmdb -y
git clone https://github.com/amazon-science/ReFinED.git
cd ReFinED
python setup.py bdist_wheel --universal
pip install dist/ReFinED-1.0-py2.py3-none-any.whl
cd ..
```

### Inference - performing EL with a trained model
We have released several trained models that are ready to use. See the code below or `example_scripts/refined_demo.py` for a working example. Inference speed can be improved by setting use_precomputed_descriptions=True which increases disk usage.

```python
from refined.inference.processor import Refined


refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                  entity_set="wikipedia")

spans = refined.process_text("England won the FIFA World Cup in 1966.")

print(spans)
```
Expected output:
```text
[['England', Entity(wikidata_entity_id=Q47762, wikipedia_entity_title=England national football team), 'ORG'], ['FIFA World Cup', Entity(wikidata_entity_id=Q19317, wikipedia_entity_title=FIFA World Cup), 'EVENT'], ['1966', Entity(...), 'DATE']]
```
Note that str(span) only returns a few fields of the returned object for readability. Many other fields, such as top-k predictions and predicted fine-grained entity types, are also accessible from the returned `Span`.

#### Parameters
***model_name***: We provide four pretrained models
1. 'wikipedia_model': This is the model which matches the setup described in the paper
2. 'wikipedia_model_with_numbers': This model extends the above model, to also include detection of SpaCy numerical data types in 
the mention detection layer ("DATE", "CARDINAL", "MONEY", "PERCENT", "TIME", "ORDINAL", "QUANTITY"). The detected types are 
available at ``span.coarse_type``. If the coarse_type is detected as "DATE", the date will be normalised to a standard 
format available at ``span.date``. All non-numerical types will have a coarse_type of "MENTION", and will be passed through 
the entity disambiguation layer to attempt to resolve them to a wikidata entity.
3. 'aida_model': This is the model which matches the setup described in the paper for fine-tuning the model on AIDA for entity linking. Note that this model is different to the model fine-tuned on AIDA for entity disambiguation only, which is also described in the paper.
4. 'questions_model': This model is fine-tuned on short question text (lowercase text). The model was fine-tuned on the WebQSP EL dataset and the setup is described in our paper.

***entity_set***: Set to "wikidata" to resolve against all ~33M (after some filtering) entities in wikidata (requires more memory) or to "wikipedia" to 
limit to resolving against the ~6M entities which have a wikipedia page.

***data_dir*** (optional): The local directory where the data/model files will be downloaded to/loaded from (defaults to ~/.cache/refined/).

***download_files*** (optional): Set to True the first time the code is run, to automatically download the data/model files from S3 to your 
local directory. Files will not be downloaded if they already exist but network calls will still be made to compare timestamps.

***use_precomputed_descriptions*** (optional): Set to True to use precomputed embeddings of all descriptions of entities 
in the knowledge base (speeds up inference).

***device*** (optional): The device to load the model/run inference on.


### Evaluation

#### Entity disambiguation
We provide the script `replicate_results.py` which replicates the results reported in our paper.

Entity disambiguation evaluation is run using the ``eval_all`` function:

```python
from refined.inference.processor import Refined
from refined.evaluation.evaluation import eval_all

refined = Refined.from_pretrained(model_name='wikipedia_model',
                                  entity_set="wikipedia")

results_numbers = eval_all(refined=refined, el=False)
```

The script will automatically download the test dataset splits to `~/.cache/refined/`. Please ensure you have the 
permission to use each dataset for your use case as defined by their independent licenses. 

**Expected results:**

We show the expected results from the evaluation scripts below. The numbers for "wikipedia_model" with entity set "wikipedia" most closely match 
the numbers in the paper (they differ marginally as we have updated to a newer version of Wikipedia). For both models, 
performance on Wikidata entities is slightly lower, as all entities in the datasets are linked to Wikipedia entities (so 
adding Wikidata entities just adds a large quantity of entities that will never appear in the gold labels). 

The performance of "wikipedia_model_with_numbers" is
slightly lower, which is expected as the model is also trained to identify numerical types.

| model_name     | entity_set |  AIDA | MSNBC | AQUAINT | ACE2004 | CWEB | WIKI |
| ----------- | ----------- | ----------- |-------|---------| ----------- | ----------- |------| 
| wikipedia_model      | wikipedia       |  87.4 | 94.5  | 91.9    | 91.4 | 77.7 | 88.7 |
| wikipedia_model      | wikidata       |  85.6 | 92.8  | 90.4    | 91.1 | 76.3 | 88.2 |
| wikipedia_model_with_numbers   | wikipedia       | 85.1  | 93.5  | 90.3    | 91.7 | 76.4 | 89.4 |
| wikipedia_model_with_numbers   | wikidata        |  84.9 | 93.6  | 90.0    | 91.2 | 75.8 | 88.9 |

#### Entity linking
Entity linking evaluation is run using the ``eval_all`` function with `el=True`:

```python
from refined.inference.processor import Refined
from refined.evaluation.evaluation import eval_all

refined = Refined.from_pretrained(model_name='aida_model',
                                  entity_set="wikipedia")

results_numbers = eval_all(refined=refined, el=True)
```

The results below slightly differ from the ones reported in our paper (which were produced by the Gerbil framework using an older version of Wikipedia). The `wikipedia_model` is not trained on Wikipedia hyperlinks only. Whereas, the `aida_model` is fine-tuned on the AIDA training dataset (with the weights initialised from the `wikipedia_model`). 

| model_name                              | entity_set | AIDA | MSNBC | 
|-----------------------------------------| ----------- |------|-------| 
| aida_model          | wikipedia       | 85.0 | 75.1  |
| wikipedia_model  | wikipedia       | 78.3 | 73.4  |

We observe that most EL errors on the AIDA dataset are actually dataset annotation errors.
The AIDA dataset does not provide the entity label for every mention that can be linked to Wikipedia.
Instead, many mentions are incorrectly labelled as NIL mentions, meaning no corresponding Wikipedia page was found for
the mention (during annotation). This means that EL model predictions for these mentions will be unfairly considered as incorrect.
To measure the impact, we added an option to filter out model predictions which **exactly** align with
NIL mentions in the dataset:
```python
eval_all(refined=refined, el=True, filter_nil_spans=True)
```
We report 90.2 F1 on the AIDA dataset when we set `filter_nil_spans=True`, when using our "aida_model".

### Inference speed
We run the model over the AIDA dataset development split using our script, `benchmark_on_aida.py`.

| Hardware | Time taken to run EL on AIDA test dataset (231 news articles) | 
|----------|---------------------------------------------------------------| 
| V100 GPU | 6.5s                                                           
| T4 GPU   | 7.4s                                                          |
| CPU      | 29.7s                                                         |

The first time the model is loaded it will take longer because the data files need to be downloaded to disk.

### Fine-tuning
See [FINE_TUNING.md](FINE_TUNING.md) for instructions on how to fine-tune the model on standard and custom datasets.

### Training
See [TRAINING.md](TRAINING.md) for instructions on how to train the model on our Wikipedia hyperlinks dataset.

### Generating and updating the data files and training dataset
To regenerate **all** data files run the `preprocess_all.py` script. This script downloads the most recent Wikipedia and Wikidata dump and generates the data files and Wikipedia training dataset.
Note that ReFinED is capable of zero-shot entity linking, which means the data files (which will include recently added entities) can be updated without having retraining the model.

### Adding additional/custom entities
Additional entities (which are not in Wikidata) can be added to the entity set considered by ReFinED by running `preprocess_all.py` script with the argument `--additional_entities_file <path_to_file>`.
The file must be a jsonlines file where each row is the JSON string for an `AdditionalEntity`. Ideally, the entity types provided should be Wikidata classes such as "Q5" for human.

### Built With

* [PyTorch](https://pytorch.org/) - PyTorch is an open source machine learning library based on the Torch library.
* [Transformers](https://pytorch.org/hub/huggingface_pytorch-transformers/) - Implementations of Transformer models.
* Works with Python 3.8.10.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the Apache 2.0 License.

## Contact us
If you have questions please open Github issues instead of sending us emails, as some of the listed email addresses are no longer active.
