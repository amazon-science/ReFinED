# ReFinED
## Overview
ReFinED is an entity linking (EL) system which links entity mentions in documents to their corresponding entities in Wikipedia or Wikidata.
The combination of accuracy, speed and scale of ReFinED means the system is capable of being deployed to extract entities from web-scale  datasets with higher accuracy and an order of magnitude lower cost than existing approaches.

### Model Architecture
In summary, ReFinED uses a Transformer model to perform mention detection, entity typing, and entity disambiguation for all mentions in a document in a single forward pass. The model is trained on a dataset we generated dataset using Wikipedia hyperlinks, which consists of over 150M entity mentions. The model uses entity descriptions and fine-grained entity types to perform linking. Therefore, new entities can be added to the system without retraining.

#### ReFinED Paper
The ReFinED model architecture is described in the paper below (https://arxiv.org/abs/2207.04108):
```bibtex
@inproceedings{ayoola-etal-2022-refined,
    title = "{R}e{F}in{ED}: An Efficient Zero-shot-capable Approach to End-to-End Entity Linking",
    author = "Ayoola Tom, Tyagi Shubhi, Fisher Joseph, Christodoulopoulos Christos, Pierleoni Andrea",
    booktitle = "NAACL",
    year = "2022"
}

```
 

#### Incorporating Knowledge Base Information Paper
The following paper is an extension of ReFinED which incorporates Knowledge Base (KB) information into the ED model in a fully differentiable and scalable manner (https://arxiv.org/abs/2207.04106):
```bibtex
@inproceedings{ayoola-etal-2022-improving,
    title = "Improving Entity Disambiguation by Reasoning over a Knowledge Base",
    author = "Ayoola Tom, Fisher Joseph, Pierleoni Andrea",
    booktitle = "NAACL",
    year = "2022"
}
```

### Examples
While classical NER systems, such as widely used spaCy, classify entities to high-level classes (e.g. PERSON, LOCATION, NUMBER, ...; 26 in total for spaCy), ReFinED supports over 1k low-level classes (e.g. Human, Human born in United States, Politician, Screen Writer, Association Football Player, Guitarist, ...). As an example, for the sentence "England qualified for the 1970 FIFA World Cup in Mexico as reigning champions.", ReFinED predicts "England" → {national football team} and "Mexico" → {country}; while spaCy maps both "England" and "Mexico" → {GPE - country}. Using fine-grained classes, the model is able to probabilistically narrow-down the set of possible candidates for "England" leading to correct disambiguation of the entity. Additionally, ReFinED uses textual descriptions of entities to perform disambiguation.

## Library

### Getting Started
The setup for ReFinED is very simple because the data files and datasets are downloaded automatically.
1. Install the dependencies using the command below:
```commandline
pip install -r requirements.txt
```

2. Add the `src` folder to your Python path. One way to do this is by running this command:
```commandline
export PYTHONPATH=$PYTHONPATH:src
```

### Running inference with a pretrained model
See `demo.py` for a working example. The model (`wikipedia_model`) requires approximately 60 GB disk space and 40 GB RAM.
```python
from refined.processor import Refined

refined = Refined.from_pretrained(model_name='wikipedia_model', 
                                  entity_set="wikipedia",
                                  data_dir="/path/to/download/data/to/", 
                                  download_files=True,
                                  use_precomputed_descriptions=True,
                                  device="cuda:0")

ents = refined.process_text("England won the FIFA World Cup in 1966.")

print([(ent.text, ent.pred_entity_id, ent.pred_types) for ent in ents])
```

#### Parameters

***model_name***: We provide two pretrained models
1. 'wikipedia_model': This is the model which matches the setup described in the paper
2. 'wikipedia_model_with_numbers': This model extends the above model, to also include detection of Spacy numerical data types in 
the mention detection layer ("DATE", "CARDINAL", "MONEY", "PERCENT", "TIME", "ORDINAL", "QUANTITY"). The detected types are 
available at ``ent.coarse_type``. If the coarse_type is detected as "DATE", the date will be normalised to a standard 
format available at ``ent.date``. All non-numerical types will have a coarse_type of "MENTION", and will be passed through 
the entity disambiguation layer to attempt to resolve them to a wikidata entity. 

***entity_set***: Set to "wikidata" to resolve against all ~33M (after some filtering) entities in wikidata (requires more memory) or to "wikipedia" to 
limit to resolving against the ~6M entities which have a wikipedia page.

***data_dir***: The local directory where the data/model files will be downloaded to/loaded from

***download_files***: Set to True the first time the code is run, to automatically download the data/model files from S3 to your 
local directory

***use_precomputed_descriptions***: Set to True to use precomputed embeddings of all descriptions of entities 
in the knowledge base (speeds up inference)

***device***: The device to load the model/run inference on


### Evaluation

#### Entity disambiguation
We provide the script `replicate_results.py` which replicates the ED results reported in our paper.

Entity disambiguation evaluation is run using the ``eval_all`` function:

```python

from refined.processor import Refined
from evaluation.evaluation import eval_all

refined = Refined.from_pretrained(model_name='wikipedia_model', 
                                  entity_set="wikipedia",
                                  data_dir="/path/to/download/data/to/", 
                                  download_files=True,
                                  use_precomputed_descriptions=True,
                                  device="cuda:0")

results_numbers = eval_all(refined=refined, datasets_dir="/path/to/datasets", filter_not_in_kb=True, download=True)

```

The script will automatically download the test dataset splits to ``datasets_dir``. Please ensure you have the 
permission to use each dataset for your usecase as defined by their independent licenses. 

**Expected results:**

We show the expected results from the evaluation scripts below. The numbers for "wikipedia_model" with entity set "wikipedia" most closely match 
the numbers in the paper (they differ marginally as we have updated to a newer version of wikipedia). For both models, 
performance on wikidata entities is slightly lower, as all entities in the datasets are linked to wikipedia entities (so 
adding wikidata entities just adds a large quantity of entities that will never appear in the gold labels). 

The performance of "wikipedia_model_with_numbers" is
slightly lower, which is expected as the model is only trained to disambiguate entities which are not one of the Spacy numerical types.

| model_name     | entity_set |  AIDA | MSNBC | AQUAINT | ACE2004 | CWEB |  WIKI |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | 
| wikipedia_model      | wikipedia       |  87.4 | 94.4 | 91.7 | 91.4 | 77.7 | 88.6 |
| wikipedia_model      | wikidata       |  85.6 | 92.8 | 90.4 | 91.1 | 76.3 | 88.2 |
| wikipedia_model_with_numbers   | wikipedia       | 85.1  | 93.5 | 90.3 | 91.7 | 76.4 | 89.4 |
| wikipedia_model_with_numbers   | wikidata        |  84.9 | 93.6 | 90.0 | 91.2 | 75.8 | 88.9 |

#### Entity linking
   
TODO

### Training

TODO: we will upload instructions for training a new model shortly

### Built With

* [PyTorch](https://pytorch.org/) - PyTorch is an open source machine learning library based on the Torch library.
* [Transformers](https://pytorch.org/hub/huggingface_pytorch-transformers/) - Implementations of Transformer models

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the CC-BY-NC 4.0 License.


