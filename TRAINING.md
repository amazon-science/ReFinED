# Training the ReFinED model
## Training
ReFinED is trained on the entirety of English Wikipedia which consists of over 150M hyperlinks (and title entity mentions).
Training the ReFinED model on this large volume of high quality data leads to a good quality general EL model, which can be further
fine-tuned on specific datasets.

## Training instructions
1. Ensure the ReFinED source directory is in your Python path:
```
export PYTHONPATH=$PYTHONPATH:src
```
2. Run the train.py script (to list the arguments and help, run `train.py -h`):
```
python3 src/refined/training/train/train.py --experiment_name test
```

The `train.py` script will automatically download the training and development split for the Wikipedia hyperlinks dataset and 
use this for training and evaluation. Training takes around 15 hours per epoch when using 8 V100 32 GB GPUs with a per gpu batch_size of 16.

## Using a trained model
To use a trained model provide the file path to the directory containing the trained model (requires `model.pt` and `config.json` files) to the refined.from_pretrained(...) method as follows:
```python
refined = Refined.from_pretrained(model_name='<absolute_file_path_to_directory_containing_trained_model>',
                                  entity_set='wikipedia',
                                  use_precomputed_descriptions=False)
```
Then use the `refined` object as usual.

### Speeding up the model by pre-computing description embeddings
The steps above will not use precomputed entity description embeddings. This means the embeddings will be computed on-the-fly, which doubles the inference time.
To generate precomputed_description_embeddings, run the `precompute_description_embeddings.py` script (or run `refined.precompute_description_embeddings()`) and then copy the embeddings file into the directory with the trained model. Lastly, ensure you have `use_precomputed_descriptions=True` when you call `Refined.from_pretrained(...)`. 
