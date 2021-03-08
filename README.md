# Beat and Downbeat Tracking
ECS7006 Music Informatics 2021, Coursework 1

A Conv-biLSTM trained on the ballroom dataset.

### Dependencies
```
pip install -r requirements.txt
```

### Train
Change the __audio_dir__ and __annot_dir__ in train.py and eval.py to the audio and annotation directories.

```
  python train.py --cuda --log_dir=/path/to/tensorboard/logs/ --checkpoint_dir=/path/to/checkpoints/
```

### Evaluate

```
  python eval.py --load_model=checkpoints/checkpoint_best
```

### Beat-tracking Wrapper

```
from eval import beatTracker
beats, downbeats = beatTracker(inputFile)
```
Please refer to __example_and_figures.ipynb__ or detailed examples.
