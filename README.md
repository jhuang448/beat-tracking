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
Note: you need the hdf5 file for the test set generated by train.py to run the evaluation.

```
  python eval.py --load_model=checkpoints/checkpoint_best
```

### Beat-tracking Wrapper

```
from eval import beatTracker
beats, downbeats = beatTracker(inputFile)
```
Please refer to __example_and_figures.ipynb__ for detailed examples.

### References
[1] Sebastian Böck, Florian Krebs, and Gerhard Widmer.Joint beat and downbeat tracking with recurrent neuralnetworks. InISMIR, pages 255–261. New York City,2016.

[2] Florian Krebs, Sebastian Böck, Matthias Dorfer, andGerhard Widmer. Downbeat tracking using beat syn-chronous features with recurrent neural networks. InISMIR, pages 129–135, 2016.

[3] Sebastian  Böck  and  Markus  Schedl.  Enhanced  beattracking with context-aware neural networks. InProc.Int. Conf. Digital Audio Effects, pages 135–139, 2011.
