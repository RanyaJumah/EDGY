# Training 

## Dependencies
* Install pip dependencies:
```
pip3 install requirements.txt
```


## Data and Preprocessing
* [LibriSpeech](http://www.openslr.org/12) 

## Training

* Train VQ-VAE model
```
python train.py checkpoint_dir=path/to/checkpoint_dir dataset=../...
```

* Train Speaker model
```
pip3 install requirements.txt
```

## References
1. [VQ-VAE] https://github.com/bshall/VectorQuantizedCPC
2. [Speaker Trainer] https://github.com/clovaai/voxceleb_trainer
