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
python train_VQ.py checkpoint_dir=[?] dataset=[LibriSpeech]
```

* Train Speaker model
```
python train_speaker.py checkpoint_dir=[?] dataset=[LibriSpeech]
```

## References
1. [VQ-VAE] https://github.com/bshall/VectorQuantizedCPC
2. [Speaker Trainer] https://github.com/clovaai/voxceleb_trainer
