# Training 

## Dependencies
* Install pip dependencies:
```
pip3 install requirements.txt
```


## Data and Preprocessing
1. Download [LibriSpeech](http://www.openslr.org/12) 
2. Use train/test splits 
3. Preprocess wav files and extract train/test log-Mel spectrograms:
```
python preprocess.py in_dir=[?] dataset=[LibriSpeech]
```

## Training

* Train VQ-VAE model
```
python VQ-VAE/train_VQ.py checkpoint_dir=[?] dataset=[LibriSpeech]
```

* Train Speaker model
```
python speaker/train_speaker.py checkpoint_dir=[?] 
```

## References
1. [VQ-VAE] https://github.com/bshall/VectorQuantizedCPC
2. [Speaker Trainer] https://github.com/clovaai/voxceleb_trainer
