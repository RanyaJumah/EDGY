

## Data and Preprocessing
1. Download [LibriSpeech](http://www.openslr.org/12) (train-clean-100 set)
2. Prepare train/test splits JSON files
3. Preprocess wav files and extract train/test log-Mel spectrograms:
```
python preprocess.py in_dir=[?] dataset=[LibriSpeech]
```

## Training

* Train VQ-VAE model
```
python VQ-VAE/train_VQ.py checkpoint_dir=[?] dataset=[LibriSpeech]
```
