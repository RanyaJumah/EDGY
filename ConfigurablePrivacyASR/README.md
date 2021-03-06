```
This repo provides the code for 
"Configurable Privacy-Preserving Automatic Speech Recognition" paper.
```
Our pipeline consists of three basic components: Separation, ASR, and Discretization.
- Speech Separation (SEP),
Generating a set of non-overlapped speech signals from an audio signal with overlapped utterances by a varying degree.
- Speech Recognition (ASR),
Speech to text conversion is the process of converting spoken words into written texts.
- Speech Discretization (DIS),
Discovering discrete units (e.g. phonemes) of the speech signal while being invariant to speaker-specific and background noise.


## Requirements
- Install [SpeechBrain](https://speechbrain.readthedocs.io/en/latest/index.html): pip install speechbrain

## Dataset
- [Libri-light](https://github.com/facebookresearch/libri-light/tree/master/data_preparation)
- [LibriCSS](https://github.com/chenzhuo1011/libri_css)
- [phoneticsMix]()
- [WHAM]()
- Paralinguistics Datasets: [VoxCeleb](https://www.tensorflow.org/datasets/catalog/voxceleb), [Common Voice](https://www.tensorflow.org/datasets/catalog/common_voice), [CREMA-D](https://www.tensorflow.org/datasets/catalog/crema_d), and [SAVEE](https://www.tensorflow.org/datasets/catalog/savee)



## References
- [ZeroSpeech Challenge 2021](https://github.com/bootphon/zerospeech2021_baseline)
- [TRILL](https://github.com/google-research/google-research/tree/master/non_semantic_speech_benchmark)

*** Note: these instructions will be updated over the next few days
