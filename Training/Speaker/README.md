## Speaker Embedding

To train Speaker model, we use the following [repository](https://github.com/clovaai/voxceleb_trainer), then we use the trained model to extract the speaker embadding.

#### Data preparation

The following script can be used to download and prepare the VoxCeleb dataset for training.

```
python ./dataprep.py --save_path /home/joon/voxceleb --download --user USERNAME --password PASSWORD 
python ./dataprep.py --save_path /home/joon/voxceleb --extract
python ./dataprep.py --save_path /home/joon/voxceleb --convert
```

In addition to the Python dependencies, `wget` and `ffmpeg` must be installed on the system.

#### Training examples

- AM-Softmax:
```
python ./trainSpeakerNet.py --model ResNetSE34L --encoder SAP --trainfunc amsoftmax --save_path data/exp1 --nSpeakers 5994 --batch_size 200 --scale 30 --margin 0.3 --train_list /home/joon/voxceleb/train_list.txt --test_list /home/joon/voxceleb/test_list.txt --train_path /home/joon/voxceleb/voxceleb2 --test_path /home/joon/voxceleb/voxceleb1
```

- Angular prototypical:
```
python ./trainSpeakerNet.py --model ResNetSE34L --encoder SAP --trainfunc angleproto --save_path data/exp2 --nPerSpeaker 2 --batch_size 200 --train_list /home/joon/voxceleb/train_list.txt --test_list /home/joon/voxceleb/test_list.txt --train_path /home/joon/voxceleb/voxceleb2 --test_path /home/joon/voxceleb/voxceleb1
```

#### Pretrained models

A pretrained model can be downloaded from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/models/baseline_lite_ap.model).

You can check that the following script returns: `EER 2.2322`. You will be given an option to save the scores.

```
python ./trainSpeakerNet.py --eval --model ResNetSE34L --trainfunc angleproto --save_path data/test --eval_frames 300 --test_list /home/joon/voxceleb/test_list.txt --test_path /home/joon/voxceleb/voxceleb1 --initial_model baseline_lite_ap.model
```

An alternative model using the `ResNetSE34` architecture can be downloaded from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/models/baseline_ap.model), and should return `EER 2.2587`.

#### Implemented loss functions
```
Softmax (softmax)
AM-Softmax (amsoftmax)
AAM-Softmax (aamsoftmax)
GE2E (ge2e)
Prototypical (proto)
Triplet (triplet)
Contrastive (contrastive)
Angular Prototypical (angleproto)
```

#### Implemented models and encoders
```
ResNetSE34 (SAP)
ResNetSE34L (SAP)
VGGVox40 (SAP, TAP, MAX)
```

#### Data

The [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) datasets.

The train list should contain the identity and the file path, one line per utterance, as follows:
```
id00000 id00000/youtube_key/12345.wav
id00012 id00012/21Uxsk56VDQ/00001.wav
```

The train list for VoxCeleb2 can be download from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt) and the
test list for VoxCeleb1 from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt).
