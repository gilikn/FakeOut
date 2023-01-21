<div align="center">
<img src="fakeout/images/fakeout_logo.svg" alt="logo" width=500></img>
</div>

<!-- # FakeOut -->
# FakeOut: Leveraging Out-of-domain Self-supervision for Multi-modal Video Deepfake Detection

This is a Haiku & Jax implementation of the <i>FakeOut</i> paper.

<p> 
</p>
<div align="center">
<img src="fakeout/images/Architecture.svg" alt="architecture"></img>
</div>
<p> 
</p>

The figure above describes the two parts of <i>FakeOut</i>. On the left is the pre-training out-of-domain self-supervision phase, via [MMV](https://vitalab.github.io/article/2022/04/14/MultiModalVersatileNetworks.html). On the right is the adaption phase of <i>FakeOut</i> to the video deepfake detection task.

## Setup

Prepare the videos of the desired dataset in train, val, test dedicated directories. 
Run our face tracking pipeline as documented in the [FacePipe](https://github.com/gilikn/FacePipe) repo.

To use FakeOut, first clone the repo:
```
cd /local/path/for/clone
git clone https://github.com/gilikn/FakeOut.git
cd FakeOut
```
Install requirements
```
pip install requirements.txt
```
Prepare your dataset using the script:
```
python fakeout/data/data_preparation.py
    --dataset_name {face_forensics, dfdc, deeper_forensics, face_shifter, celeb}
    --split {train, test}
    --videos_path /path/to/facepipe/output
```
Our checkpoints are made available in the following link:
```
```
For inference, execute the following script (DFDC test-set running example):
```
python fakeout/inference.py
    --model_path /local/path/to/checkpoint
    --dataset_name {face_forensics, dfdc, deeper_forensics, face_shifter, celeb}
    --use_audio {true, false}
    --mlp_first_layer_size {6144 (for Video&Audio TSM-50x2), 4096 (for Video TSMx2)}
    --num_test_windows {e.g. 10}
```
For fine-tuning, execute the following script (FF++ train-set running example):
```
Will be availabe soon...
```





## BibTex
If you find <i>FakeOut</i> useful for your research, please cite the paper:
```bib
@article{knafo2022fakeout,
  title={FakeOut: Leveraging Out-of-domain Self-supervision for Multi-modal Video Deepfake Detection},
  author = {Knafo, Gil and Fried, Ohad},
  journal={arXiv preprint arXiv:2212.00773},
  year={2022}
}
```
