## DifFSS: Diffusion Model for Few-Shot Semantic Segmentation
This is the implementation of the auxiliary support generator in paper "DifFSS: Diffusion Model for Few-Shot Semantic Segmentation".

For more information, Please refer to the the paper on [[arXiv](https://arxiv.org/abs/2307.00773)].

## :hammer_and_wrench: Getting Started with ControlNet4FSS
### 1. Setup Environment 
```sh
    conda env create -f environment.yaml
    conda activate control
```
### 2. Download the pre-trained models
All models and detectors can be downloaded from [Hugging Face page](https://huggingface.co/lllyasviel/ControlNet). Make sure that SD models are put in "ControlNet/models" and detectors are put in "ControlNet/annotator/ckpts". Make sure that you download all necessary pretrained weights and detector models from that Hugging Face page.

For DifFSS, we utilized the seg, scribble, hed model of ControlNet.

### 3. Preparation of the dataset

Please download the following datasets:

+ PASCAL-5<sup>i</sup>: [**PASCAL VOC 2012**](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and [**SBD**](http://home.bharathh.info/pubs/codes/SBD/download.html)

+ COCO-20<sup>i</sup>: [**COCO 2014**](https://cocodataset.org/#download).

+ FSS-1000: [**FSS-1000**](https://github.com/HKUSTCV/FSS-1000).

### 4. Start generating

```sh
python FSSpregenerate.py --st 0 --end -1 --imgdir /data/user6/coco/ --maskdir /data/user6/coco/annotations/ --dstdir /data/user6/justtest/ --list ./list/coco_all.txt --dataset coco --guidance seg --save_control 0
```
Please refer to the source code for the functions of arguments.

## :trophy: Try DifFSS on FSS models

The DifFSS method can be easily applied to existing FSS models, please refer to one of our implementations on BAM [here](https://github.com/TrinitialChan/DifFSS-BAM).

