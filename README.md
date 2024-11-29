<div align="center">

# Illumination-Aware Two-stage Enhancement for Low-Light Object Detection

Bohan Xiong, Kan Chang, Mingyang Ling, Shilin Huang, Shucheng Xia, Ran Wei

[![python](https://img.shields.io/badge/-Python_3.6_%7C_3.7-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

</div>


The codes will be released after our paper is accepted.

# Dependencies

* python==3.8.0
* torch==1.11.0
* torchvision==0.12.0
* numpy==1.23.4
* opencv-python==4.8.1.78
  
```
cd IT-YOLO
pip install -r ./requirements.txt
```

# Datasets
Please download the processed datasets and pretrained models from the anonymous Github links below.

[Baidu Netdisk](https://pan.baidu.com/s/1j_v8EdY9l0YXxkjq6lD66w) password：n1n3

# Folder structure
Download the datasets and pretrained models first. Please prepare the basic folder structure as follows.
```
/parent_folder
  /datasets   # folder for datasets 
    /ExDark_10
    ...
  /DE-YOLO
    /data     # config files for datasets
    /models   # python files for DE-YOLO
    /pretrained_models  # folder for pretrained models
    requirements.txt
    README.md
    ...
```

# Quick Test
## Evaluation on real-world low-light images from ExDark
```
test.py --weights ./pretrained_models/IT_YOLO.pt --data ExDark_10.yaml --img 544 --batch-size 1
```
