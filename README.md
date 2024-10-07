<div align="center">

# Illumination-Aware Two-stage Enhancement for Low-Light Object Detection

Bohan Xiong, Kan Chang, Mingyang Ling, Shilin Huang, Shucheng Xia, Ran Wei

[![python](https://img.shields.io/badge/-Python_3.6_%7C_3.7-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

</div>

## 📌 Abstract

>Detecting objects in low-light conditions is challenging since the primary features of objects may be buried in dark regions. To tackle this issue, we propose IT-YOLO, an object detector that employs an illumination-aware two-stage enhancement approach. First, an illumination feature extractor is applied to capture the lighting condition of an image. Afterwards, these illumination-related features are utilized to guide both imagelevel and feature-level enhancements in IT-YOLO. With the additional guidance, the image-level enhancement adjusts the lowlight image using an estimated reciprocal curve, while the featurelevel enhancement modulates the multi-scale features extracted by the detection backbone. Experimental results demonstrate that IT-YOLO outperforms existing methods in terms of detection accuracy, while maintaining a low number of parameters and high running speed. Our source code and pre-trained model will be available at https://github.com/bohan301/IT-YOLO.

The codes will be released after our paper is accepted.

