# Calibration Repository
## Repository created in the subject of Master Thesis

> Requirements:
  - Install required libraries\
  **NOTE:** It is recommended create a custom kernel where every package is. This prevents installing different versions and other problems.
 ```
pip install opencv-python
pip install matplotlib
```
**Note**:Make sure python3 is in use when calling pip. Otherwise use pip3.

Journey into calibration. From DLT, Zhang, binocular Calibration etc.. Some of the scripts are inspired by other github users. If so, credits are displayed.
- **0. Camera models with Numpy and Matplotlib**
  - In this script we will experience different camera models and play with then in terms of rotations and translations.\
    This script is a condensation of the work done by [Mario Larcher](https://github.com/mnslarcher/camera-models) with some added features done by me.
    In order to run the program it's necessary to import the libraries used in the original project.( those can be find in the github)
- **1.2 Ideal calibration with DLT.**\
  - In this part is it intented to achieve ideal calibration using DLT.\
    It is important to know the calibration matrix in advance. Otherwise, if we generate world points and image points at random with the purpose of finding the projection matrix, the points generate might not have real use correspondence.
  1. Calibration using DLT without normalization
  2. Calibration using DLT with normalization
- **1.6 Ideal stereo calibration**

  
