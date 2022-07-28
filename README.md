# Calibration Repository
## Repository created in the subject of Master Thesis

> Requirements:
  - Install required libraries - **requirements.txt**\
  **NOTE:** It is recommended create a custom kernel where every package is. This prevents installing different versions and other problems.
**Note**:Make sure python3 is in use when calling pip. Otherwise use pip3.

Journey into calibration. From DLT, Zhang, binocular Calibration etc.. Some of the scripts are inspired by other github users. If so, credits are displayed.
- **0. Camera models with Numpy and Matplotlib**
  - In this script we will experience different camera models and play with then in terms of rotations and translations.\
    This script is a condensation of the work done by [Mario Larcher](https://github.com/mnslarcher/camera-models) with some added features done by me.
    In order to run the program it's necessary to import the libraries used in the original project.( those can be find in the github)
- **1. Single camera calibration using DLT**
  - This script does calibration using DLT using predefined world and camera points.
- **1.1 Single Camera calibration usgin DLT and normalization with predefined points.**
  - This scripts adds to the previous normalization and compares both option( w/ and without normalization).
  - It will use predefined world and image coordinates and then calculate the projective matrix using DLT transform and retrieve the geometric error.
- **1.2 Ideal single camera calibration using DLT and random points.**
  - Compared with the point 1.1, this script uses random world points and obtain it's correspondence in the 2D plan. From that, calibration it's done.
  - In this part is it intented to achieve ideal calibration using DLT.\
    It is important to know the calibration matrix in advance. Otherwise, if we generate world points and image points at random with the purpose of finding the projection matrix, the points generate might not have real use correspondence.
- **1.2.1 Calibration using DLT w/ normalization**
  -  Adds to the previous scripts normalization
- **1.2.2 Calibration using DLT w/ normalization and LM**
  - Adds to the previous scripts Levenberg-Marquad optimization.
- **1.3 Harry's corner check**
  - This scripts uses an image to check harry's corner detection method from OpenCV.
- **1.4 Still in the worls**
- **1.5 Ideal Zhang calibration using OpenCV**
  - Calibration using ideal checkerboards and using OpenCV Zhang calibration method.
- **1.6 Ideal stereo calibration**
  - Ideal calibration using ideal checkerboards using OpenCV Zhang calibration method for stereo cameras.
- **2.1 Ideal stereo calibration with pixel**
   - This scripts, adds pixelization to the previous.
- **3. Realistic simulation setup**
- **3.3.4 Realistic simulation at 2 meters**
  - Using the previous method, it is calculate the calibration parameters and the subsquent matrixes and error for a calibration at 2 meters
- **3.3.4.1 Realistic simulation at 2 meters for 12x8**
  - Using the previous script, the only change is the chessboard calibration, from 13x9 to 12x8.
- **3.3.5 Reprojection error realistic simulation**
  - Reprojection error and triangulation for 2 meters with the data from 20, 80 and 180 chessboards at 2 meters
- **3.3.5.1 Reprojection error realistic simulation for 180**
  - Reprojection error and triangulation for 180 chessboard at 2 meters. This script tests different subpixelgrid( integration factor) 32,64,128.
- **4.1 Camera calibration using the physical setup**
  - Using images from the physical setup, calibrate using OpenCV at 2 meters.
- **4.1.1 Reprojection and triangulation error**
  - Test the reprojection and triangulation error using the matrixes from the previous script.
