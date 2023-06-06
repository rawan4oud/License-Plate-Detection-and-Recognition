# 🚗 License-Plate-Detection-and-Recognition 🚗
This repo contains an implementation of a license plate detection and recognition system.

# Description
This repository contains an implementation of a license plate detection and recognition system. The system takes an input image of a car, detects its license plate, and performs optical character recognition (OCR) to extract the license plate number.

The license plate detection and recognition system in this project consists of **three** primary stages:

1. License Plate Detection
2. Character Segmentation
3. Optical Character Recognition (OCR)

## 1. License Plate Detection:
In this stage, the system utilizes object detection techniques to identify and locate the license plate in the input image. The implementation uses transfer learning by retraining TensorFlow's pre-trained SSD MobileNet.

I would like to acknowledge the following video tutorial as a valuable reference for the license plate detection implementation in this project:
- **Title:** Automatic Number Plate Recognition using Tensorflow and EasyOCR Full Course in 2 Hours | Python
- **Author:** Nicholas Renotte 
- **Platform:** YouTube (https://www.youtube.com/watch?v=0-4p_QgrdbE)

*(check "1. Training.ipynb" for the implementation of this step)*

## 2. Character Segmentation:
After the license plate is detected, the system performs character segmentation to isolate individual characters within the license plate. This is achieved by analyzing the contours within the cropped license plate region.

*(check "2. Detection and Segmentation.ipynb" for the implementation of this step)*

## 3. Optical Character Recognition (OCR):
The final stage involves recognizing the characters on the license plate. Multiple models have been trained and implemented for character classification:

1. K-Nearest Neighbors (KNN)
2. Support Vector Machine (SVM)
3. Naive Bayes
4. Convolutional Neural Network (CNN)

For the first **three** models, various features have been explored to enhance character recognition accuracy. These features include:

- Pixel Intensity
- Sobel Edge Detection
- Canny Edge Detection
- Vertical and Horizontal Histograms
- Histogram of Gradients (HOG)
- Scale-Invariant Feature Transform (SIFT)
- Local Binary Patterns (LBP)


*Check "HOWTORUN.txt" for more details on how to run this project.*

