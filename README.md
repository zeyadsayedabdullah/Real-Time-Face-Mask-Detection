# Real-Time Face Mask Detection

This repository contains code and models for real-time face mask detection using TensorFlow and OpenCV.

The model's performance can be viewed in action in [this LinkedIn video](https://lnkd.in/d3u9bfJB).

## Files Included

1. **Real_Mask_Detect.ipynb**: Notebook for real-time face mask detection using a pre-trained MobileNetV2 model and OpenCV's deep learning module.
2. **deploy.prototxt**: Configuration file for the face detection model.
3. **res10_300x300_ssd_iter_140000.caffemodel**: Pre-trained weights for the face detection model.
4. **train_mask_detector.ipynb**: Notebook for training a custom face mask detector using a transfer learning approach with MobileNetV2.

## Real_Mask_Detect.ipynb

This notebook sets up a real-time face mask detection system using a pre-trained MobileNetV2 model for mask classification and OpenCV for face detection.

### Usage

Run this notebook to start the real-time face mask detection system. It uses your webcam to capture frames, detects faces using the `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`, and classifies each detected face as either wearing a mask or not.

## train_mask_detector.ipynb

This notebook trains a custom face mask detection model using transfer learning with MobileNetV2 as the base model.

### Data

The model was trained on a dataset containing images of my face with and without masks. The dataset includes:

- **with_mask/**: Directory containing images of me wearing mask.
- **without_mask/**: Directory containing images of me not wearing mask.

### Usage

1. Set up your dataset in the specified directory (`DIRECTORY` variable in the notebook).
2. Run the notebook to preprocess images, train the model, and evaluate its performance.

## Dependencies

- TensorFlow
- OpenCV
- NumPy
- imutils
- Matplotlib
- scikit-learn

Make sure to have these dependencies installed to run the notebooks and execute the face mask detection.

## Credits

- Face detection model based on [OpenCV's deep learning module](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector).
- Face mask detection model using transfer learning with MobileNetV2.
- Inspired by various tutorials and resources on face detection and deep learning.

LinkedIn: [ https://www.linkedin.com/in/zeyadsayed/ ]
HuggingFace: [ https://huggingface.co/Zeyad-Sayed ]
Kaggle: [ https://www.kaggle.com/zeyadsayedadbullah ]
