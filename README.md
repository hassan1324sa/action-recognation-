# README: Video Classification using ConvLSTM and LRCN Models
## Overview
This project implements two deep learning models for video classification: a ConvLSTM model and an LRCN (Long-term Recurrent Convolutional Networks) model. Both models are trained on the UCF50 dataset, which contains various action videos, and predict the class label of a video sequence based on its frames.
The main goal is to extract frames from videos, preprocess them, train the models on these frames, and evaluate the models' performance.
Libraries Used:
- OpenCV: For video handling (reading and processing).
- Pafy: To work with YouTube videos, though it is not utilized directly here.
- Matplotlib: For plotting and visualizing.
- TensorFlow/Keras: For building and training the deep learning models.
- NumPy: For numerical operations.
- Scikit-learn: For splitting the dataset into training and testing sets.
# Project Structure:
1. Extracting Random Frames: Randomly selects 20 classes from the dataset and extracts a frame from each selected video to display.
2. Frame Extraction for Dataset: The function `framesExtract` reads the frames from each video, resizes them, normalizes them, and ensures the video sequence is of a fixed length (sequenceLength).
3. Creating the Dataset: The function `createDataSet` iterates through the selected classes (classNames), extracts frames from the videos, and stores the features (frames) and labels. Splits the dataset into training and testing sets using `train_test_split`.
4. ConvLSTM Model: The ConvLSTM model is created using the function `createConvLstmModel`. It consists of multiple `ConvLSTM2D` layers, interleaved with `MaxPooling3D` and `Dropout` layers. Trained on the dataset for 20 epochs.
5. LRCN Model: The function `createLRCN` builds an LRCN model that applies `Conv2D` and `MaxPooling2D` layers followed by an LSTM layer. Trained in a similar fashion as the ConvLSTM model.
6. Model Evaluation: Both models are evaluated using the `evaluate` function to determine the accuracy on the test set.
7. Model Prediction on a Video: The function `predictOnVideo` reads a video, processes its frames, and predicts the class of the video using the trained model.
# How to Run the Code
1. Dataset Preparation: Ensure the UCF50 dataset is available in the 'UCF50/' directory, with subdirectories named after the action classes (e.g., 'WalkingWithDog', 'TaiChi', etc.).
2. Dependencies: Install the necessary libraries using pip:
```
pip install opencv-python pafy numpy tensorflow keras scikit-learn matplotlib
```
3. Training the Models: First, run the script to extract frames and prepare the dataset. Two models will be trained: ConvLSTM and LRCN. The models are saved as `new.h5` and `LRCN.h5` respectively.
4. Testing the Models: After training, both models are evaluated on the test set, and the accuracy is displayed.
5. Running Prediction on a Video: Use the `predictOnVideo` function to predict the class of any given video. Make sure to provide the correct path to the video.
Code Explanation
1. Frame Extraction:
```python
def framesExtract(videoPath):
    ...
```
- This function takes a video path, reads the video, resizes the frames to 64x64, normalizes them (divides pixel values by 255), and returns a list of processed frames.
2. Dataset Creation:
```python
def createDataSet():
    ...
```
- Loops through the selected action classes and videos, extracting frames and labels. These are stored in `features` and `labels`.
3. ConvLSTM Model:
```python
def createConvLstmModel():
    ...
```
- Constructs a ConvLSTM model with several layers of ConvLSTM2D followed by pooling and dropout layers. It flattens the output and applies a softmax layer for classification.
4. LRCN Model:
```python
def createLRCN():
    ...
```
- This model uses Conv2D for spatial feature extraction on each frame, followed by an LSTM layer to capture temporal dependencies across frames.
5. Prediction on Video:
```python
def predictOnVideo(path,sequenceLength):
    ...
```
- Processes a video in real-time, extracts frames, and makes predictions on-the-fly using the trained model.
# Conclusion
This project demonstrates the use of ConvLSTM and LRCN for video classification tasks, applied to the UCF50 action dataset. You can experiment by training these models with different parameters or adding more action classes for enhanced accuracy.

## How to install this project 
```bash
git  clone hassan1324sa/action-recognation-
```
