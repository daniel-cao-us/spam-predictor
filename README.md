# Spam Predictor

Overview
This Python program implements a Naive Bayes classifier for sentiment analysis. It is designed to classify text data (e.g., reviews) into positive or negative sentiments. The implementation uses Laplace smoothing to handle unseen words and allows customization of key parameters such as the smoothing factor and positive prior probability.

Key Features

Customizable Parameters:
Laplace smoothing factor (laplace).
Positive prior probability (pos_prior).
Efficient Training:
Uses Counter from Python's collections module for efficient word frequency computation.
Flexibility:
Options for stemming and case normalization during data loading.
Performance Logging:
Tracks progress during the development set evaluation with tqdm.

Usage
Prepare Dataset:

Structure  data into training and development sets with labels.
Load Data:
train_set, train_labels, dev_set, dev_labels = load_data(
    trainingdir="path/to/train", 
    testdir="path/to/test",
    stemming=False,
    lowercase=True
)
Train and Predict:
predictions = naive_bayes(
    train_set, 
    train_labels, 
    dev_set, 
    laplace=1, 
    pos_prior=0.8
)
To Evaluate Results:
Compare predictions with dev_labels to assess accuracy.
