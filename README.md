# Persian Emotion Detection using ArmanEmo Dataset

This project focuses on emotion detection for Persian language using the ArmanEmo dataset. The project involves preprocessing the data, cleaning it using lemmatization and normalization techniques, fine-tuning transformer models, and saving the fine-tuned models to Hugging Face. Additionally, a second phase focuses on multimodal classification using another dataset.

## Table of Contents
- [Introduction](#introduction)
- [Phase 1](#phase-1)
  - [Preprocessing Data](#preprocessing-data)
  - [Cleaning Data](#cleaning-data)
  - [Fine-tuning ParsBERT](#fine-tuning-parsbert)
  - [Fine-tuning XLM RoBERTa Large](#fine-tuning-xlm-roberta-large)
  - [Saving Model to Hugging Face](#saving-model-to-hugging-face)
  - [Testing the Model](#testing-the-model)
- [Phase 2](#phase-2)
  - [Dataset](#dataset)
  - [Feature Extraction with ResNet](#feature-extraction-with-resnet)
  - [MLP Model for Classification](#mlp-model-for-classification)

## Introduction
The goal of this project is to develop an emotion detection model for the Persian language. Emotion detection is a key tool in natural language processing (NLP) that helps in understanding the emotions expressed in text data. This project utilizes state-of-the-art transformer models to achieve high accuracy in emotion classification.

## Phase 1

### Preprocessing Data
In this step, the raw data from the ArmanEmo dataset is loaded and preprocessed. This involves:
- Tokenization
- Removing unwanted characters
- Handling missing values

### Cleaning Data
Data cleaning is performed using lemmatization and normalization techniques to ensure the text is in a consistent and usable format. The steps include:
- Lemmatization: Reducing words to their base or root form
- Normalization: Standardizing text data to a common format

### Fine-tuning ParsBERT
ParsBERT, a BERT-based model pre-trained on Persian language data, is fine-tuned on the preprocessed and cleaned ArmanEmo dataset. Fine-tuning involves:
- Setting up the model architecture
- Training the model on the dataset
- Evaluating the model performance

### Fine-tuning XLM RoBERTa Large
XLM RoBERTa Large, a multilingual transformer model, is also fine-tuned on the dataset to compare its performance with ParsBERT. The process includes:
- Configuring the model for Persian text
- Training the model
- Evaluating the results

### Saving Model to Hugging Face
After fine-tuning, the models are saved to Hugging Face for easy access and deployment. This step involves:
- Saving model weights and configurations
- Uploading the models to Hugging Face Model Hub

### Testing the Model
The final step in this phase is to test the fine-tuned models to ensure they perform well on unseen data. This includes:
- Setting up a test dataset
- Evaluating model accuracy, precision, recall, and F1 score
- Analyzing the results

## Phase 2

### Dataset
In the second phase, we utilize a different dataset for multimodal classification:
[MVSA Dataset](https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/) which focuses on sentiment analysis on multi-view social data.

### Feature Extraction with ResNet
For the image data in the dataset, we use the ResNet model to extract feature vectors. The steps include:
- Loading the ResNet model pre-trained on ImageNet
- Extracting feature vectors from the images in the dataset

### MLP Model for Classification
An MLP (Multi-Layer Perceptron) model is designed for the classification task. This involves:
- Designing the architecture of the MLP model
- Combining image features and text features
- Training the MLP model on the combined dataset
- Evaluating the model performance
