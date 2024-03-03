# Cinematic Sentiment Analysis

## Description
This project involves building a text classifier for sentiment analysis using the HuggingFace library. The goal is to fine-tune a pre-trained model, specifically the distilbert-base-uncased, on a dataset of movie reviews to classify sentiments as positive or negative. The approach is inspired by the Recursive Neural Tensor Network (RNTN) model, as detailed in the paper "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank."

## Hardware
The model is trained on a T4 GPU in Google Colab to leverage parallel processing capabilities, reducing computation time.

## Model Training Validation
The training process involves observing a decreasing trend in the loss function during training and convergence of validation accuracy after a certain number of epochs. Training accuracy shows an increasing pattern followed by convergence.

## Evaluation Metrics
The key metric for model performance is accuracy, consistent with the leaderboard benchmark. Test set results indicate an accuracy of 89.78%.

## Comparison with Original Paper and Leaderboard
The model's accuracy (89.78%) surpasses the original paper's reported accuracy (85.4%). The leaderboard's highest accuracy (97.5%) is achieved by the T5-11B model. The choice of distilbert-base-uncased as the initial model and the random initialization of weights contribute to the variance in accuracy.

## Training and Inference Time
Training time is 8.898 minutes, and the test data inference time is 2.6188 seconds.

## Hyperparameters
- Epochs: 15
- Learning rate: 2e-5
- Hidden size of the model: 768
- Dropout rate: 0.1
- Model type: distilbert
- Activation function: gelu

## Model Challenges
The model may struggle with ambiguous statements, double negations, and instances where human annotators disagree on sentiment. Ten incorrectly predicted test samples are discussed, highlighting the challenges faced.

## Potential Improvements
Combining predictions from multiple sentiment analysis models and conducting thorough error analysis can improve overall performance.

## Challenging Aspect
The most challenging part of this project was identifying and analyzing ten mispredicted values to suggest improvements.
