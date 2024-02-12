# Sentiment-Analysis
As part of this assignment, you will build your own text classifier using the HuggingFace library. By fine-tuning the pre-trained model implemented in HuggingFace model libraries on your dataset, you will replicate the high-performing text classifier and evaluate its performance against the state-of-the-art models on the Papers-with-Code leaderboard.

1.Description of the task and models with references to the original papers and model cards/repository.
The problem statement is to identify the sentiment behind the movie review provided by the audience, Sentiment behind each review can either be positive or negative.
To solve this problem, I am using the pre trained instance of distilbert-base-uncased and fine tuning the model to analyze movie review data.
In the reference paper Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank the author utilizes Sentiment Treebank to provide fine-grained sentiment labels for phrases within parse trees, offering new challenges for sentiment compositionality. The RNTN model, trained on this treebank, outperforms previous methods, achieving an 85.4% accuracy in single sentence positive/negative classification, and 80.7% accuracy in predicting fine-grained sentiment labels. Notably, it effectively captures the impact of contrastive conjunctions and negation across various tree levels for both positive and negative phrases, marking significant progress in sentiment analysis.

2. The kind of hardware you run your model on.
In order to run this model I have used the T4 GPU in google Colab to effectively utilize the parallel processing capabilities of the GPU and reduce the computation time. Other hardware specifications are mentioned below:
 

3. How do you ensure your model has been trained correctly? Do you have a learning curve graph of your training losses from forward propagation? What does it look like?
We can observe a downward trend in the loss function as expected during training, indicating that the model is performing correctly.  We can also observe that the validation accuracy fluctuates initially and then converges after a certain number of epochs indicating that training process has been completed. The training accuracy also follows an increasing pattern and then converges.


4.Evaluation metrics used in your experiment.
In this experiment I have used accuracy as the key metric to determine the performance of my model, this is primarily because accuracy is used as the model benchmark in the leaderboard provided, hence a comparison can be established between the current board and my model.
Evaluation results on test set are as follows:
 
5.Test set performance and comparison with score reported in original paper AND leaderboard. A justification is needed if it differs from the reported scores.
Based on the test set we can observe an accuracy of 89.78%. The original paper reported an accuracy score of 85.4% using Recursive Neural Tensor Network. The highest accuracy obtained in the leaderboard is97.5 using T5-11B model. The increase in accuracy of this model can be explained since distilbert-base-uncased has been selected as the initial pretrained model. The reported score for distilbert in the leader board is 91.3% this is the median accuracy after 5 runs, hence running the created model multiple times should give us an accuracy around 91.3%. The difference in accuracy can be explained due to the fact that the initial weights of the model are randomly assigned.
6.Training and inference time.
Training time 8.898 minutes, for the test data we can observe an inference time of 2.6188s.
7.Hyperparameters used in your experiment (e.g., number of epochs, learning parameter, dropout rate, hidden size of your model) and other details
Epochs:15, Learning rate:2e-5, Hidden size of model: 768, dropout rate:0.1, model type:distilbert, activation funciton: gelu
 
8. Hypothesize what kinds of samples you might think your model would struggle with and report a minimum of ten incorrectly predicted test samples with their ground-truth labels. If you also report the confidence score of the predicted labels (the last Linear layer’s softmax score) on the samples, you will receive a bonus point.
Based on the ten examples we can observe that some of the incorrectly predicted outputs are ambiguous statements without any clear positive or negative sentiment associated with it.
Human annotators also seem to have split decision on these statements.
Some of the incorrectly predicted statements have double negation which might have caused our model to interpret it incorrectly.
 
9.Potential modeling or representation ideas to improve the errors
Combining predictions from multiple sentiment analysis models can help mitigate errors and improve overall performance. Conducting thorough error analysis to understand the types of errors made by the model and iteratively refining the model based on insights gained from the analysis can lead to continuous improvement in sentiment analysis performance.

10.What was the most challenging part of this homework?
I felt the most challenging part of this assignment was finding out 10 mis predicted values and analyzing them for improvements.

End of report

