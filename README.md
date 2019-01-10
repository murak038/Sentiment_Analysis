# Sentiment_Analysis
Comparative Analysis of Different Architectures for Sentiment Analysis

## Project Overview
Sentiment analysis is a very simple yet powerful natural language processing (NLP) technique that is used to classify the sentiments of the text into different classes based on the emotions underlying the text. In the simplest case, the text can be classified as being positive or negative. However, the method can extend to more complex classes such as positive, slightly positive, neutral, slightly negative and negative or can even be applied to different classes of sentiment such as toxicity. 

Additionally, there are many different architectures that can be used to carry out the sentiment analysis task. The most basic architecture is an RNN model that takes the embedded sequence as an input. The result of the output of the RNN model is then passed through a dense layer to obtain a classification. There has been work done using more complex architectures such as bidirectional models, multi-layer models and even convolutional neural network. A recent work done by Salesforce Research has proposed a new type of RNN model that combines convolutional layers and recurrent pooling steps and applied it to the task of sentiment analysis. 

This presents the question of which architecture performs better both in terms of time and accuracy. 

## Data
The data contained in 'reviews.txt' and 'labels.txt' are a subset of reviews and sentiment of IMDB reviews. The text is classified as being 'POSITIVE' or 'NEGATIVE'. 

## Architectures
In this experiment, which participates as an extension of a sentiment analysis assignment, there are six different architectures tested:
1. Quasi Recurrent Neural Network (QRNN)
2. Unidirectional Gated Recurrent Unit (GRU)
3. Unidirectional Long-Short Term Unit (UniLSTM)
4. Bidirectional LSTM (BiLSTM)
5. Multi-Layer Bidirectional LSTM (MBiLSTM)
6. Convolutional Neural Network (CNN)

### Install

This project requires **Python 3.6** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [PyTorch 1.0.0](https://pytorch.org/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 3.6 installer. 

## Run Code
In a terminal or command window, navigate to the top-level project directory `Sentiment_Analysis/` (that contains this README) and run the following command:

```bash
jupyter notebook sentiment_analysis.ipynb
```

This will open the iPython Notebook software and project file in your browser.

## Results

| Architecture        | Test Accuracy - 5 Epochs           | Test Accuracy - 10 Epochs  | Time to Train Epoch |
| ------------- |:-------------:| :-------------:| :-------------:|
| QRNN      | 81.880 | 81.920 |  34.475 s  |
| GRU      | 75.960      |  78.960 |  18.935 s  |
| UniLSTM | 81.400     |  80.080 |  20.237 s  |
| BiLSTM | 78.840     |  81.120 |  30.113 s  |
| MBiLSTM | 80.800     |  81.040 |  57.802 s  |
| CNN | 72.880     |  75.560 |  18.754 s  |

## Summary
From the results seen above, the performance of the QRNN model was comparable to that of the multi-layer bidirectional LSTM while the training time is decreased by 40%. Additionally, while the GRU model and the CNN model took less time to train per epoch, the model requires more epochs for the model to achieve maximal performance.  
