# Pytorch-RNN

## DESCRIPTION: 
Multivariate time series forecasting with Reccurent Neural Network implemented in Pytorch

## DATA: 

https://www.kaggle.com/c/tabular-playground-series-jul-2021/overview

train.csv - 7111 data points
data point - 11 features (the last 3 are the target variables to be predicted in future)


## PROJECT:

- I wanted to implement RNN on multivariate time series data although it is arguably not the best nor the easiest forecasting method in this case
- The model is functional and seems to be working decently which was my main goal, however I am aware that there is a huge space for improvements especially in predictions of further time steps 
- There are many comments in the code commenting on the dimensions of various tensors, which is the most important thing to keep an eye on
- I tried using <code>GRU</code>, <code>RNN</code> from the module <code>torch.nn</code> without big difference in performance
- In the following section I briefly clarify a few concepts from my code

## FILES:
model.py - RNN class definition
training-testing.py - hyperparameters, datasets, training, testing</br>
train.csv - data used for training and testing</br>
test.csv - unknown data for the Kaggle competition, not used

## CODE CLARIFICATION: 
#### Dataset Constructors: 

<code>Train_Time_Series_Dataset</code> - transforms time series data into supervised learning training data X,y

<code>Eval_Dataset</code> - simply returns the data; used for evaluation and testing 


#### Many to One Architecture RNN

<code>RNN</code> - Many to One Architecture RNN, taking in sequence of data and returning single output </br>

<b>input:</b> fixed length time sequence of time steps (t-sequence_length,t-1) </br>
<b>output:</b> one data point for time step (t)
the data point (t) is then appended to the sequence used for computation of time step (t+1) in <code>RNN.predict()</code>

#### Predict Method

- <code>RNN.predict(known_sequence,future_sequence)</code> method within the model accepts two arguments, first of which is the <code>known_sequence</code> which are "the last datapoints" for which we have all the 11 features; the second argument is <code>future_sequence</code> which contains future datapoints immediatelly following <code>known_sequence</code> where we know only the first 8 features and need to predict the last 3.
- from the <code>known_data</code> we get a sequence <code>x</code> of same <code>sequence_length</code> that was used in training. <coe>x</code> is used to predict single new data point, then updated and used again on the next data point, and so on and so on
- the "oldest" data point of <code>x</code> is thus deleted and new point is added at every time step prediction
- all of this applies only in <code>predict</code> method!


## CREDITS/SOURCES:
- https://www.python-engineer.com/
- https://towardsdatascience.com/tutorial-on-lstm-a-computational-perspective-f3417442c2cd
- https://pytorch.org/

<i>I would be very thankful for any feedback</i>
