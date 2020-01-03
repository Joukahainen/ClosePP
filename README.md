# LSTM-closePP
Estimation of closing price using LSTM

This is a repo for work at home on the closing price prediction project.

Additional code at work.

## Relevant literature

Dixon 2017: "Sequence Classiﬁcation of the Limit Order Book using Recurrent Neural Networks"

Kercheval and Zhang 2013: "Modeling high-frequency limit order book dynamics with support vector machines"

Sirignano & Cont 2019 QF: "Universal features of price formation in ﬁnancial markets: perspectives from deep learning"

Tsantekidis et al. 2018: "Using Deep Learning for price prediction by exploiting stationary limit order book features"

Zhang et al. 2019: "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"





## Step 1: Use of the "Benchmark dataset"



[Benchmark data home page](https://etsin.avointiede.fi/dataset/urn-nbn-fi-csc-kata20170601153214969115)

[Link to data](https://avaa.tdata.fi/openida/dl.jsp?pid=urn:nbn:fi:csc-ida-10x201709252015017461612s)



1. import the dataset, using Tensorflow
2. run a simple LSTM model 
3. compare results with the relevant papers, such as 

### Benchmark dataset structure

Copied from [Link](https://etsin.avointiede.fi/dataset/urn-nbn-fi-csc-kata20170601153214969115)



> Here we provide the normalized datasets as .txt files. The datasets are divided into two main categories: datasets that include the auction period and datasets that do not. For each of these two categories we provide three normalization set-ups based on z-score, min-max, and decimal-precision normalization. Since we followed the anchored cross-validation method for 10 days for 5 stocks, the user can find nine (cross-fold) datasets for each normalization set-up for training and testing. Every training and testing dataset contains information for all the stocks. For example, the first fold contains one-day of training and one-day of testing for all the five stocks. The second fold contains the training dataset for two days and the testing dataset for one day. The two-days information the training dataset has is the training and testing from the first fold and so on.

> The title of the .txt files contains the information in the following order: 
>
> - training or testing set
> - with or without auction period
> - type of the normalization setup
> - fold number (from 1 to 9) based on the above cross-validation method

> ATTENTION: The given files contain both the feature set and the labels. From row 1 to row 144 we provide the features (see 'Benchmark Dataset for Mid-Price Prediction of Limit Order Book Data' for the description) and from row 145 to row 149 we provide labels for 5 classification problems. Labels (row 145 to the end) have the following explanation ‘1’ is for up-movement, ‘2’ is for stationary condition and ‘3’ is for down-movement.







