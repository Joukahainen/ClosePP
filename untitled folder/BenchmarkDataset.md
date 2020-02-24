# ClosePP: Step 1: Use of the "Benchmark dataset"



[Benchmark data home page](https://etsin.avointiede.fi/dataset/urn-nbn-fi-csc-kata20170601153214969115)

[New homepage](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649)

[Link to data](https://avaa.tdata.fi/openida/dl.jsp?pid=urn:nbn:fi:csc-ida-10x201709252015017461612s)



## Description of the "Benchmark dataset"



[Benchmark data home page](https://etsin.avointiede.fi/dataset/urn-nbn-fi-csc-kata20170601153214969115)

[Link to data](https://avaa.tdata.fi/openida/dl.jsp?pid=urn:nbn:fi:csc-ida-10x201709252015017461612s)



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



Ntakaris et al. 2018: 

>  More specifically, the training set is increased by 1 day in each fold and stops after $n − 1$ days (i.e., after 9 days in our case where n = 10). On each fold, the test set corresponds to 1 day of data, which moves in a rolling window format.

> Our labels $l_i^{(j)}$ describe the percentage change of the mid-price.

> where $m_j$ is the future mid-price (k = 1, 2, 3, 5, or 10 next events in our representations) and $m_i$ is the current mid-price. The extracted labels are based on a threshold for the percentage change of 0.002. 
>
> Labels:
>
> - 1: price change >= 20 bps
> - 2: price change -20 to 20 bps
> - 3: price change <= -20 bps



## Models

### Ntakaris et al 2018

- ridge regression (RR)
- SLFN (single layer feedforward network)
  - step 1: determine hidden layer weights
    - randomly (Huang et al 2012)
    - K-means clustering on the training data

##### Results

table of results for 

- each label (1, 2, 3, 5, 10 event horizons)
- normalization
  - unfiltered
  - z-score
  - min-max
  - decimal precision

- accuracy
- precision
- recall
- F1

### Tsantekidis et al 2018

- Convolution
  - Figure 1
  - 5 layers
- LSTM
  - page 10, equations 11-16
  - standard LSTM
- combination of models
  - CNN acts as the feature extractor of the LOB depth time series
  - LSTM is then applied on the time series produced by the CNN
  - output: a label for each time step
- tech: Keras & Tensorflow



### Zhang et al 2018

#### Model

- figure 3, page 5

- convolution (3 layers)

  - layer 1:

    - 1x2@16 (stride = 1,2)
    - 4x1@16 
    - 4x1@16

  - layer 2:

    - 1x10@16 (stride = 1x2)
    - 4x1@16 
    - 4x1@16

  - layer 3:

    - 1x10@16
    - 4x1@16
    - 4x1@16

    

- inception

- LSTM

- 



##### Convolutional layer

> ... it is seen [M. D. Gould, M. A. Porter, S. Williams, M. McDonald, D. J. Fenn, and S. D. Howison, “Limit order books,” Quantitative Finance, vol. 13, no. 11, pp. 1709–1742, 2013] that more than 90% of orders end in cancellation rather than matching, therefore practitioners consider levels further away from best bid and ask levels to be less useful in any LOB. 
>
> In addition, the work of [C. Cao, O. Hansch, and X. Wang, “The information content of an open limit-order book,” Journal of futures markets, vol. 29, no. 1, pp. 16–41, 2009] suggests that the best ask and best bid (L1-Ask and L1-Bid) contribute most to the price discovery and the contribution of all other levels is considerably less, estimated at as little as 20%. As a result, it would be otiose to feed all level information to a neural network as levels deep in a LOB are less useful and can potentially even be misleading. Naturally, we can smooth these signals by _summarising the information_ contained in deeper levels. We note that convolution ﬁlters used in any CNN architecture are discrete convolutions, or ﬁnite impulse response (FIR) ﬁlters, from the viewpoint of signal processing [54]. 
>
> FIR ﬁlters are popular smoothing techniques for denoising target signals and they are simple to implement and work with. 
>
> [more detail on FIR filters on page 5]

##### First convolutional layer

The details of the ﬁrst convolutional layer inevitably need some consideration. As convolutional layers operate a small kernel to “scan” through input data, the layout of limit order book information is vital. Recall that we take the most 100 recent updates of an order book to form a single input and there are 40 features per time stamp, so the size of a single input is (100 × 40). We organise the 40 features as following:

${p_a^{(i)}(t),v_a^{(i)}(t),p_b^{(i)}(t),v_b^{(i)}(t) }_{i=1}^{n=10}$

where $i$ denotes the $i$-th level of a limit order book. The size of our ﬁrst convolutional ﬁlter is $(1 × 2)$ with stride of $(1 × 2)$. The ﬁrst layer essentially summarises information between price and volume ${ p^{(i)} , v^{(i)} }$ at each order book level. The usage of *stride* is necessary here as an important property of convolutional layers is parameter sharing. This property is attractive as less parameters are estimated, largely avoiding overﬁtting problems. However, without strides, we would apply same parameters to { p (i) , v (i) } and { v (i) , p (i+1) } . In other words, p (i) and v (i) would share same parameters because the kernel ﬁlter moves by one step, which is obviously wrong as price and volume form different dynamic behaviors.

Because the ﬁrst layer only captures information at each order book level, we would expect representative features to be extracted when integrating information across multiple order book levels. We can do this by utilising another convolutional layer with ﬁlter size (1 × 2) and stride (1 × 2). The resulting feature maps actually form the micro-price deﬁned by [J. Gatheral and R. C. Oomen, “Zero-intelligence realized variance estimation,” Finance and Stochastics, vol. 14, no. 2, pp. 249–283, 2010.]

_Def microprice: imbalance based weighted mid_

The weight $I$ is called the imbalance. The micro-price is an important indicator as it considers volumes on bid and ask side, and the imbalance between bid and ask size is a very strong indicator of the next price move. This feature of imbalances has been reported by a variety of researchers:

- J. Gatheral and R. C. Oomen, “Zero-intelligence realized variance estimation,” Finance and Stochastics, vol. 14, no. 2, pp. 249–283, 2010. 
- Y. Nevmyvaka, Y. Feng, and M. Kearns, “Reinforcement learning for optimized trade execution,” 
- M. Avellaneda, J. Reed, and S. Stoikov, “Forecasting prices from Level-I quotes in the presence of hidden liquidity,” 
- Y. Burlakov, M. Kamal, and M. Salvadore, “Optimal limit order execution in a simple model for market microstructure dynamics,” 2012. 
- L. Harris, “Maker-taker pricing effects on market quotations,”  
- A. Lipton, U. Pesavento, and M. G. Sotiropoulos, “Trade arrival dynamics and quote imbalance in a limit order book,” 

Unlike the micro-price where only the ﬁrst order book level is considered, we utilise convolutions to form microprices for all levels of a LOB so the resulting features maps are of size (100, 10) after two layers with strides. Finally, we integrate all information by using a large ﬁlter of size (1×10) and the dimension of our feature maps before the Inception Module is (100, 1).

We apply zero padding to every convolutional layer so the time dimension of our inputs does not change and Leaky Rectifying Linear Units (Leaky-ReLU) [61] are used as activation functions. The hyper-parameter (the small gradient when the unit is not active) of the Leaky-ReLU is set to 0.01, evaluated by grid search on the validation set.

Another important property of convolution is that of equivariance to translation [I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. MIT Press, 2016. NB: where is this discussed? cannot find it in the book]. Speciﬁcally, a function $f(x)$ is equivariant to a function $g$ if $f(g(x)) = g(f(x))$. For example, suppose that there exists a main classiﬁcation feature m located at $(x_m , y_m )$ of an image $I(x, y)$. If we shift every pixel of $I$ one unit to the right, we get a new image $I′$ where $I′(x, y) = I(x − 1, y)$. We can still obtain the main classiﬁcation feature $m′$ in $I′$ and $m = m′$ , while the location of $m′$ will be at $(x_{m′} , y_{m′} ) = (x_m −1, y_m )$. This is important to time-series data, because convolution can ﬁnd universal features that are decisive to ﬁnal outputs. In our case, suppose a feature that studies imbalance is obtained at time $t$. If the same event happens later at time $t′$ in the input, the exact feature can be extracted later at $t′$ .

##### Inception module

Figure 4 on page 6.

##### LSTM module

In general, a fully connected layer is used to classify the input data. However, all Maxpool inputs to the fully connected layer are assumed independent of each other unless multiple fully connected layers are used. Due to the usage of Inception Module in our work, we have a large number of features at end. Just using one fully connected layer with 64 units would result in more than 630,000 parameters to be estimated, not to mention multiple layers. In order to capture temporal relationship that exist in the extracted features, we _replace the fully connected layers with LSTM units_. The activation of a LSTM unit is fed back to itself and the memory of past activations is kept with a separate set of weights, so the temporal dynamics of our features can be modelled. We use 64 LSTM units in our work, resulting in about 60,000 parameters, leading to 10 times fewer parameters to be estimated. The last output layer uses a softmax activation function and hence the ﬁnal output elements represent the probability of each price movement class at each time step.



### Experimental results



- objective: minimising the categorical crossentropy loss. 
- ADAM, 
  -  “epsilon” = 1 
  - learning rate = 0.01. 
  - The learning is stopped when validation accuracy does not improve for 20 more epochs. This is about 100 epochs for the FI-2010 dataset and 40 epochs for the LSE dataset. 
  - We train with mini-batches of size 32. 



We choose a small mini-batch size due to the ﬁndings in [66] in which they suggest that large-batch methods tend to converge to narrow deep minima of the training functions, but small-batch methods consistently converge to shallow broad minima. All models are built using Keras [67] based on the TensorFlow backend , and we train them using a single NVIDIA Tesla P100 GPU.



##### Experimental setup

Setup 1 and Setup 2, following N. Passalis, A. Tefas, J. Kanniainen, M. Gabbouj, and A. Iosiﬁdis, 2018, “Temporal bag-of-features learning for predicting mid price movements using high frequency limit order book data,”.

###### Setup 1 

Setup 1 splits the dataset into 9 folds based on a day basis (a standard anchored forward split). In the $i$-th fold, we train our model on the ﬁrst $i$ days and test it on the $(i + 1)$-th day where $i = 1, · · · , 9$. 

###### Setup 2 

The second setting, Setup 2, originates from the works 

- D. T. Tran, A. Iosiﬁdis, J. Kanniainen, and M. Gabbouj, 2018, “Temporal attention-augmented bilinear network for ﬁnancial time-series data analysis,” .

- A. Tsantekidis, N. Passalis, A. Tefas, J. Kanniainen, M. Gabbouj, and A. Iosiﬁdis, 2017, “Forecasting stock prices from the limit order book using convolutional neural networks”.

-  ——, 2018, “Using Deep Learning for price prediction by exploiting stationary limit order book features” .

- ——, 2017, “Using deep learning to detect price change indications in ﬁnancial markets”.

The use of data: _the ﬁrst 7 days_ are used as the train data and _the last 3 days_ are used as the test data in this setup. We evaluate our model in both setups here. 

