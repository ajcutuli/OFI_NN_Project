# A Deep Learning Approach to Limit Order Book Forecasting
## Abstract
In this [notebook](https://github.com/ajcutuli/OFI_NN_Project/blob/main/DeepOFI.ipynb), we implement an artificial neural network originally employed by Zhang et al[1] that combines convolutional neural networks (CNNs) and a long short-term memory (LSTM) neural network in order to classify future directions of an order book at a high frequency. Specifically, given Coinbase order book data for Bitcoin, we seek to predict whether the mid price increases, decreases, or does not change in the next observation of the time series. Unlike Zhang et al's papers we reference, which use non-stationary order book states as inputs to the network, our instantiation of the architecture is trained on order flow and order flow imbalance, which are stationary quantities derived from the limit order book. Hence, this discussion also draws heavy inspiration from a 2021 article by Kolm et al[4], which demonstrated that forecasting using order flow significantly outperforms raw order book inputs. Today, we further this discussion by doing an analysis of the impact that differencing order flow into order flow imbalance has on the forecasting performance of the model. We also approach the problem from a time series modeling perspective by analyzing the time-lag of the processes, a procedure that was not explicitly documented in those papers. We find that for our sparse data, the models are similarly poor in their ability to predict downward moves, but the model trained on order flow significantly outperforms the other when it comes to predicting upward moves. In the trading scenario we posit, we see that this failure to precisely predict downward moves is troublesome for intraday trading in a bearish market.
