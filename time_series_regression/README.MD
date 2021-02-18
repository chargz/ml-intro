# Time Series Regression

[Time series](https://en.wikipedia.org/wiki/Time_series) analysis comprises methods for analyzing time series data in order to extract meaningful statistics and other characteristics of the data. Time series forecasting is the use of a model to predict future values based on previously observed values.

# Requirements

Python 3 is required. Apart from that, a few `pip` packages are needed. The following command will install the required packages:

`pip3 install -U fbprophet pandas`

If `pip` throws errors while installing fbprophet, `conda` can be used to install the packages. [Install conda](https://docs.conda.io/en/latest/miniconda.html) and run `conda install -c conda-forge fbprophet`.

# Details

- The Prophet quickstart can be found [here](https://facebook.github.io/prophet/docs/quick_start.html). Currently, the software can only import 2 columns - the *timeframe* and the *numerical metric* to predict. (For example, a file containing the *timestamp* and *stock price* of Apple).

- The timestamp can be of 2 formats: 
`YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`.

- The stocks data can be found on `stocks.csv`.

- Remaining details can be found as comments on the *time_series_regression.py* file.
