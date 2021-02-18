import pandas as pd
from fbprophet import Prophet

# Read CSV file and convert to a Pandas object for ease of use. Can replace with own CSV with right format - as mentioned in README.
df = pd.read_csv('./stocks.csv')

# Initiate Prophet object
m = Prophet()

# Fit the CSV data into the Prophet object for running analysis
m.fit(df)

# Create a new dataframe "future" with a specified "period" in days. If the last date on CSV is 2019-1-1 and 'periods' is 10, prediction
# of numeric-quantity will be performed till 2019-1-10
future = m.make_future_dataframe(periods=365)

# Last command created the dataframe with empty values, the 'predict' command actually fills in predicted values
forecast = m.predict(future)

# ds is the date, yhat is the predicted value, yhat_lower and yhat_upper are the upper and lower limits of the prediction.
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# To export the dataframe to a CSV
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('predicted.csv')