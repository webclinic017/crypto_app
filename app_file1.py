from flask import Flask, Blueprint, render_template, session,abort
import pickle
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler



#loading ETH model
model2 = pickle.load(open('model2.pkl', 'rb'))


#ETH data prcoessing for input
scaler = MinMaxScaler(feature_range=(0, 1))
crypto_currency_eth = 'ETH'
against_currency = 'GBP'
prediction_days = 60

start = dt.datetime(2020, 1, 1)
end = dt.datetime.now()

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

data_eth = web.DataReader(f'{crypto_currency_eth}-{against_currency}', 'yahoo', start, end)
test_data_eth = web.DataReader(f'{crypto_currency_eth}-{against_currency}', 'yahoo', test_start, test_end)

actual_prices_eth = test_data_eth['Close'].values

total_dataset_eth = pd.concat((data_eth['Close'], test_data_eth['Close']), axis=0)

model_inputs_eth = total_dataset_eth[len(total_dataset_eth) - len(test_data_eth) - prediction_days:].values
model_inputs_eth = model_inputs_eth.reshape(-1, 1)
model_inputs_eth = scaler.fit_transform(model_inputs_eth)

x_test_eth = []
for a in range(prediction_days, len(model_inputs_eth)):
    x_test_eth.append(model_inputs_eth[a - prediction_days:a, 0])

x_test_eth = np.array(x_test_eth)
x_test_eth = np.reshape(x_test_eth, (x_test_eth.shape[0], x_test_eth.shape[1], 1))

# predicting next day closing price

real_data_eth = [model_inputs_eth[len(model_inputs_eth) - prediction_days:len(model_inputs_eth) + 1, 0]]
real_data_eth = np.array(real_data_eth)
real_data_eth = np.reshape(real_data_eth, (real_data_eth.shape[0], real_data_eth.shape[1], 1))




app_file1 = Blueprint('app_file1',__name__)
@app_file1.route("/eth")
def eth():
    prediction2 = model2.predict(real_data_eth)
    prediction2 = scaler.inverse_transform(prediction2)
    print(prediction2)
    return render_template('index.html', eth_prediction=prediction2)

