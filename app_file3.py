from flask import Flask, Blueprint, render_template, session,abort
import pickle
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler



#loading BCH model
model4 = pickle.load(open('model4.pkl', 'rb'))



#BCH data prcoessing for input
scaler = MinMaxScaler(feature_range=(0, 1))
crypto_currency_bch = 'BCH'
against_currency = 'GBP'
prediction_days = 60

start = dt.datetime(2020, 1, 1)
end = dt.datetime.now()

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

data_bch = web.DataReader(f'{crypto_currency_bch}-{against_currency}', 'yahoo', start, end)
test_data_bch = web.DataReader(f'{crypto_currency_bch}-{against_currency}', 'yahoo', test_start, test_end)

actual_prices_bch = test_data_bch['Close'].values

total_dataset_bch = pd.concat((data_bch['Close'], test_data_bch['Close']), axis=0)

model_inputs_bch = total_dataset_bch[len(total_dataset_bch) - len(test_data_bch) - prediction_days:].values
model_inputs_bch = model_inputs_bch.reshape(-1, 1)
model_inputs_bch = scaler.fit_transform(model_inputs_bch)

x_test_bch = []
for c in range(prediction_days, len(model_inputs_bch)):
    x_test_bch.append(model_inputs_bch[c - prediction_days:c, 0])

x_test_bch = np.array(x_test_bch)
x_test_bch = np.reshape(x_test_bch, (x_test_bch.shape[0], x_test_bch.shape[1], 1))

# predicting next day closing price

real_data_bch = [model_inputs_bch[len(model_inputs_bch) - prediction_days:len(model_inputs_bch) + 1, 0]]
real_data_bch = np.array(real_data_bch)
real_data_bch = np.reshape(real_data_bch, (real_data_bch.shape[0], real_data_bch.shape[1], 1))





app_file3 = Blueprint('app_file3',__name__)
@app_file3.route("/bch")
def bch():
    prediction4 = model4.predict(real_data_bch)
    prediction4 = scaler.inverse_transform(prediction4)
    print(prediction4)
    return render_template('index.html', bch_prediction=prediction4)