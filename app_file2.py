from flask import Flask, Blueprint, render_template, session,abort
import pickle
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler





#loading LTC model
model3 = pickle.load(open('model3.pkl', 'rb'))


#LTC data prcoessing for input
scaler = MinMaxScaler(feature_range=(0, 1))
crypto_currency_ltc = 'LTC'
against_currency = 'GBP'
prediction_days = 60

start = dt.datetime(2020, 1, 1)
end = dt.datetime.now()

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

data_ltc = web.DataReader(f'{crypto_currency_ltc}-{against_currency}', 'yahoo', start, end)
test_data_ltc = web.DataReader(f'{crypto_currency_ltc}-{against_currency}', 'yahoo', test_start, test_end)

actual_prices_ltc = test_data_ltc['Close'].values

total_dataset_ltc = pd.concat((data_ltc['Close'], test_data_ltc['Close']), axis=0)

model_inputs_ltc = total_dataset_ltc[len(total_dataset_ltc) - len(test_data_ltc) - prediction_days:].values
model_inputs_ltc = model_inputs_ltc.reshape(-1, 1)
model_inputs_ltc = scaler.fit_transform(model_inputs_ltc)

x_test_ltc = []
for b in range(prediction_days, len(model_inputs_ltc)):
    x_test_ltc.append(model_inputs_ltc[b - prediction_days:b, 0])

x_test_ltc = np.array(x_test_ltc)
x_test_ltc = np.reshape(x_test_ltc, (x_test_ltc.shape[0], x_test_ltc.shape[1], 1))

# predicting next day closing price

real_data_ltc = [model_inputs_ltc[len(model_inputs_ltc) - prediction_days:len(model_inputs_ltc) + 1, 0]]
real_data_ltc = np.array(real_data_ltc)
real_data_ltc = np.reshape(real_data_ltc, (real_data_ltc.shape[0], real_data_ltc.shape[1], 1))








app_file2 = Blueprint('app_file2',__name__)
@app_file2.route("/ltc")
def eth():
    prediction3 = model3.predict(real_data_ltc)
    prediction3 = scaler.inverse_transform(prediction3)
    print(prediction3)
    return render_template('index.html', ltc_prediction=prediction3)


