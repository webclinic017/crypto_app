from flask import Flask, Blueprint, render_template, session,abort
import pickle
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler


#loading LTC model
model5 = pickle.load(open('model5.pkl', 'rb'))




#ADA data prcoessing for input
scaler = MinMaxScaler(feature_range=(0, 1))
crypto_currency_ada = 'ADA'
against_currency = 'GBP'
prediction_days = 60

start = dt.datetime(2020, 1, 1)
end = dt.datetime.now()

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

data_ada = web.DataReader(f'{crypto_currency_ada}-{against_currency}', 'yahoo', start, end)
test_data_ada = web.DataReader(f'{crypto_currency_ada}-{against_currency}', 'yahoo', test_start, test_end)

actual_prices_ada = test_data_ada['Close'].values

total_dataset_ada = pd.concat((data_ada['Close'], test_data_ada['Close']), axis=0)

model_inputs_ada = total_dataset_ada[len(total_dataset_ada) - len(test_data_ada) - prediction_days:].values
model_inputs_ada = model_inputs_ada.reshape(-1, 1)
model_inputs_ada = scaler.fit_transform(model_inputs_ada)

x_test_ada = []
for d in range(prediction_days, len(model_inputs_ada)):
    x_test_ada.append(model_inputs_ada[d - prediction_days:d, 0])

x_test_ada = np.array(x_test_ada)
x_test_ada = np.reshape(x_test_ada, (x_test_ada.shape[0], x_test_ada.shape[1], 1))

# predicting next day closing price

real_data_ada = [model_inputs_ada[len(model_inputs_ada) - prediction_days:len(model_inputs_ada) + 1, 0]]
real_data_ada = np.array(real_data_ada)
real_data_ada = np.reshape(real_data_ada, (real_data_ada.shape[0], real_data_ada.shape[1], 1))










app_file4 = Blueprint('app_file4',__name__)
@app_file4.route("/ada")
def ada():
    prediction5 = model5.predict(real_data_ada)
    prediction5 = scaler.inverse_transform(prediction5)
    print(prediction5)
    return render_template('index.html', ada_prediction=prediction5)