from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from flask import Flask
from app_file1 import app_file1
from app_file2 import app_file2
from app_file3 import app_file3
from app_file4 import app_file4





#loading BTC model
model1 = pickle.load(open('model1.pkl', 'rb'))


#create flask app
app = Flask(__name__)
app.register_blueprint(app_file1)
app.register_blueprint(app_file2)
app.register_blueprint(app_file3)
app.register_blueprint(app_file4)




#BTC data prcoessing for input
scaler = MinMaxScaler(feature_range=(0, 1))
crypto_currency_btc = 'BTC'
against_currency = 'GBP'
prediction_days = 60

start = dt.datetime(2020, 1, 1)
end = dt.datetime.now()

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

data_btc = web.DataReader(f'{crypto_currency_btc}-{against_currency}', 'yahoo', start, end)
test_data_btc = web.DataReader(f'{crypto_currency_btc}-{against_currency}', 'yahoo', test_start, test_end)

actual_prices_btc = test_data_btc['Close'].values

total_dataset_btc = pd.concat((data_btc['Close'], test_data_btc['Close']), axis=0)

model_inputs_btc = total_dataset_btc[len(total_dataset_btc) - len(test_data_btc) - prediction_days:].values
model_inputs_btc = model_inputs_btc.reshape(-1, 1)
model_inputs_btc = scaler.fit_transform(model_inputs_btc)

x_test_btc = []
for x in range(prediction_days, len(model_inputs_btc)):
    x_test_btc.append(model_inputs_btc[x - prediction_days:x, 0])

x_test_btc = np.array(x_test_btc)
x_test_btc = np.reshape(x_test_btc, (x_test_btc.shape[0], x_test_btc.shape[1], 1))

# predicting next day closing price

real_data_btc = [model_inputs_btc[len(model_inputs_btc) - prediction_days:len(model_inputs_btc) + 1, 0]]
real_data_btc = np.array(real_data_btc)
real_data_btc = np.reshape(real_data_btc, (real_data_btc.shape[0], real_data_btc.shape[1], 1))

















@app.route('/')
def btc():
    prediction1 = model1.predict(real_data_btc)
    prediction1 = scaler.inverse_transform(prediction1)
    print(prediction1)
    return render_template('index.html', btc_prediction=prediction1)


if __name__ == '__main__':
    app.run(debug=True)



