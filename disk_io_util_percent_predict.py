#%% md

# Stock Prices Prediction Using Keras Long Term Short Memory

#%%

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

from IPython import embed
#%%

# Importing the training set
dataset_train = pd.read_csv('./data/machine_usage.csv',error_bad_lines=False, sep ="\t")
training_set = dataset_train.iloc[0:3325, 8:9].values / 100
# training_set.shape
# training_set

#%%

dataset_train.head()

#%%

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#%%

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(120, 3320):
    X_train.append(training_set_scaled[i-120:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#%%

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#%%

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#%%

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#%%

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#%%


# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#%%

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


#%%

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 87, batch_size = 16)



#%%

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('./data/machine_usage.csv',error_bad_lines=False, sep ="\t")
# dataset_test = pd.read_csv('data/test.csv', error_bad_lines=False, sep ="\t")
real_stock_price = dataset_test.iloc[0:201, 8:9].values / 100
# real_stock_price.shape
# dataset_test.head()

#%%

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['disk_io_percent'], dataset_test['disk_io_percent']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 120:].values / 100
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(120, 320):
    X_test.append(inputs[i-120:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#%%
# embed()
# 计算回归评价指标
print('MSE Value= ', mean_squared_error(real_stock_price[:-1], predicted_stock_price))
print('MAE Value= ', mean_absolute_error(real_stock_price[:-1], predicted_stock_price))
print('RMSE Value= ', sqrt(mean_squared_error(real_stock_price[:-1], predicted_stock_price)))
print('MAPE Value= ', mean_absolute_percentage_error(real_stock_price[:-1], predicted_stock_price))

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'real disk util percent')
plt.plot(predicted_stock_price, color = 'green', label = 'predicted disk util percent')
plt.title('disk util percent prediction')
plt.xlabel('Time')
plt.ylabel('disk util percent')
plt.legend()
plt.show()

#%%