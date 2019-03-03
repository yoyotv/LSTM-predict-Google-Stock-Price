#Use the information of previous day to predict the next day

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input,LSTM

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv') 
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
training_set = dataset_train.iloc[:, 1:2].values 
real_stock_price = dataset_test.iloc[:, 1:2].values      #groud_truth for test state 
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

#squeeze the feature into 0~1
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)

#arrange the training data and labels
X_train = []  
y_train = []  

for i in range(0, 1258-20-1):  #previous 1 day
  X_train.append(training_set_scaled[i:i+20,0])
  y_train.append(training_set_scaled[i:i+20,0])

X_train, y_train = np.array(X_train), np.array(y_train) 
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))     #previous 1 day as input         
y_train = np.reshape(y_train, (y_train.shape[0],y_train.shape[1],1))

# Initialize the LSTM
input1=Input(shape=(20,1))
lstm=LSTM(50,return_sequences=True)(input1)
dropout=Dropout(0.2)(lstm)
final=Dense(1)(dropout)
model=Model(input=input1,output=final)
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs = 100, batch_size = 32)

#deal with the test input
inputs = dataset_total[1256:-2].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs) 

X_test = []
X_test = np.array(inputs)
X_test = np.reshape(X_test,(1,X_test.shape[0],1))

#predict 
predicted_stock_price = model.predict(X_test)

#transfer from 0~1
predicted_stock_price=np.reshape(predicted_stock_price,(predicted_stock_price.shape[1],1))
predicted_stock_price = sc.inverse_transform(predicted_stock_price)  

#plot result
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price') 
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')  
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

















