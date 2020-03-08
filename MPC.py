import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import re

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df.describe()
train_df.isna().sum()

sb.kdeplot(train_df['battery_power'],legend=False,color="blue",shade=True)
sb.kdeplot(train_df['clock_speed'],legend=False,color="blue",shade=True)
sb.kdeplot(train_df['ram'],legend=False,color="blue",shade=True)

plt.figure(figsize=(12,10))
sb.countplot(x='price_range',data=train_df)

sb.jointplot(x='ram',y='price_range',data=train_df,color='blue',kind='kde')
sb.jointplot(x='int_memory',y='price_range',data=train_df,color='red',kind='kde')
sb.jointplot(x='battery_power',y='price_range',data=train_df,color='red',kind='kde')

plt.figure(figsize=(10,6))
train_df['fc'].hist(alpha=0.5,color='blue',label='Front camera')
train_df['pc'].hist(alpha=0.5,color='red',label='Primary camera')
plt.legend()
plt.xlabel('MegaPixels')

X = train_df.iloc[:,0:22].values
Y = train_df.price_range

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state = 0)

from sklearn.ensemble import RandomForestRegressor
rfReg = RandomForestRegressor(n_estimators = 100 , random_state = 0)
rfReg.fit(X_train,Y_train)
y_pred = rfReg.predict(X_test)
accuracy = rfReg.score(X_test,Y_test)
print(accuracy*100,'%')

import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12, input_dim=21, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history=model.fit(X_train,Y_train, epochs=50, batch_size=50,  verbose=1, validation_split=0.2)

print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

y_pred = model.predict(X_test)