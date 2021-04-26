import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_absolute_error,mean_squared_error

data=pd.read_csv("houseprices_dataset.csv")
print(data.head())
print(data.shape)
print(data.describe)


#data cleaning

print(data.isnull().sum())

#traning and testing of the data
X=data.drop("House_price_inlakhs",axis=1)
Y=data["House_price_inlakhs"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=123)
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

y_pred=regressor.predict(X_test)

#prediction
mse=mean_squared_error(y_pred,Y_test)
mae=mean_absolute_error(y_pred,Y_test)
print(mse,mae)
import math
rmse = math.sqrt(mse)
print(rmse)

#serialization and deserialization
pickle.dump(regressor,open("model.pkl",'wb'))
model=pickle.load(open("model.pkl","rb"))
print(model.predict([[23,54,12788]]))
