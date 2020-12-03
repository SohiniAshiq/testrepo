# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 22:47:17 2020

@author: moham
"""
#Simple Linear Regression using Pyhton
import pandas as pd

dataset = pd.read_csv('01Students.csv')
df =dataset.copy()

#Split the data
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

#split the dtaset by rows into training ans test datasets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =   \
   train_test_split(X, Y, test_size=0.3, random_state=12)

# Create and train the simple linear regression model
from sklearn.linear_model import LinearRegression
#Create regressor
std_reg = LinearRegression() #Create Regressor
#Train or fit the training data # fit() will calculate value of b0 and b1
std_reg.fit(x_train, y_train)
#you just train your machine learning algorithm on your training data
# lets now predict the values of Y from test data
y_predict = std_reg.predict(x_test)
#Calculate the R-squared and equation of the line
slr_score = std_reg.score(x_test, y_test)
#coeeficient and theintercept
slr_coefficient = std_reg.coef_
slr_intercept = std_reg.intercept_
#equation of the line  #y = 35.27 + 4.823x
# RMSE
from sklearn.metrics import mean_squared_error
import math
slr_rmse = math.sqrt(mean_squared_error(y_test, y_predict))
#Plotting the result
import matplotlib.pyplot as plt
plt.scatter(x_test, y_test)  #scatter plot for the test data

plt.plot(x_test, y_predict) #trendline of the predictions
plt.ylim(ymin=0)
plt.show()



                                  

    




