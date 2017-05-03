#import appropriate libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model

#read in the data
df = pd.read_csv('../../Data/LinReg_Data.csv')
# df.info()

#convert the 'Occ Date' field to a pandas datetime data type
df['Occ Date'] = pd.to_datetime(df['Occ Date'])

#slice the data to get the predictors and convert the 'Year-Week' categorical data to dummies
df_predictors = df[['Year-Week','DAILYAverageDryBulbTemp','Occ Date: Month']]
cols_to_transform = ['Year-Week']
df_with_dummies = pd.get_dummies(df_predictors,columns=cols_to_transform)

#store the predictos(including dummies for 'Year-Week') and response variables into x and y
x = df_with_dummies
y = df['DailyTotalOffenses']

#create the train and test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)

#create and train the linear regression model
linreg = linear_model.LinearRegression()
linreg.fit(x_train,y_train)

#print the R^2 of the linear regression model
print 'Linear Regression Model Output'
print '------------------------------'
print 'R-Square: ' + str(linreg.score(x_test,y_test))