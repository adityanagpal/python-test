#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 09:29:01 2018

@author: aditya
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date
from nsepy import get_history

dataset = get_history(symbol="NIFTY", 
                    start=date(2015,1,1), 
                    end=date(2015,12,31),
					index=True)

from pandas import Series
from pandas import DataFrame

dataframe = DataFrame()
dataframe['Turnover'] = [nifty.index[i].month for i in range(len(nifty))]


X=dataset.iloc[:,[0,1,2,4,5]].values
y=dataset.iloc[:,3].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

import statsmodels.formula.api as sm
X=np.append(arr=X,values=np.ones((248,1)).astype(int),axis=1)
#significance level = 0.05
Xopt=X[:,[0,1,2,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=Xopt).fit()
regressor_OLS.summary()

Xopt=X[:,[1,2,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=Xopt).fit()
regressor_OLS.summary()

Xopt=X[:,[2,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=Xopt).fit()
regressor_OLS.summary()


Xopt=X[:,[4,5]]
regressor_OLS=sm.OLS(endog=y,exog=Xopt).fit()
regressor_OLS.summary()
#final select




dataset[['Close', 'Turnover']].plot(secondary_y='Turnover')