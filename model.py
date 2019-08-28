#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:48:15 2019

@author: chidi
"""
# Imported Libraries
import pandas as pd

# Importing the datasets
train_data = pd.read_csv("../input/financial-inclusion-in-africa/Train_v2 (1).csv")

# Assigning the dependent and Independent variables
X = train_data.drop(columns = ['bank_account', 'uniqueid']) # Independent Varaible
y = train_data['bank_account']                # Dependent Variable

# Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
z = train_data.iloc[:, [6, 7]] # Dummy variable to hold numerical features 
X = X.drop(columns = ['household_size', 'age_of_respondent', 'cellphone_access'])      

# train_data
labelencoder_X = LabelEncoder()
X = X.apply( lambda col: labelencoder_X.fit_transform(col))
onehotencoder_X = OneHotEncoder(drop='first', categories='auto')
X = onehotencoder_X.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Converting X and y back to dataframe 
X = pd.DataFrame(data = X)
y = pd.DataFrame(data = y)

# Joinning the numerical variables back
X = X.join(z)

# Balancing the dataset using ADASYN
from imblearn.over_sampling import ADASYN
sm = ADASYN()
X, y = sm.fit_sample(X, y)

#splitting training test and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

# Importing XGBoost and fitting ,to the train and test set
import xgboost as xgb
xgb_model = xgb.XGBClassifier(booster='gbtree',
                              eta = 0.1,
                              learning_rate=0.03,
                              max_depth=15,
                              n_estimators=100,
                              objective='binary:logistic')
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
print(y_pred)
"""from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators': [100],
              'objective': ['binary:logistic'],
              'learning_rate':[0.03, 0.01],
              'max_depth':[15],
              'colsample_bytree':[0.16, 0.18],
              'booster':['gbtree'],
              'eta': [0.1, 0.3],
              }

model = GridSearchCV(estimator = xgb_model, 
                     param_grid = parameters, 
                     scoring = 'accuracy',
                     n_jobs = -1,
                     cv = 3)
model = model.fit(X, y)
best_accuracy_R = model.best_score_
best_parameters_R = model.best_params_

print(best_accuracy_R)
print(best_parameters_R)"""