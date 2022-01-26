# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:29:42 2022

@author: Clo
"""


import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv (r'C:/Users/Clo/Documents/MasterIA4OneHealth/Hackathon/data/preprocessed_data.csv')   #read the csv file (put 'r' before the path string to address any special characters in the path, such as '\'). Don't forget to put the file name at the end of the path + ".csv"
#print (data)

X = pd.DataFrame(data, columns= ['gender','age','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status'])
X = StandardScaler().fit_transform(X)
Y_stroke = pd.DataFrame(data, columns= ['stroke'])
Y_heartD = pd.DataFrame(data, columns= ['heart_disease'])
Y_hypertension = pd.DataFrame(data, columns= ['hypertension'])

#Split the datasets
X_train, X_test, Y_stroke_train, Y_stroke_test, Y_HD_train,Y_HD_test, Y_HT_train, Y_HT_test = train_test_split(X, Y_stroke, Y_heartD,Y_hypertension, test_size = 0.2, random_state = 0)

  

#RandomForestClassifier for stroke
from sklearn.ensemble import RandomForestClassifier
model_1_stroke = RandomForestClassifier(n_estimators=10, random_state=0)
model_1_stroke.fit(X_train, np.ravel(Y_stroke_train)).predict_proba(X_test)
print("Accuracy for stroke model is:",model_1_stroke.score(X_test,Y_stroke_test))

model_1_HD = RandomForestClassifier(n_estimators=10, random_state=0)
model_1_HD.fit(X_train, np.ravel(Y_HD_train)).predict_proba(X_test)
print("Accuracy for heart disease model is:",model_1_HD.score(X_test,Y_HD_test))

model_1_HT = RandomForestClassifier(n_estimators=10, random_state=0)
model_1_HT.fit(X_train, np.ravel(Y_HT_train)).predict_proba(X_test)
print("Accuracy for hypertension model is:",model_1_HT.score(X_test,Y_HT_test))


