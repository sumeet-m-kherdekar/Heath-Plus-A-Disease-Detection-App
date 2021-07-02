# Python code to create ML-models for all the diseases

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Creating a model for Diabetes----------------------------------------------------------------------
df = pd.read_csv('F:/Coding/Python_program/Project2/Dataset/diabetes.csv')
df = df.dropna()
x= df.iloc[:,0:-1].values
y= df.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state = 0)

classifier = RandomForestClassifier()
classifier.fit(x_train,y_train)
prediction = classifier.predict(x_test)
print('Diabetes = ',accuracy_score(y_test,prediction))
# save the model to disk
filename = 'F:/Coding/Python_program/Project2/Diabetes_model.h5'
joblib.dump(classifier, filename)

# load the model from disk
# loaded_model = joblib.load(filename)
# result = loaded_model.score(x_test, y_test)
# print(result)

# Creating a model for Heart Disease------------------------------------------------------------------
df = pd.read_csv('F:\Coding\Python_program\Project2\Dataset\heart.csv')
df = df.dropna()
x= df.iloc[:,0:-1].values
y= df.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state = 0)
classifier = RandomForestClassifier()
classifier.fit(x_train,y_train)
prediction = classifier.predict(x_test)
print('Heart = ',accuracy_score(y_test,prediction))
filename = 'F:/Coding/Python_program/Project2/Heart_model.h5'
joblib.dump(classifier, filename)

# Creating a model for Kidney Disease----------------------------------------------------------
df = pd.read_csv('F:\Coding\Python_program\Project2\Dataset\kidney_disease.csv')
df = df.dropna()
df.replace(to_replace='normal', value=0 ,inplace=True)
df.replace(to_replace='abnormal', value=1,inplace=True)
df.replace(to_replace='notpresent', value=0,inplace=True)
df.replace(to_replace='present', value=1,inplace=True)
df.replace(to_replace='yes', value=1,inplace=True)
df.replace(to_replace='no', value=0,inplace=True)
df.replace(to_replace='good', value=1,inplace=True)
df.replace(to_replace='poor', value=0,inplace=True)
df.replace(to_replace='ckd', value=1,inplace=True)
df.replace(to_replace='notckd', value=0,inplace=True)

x= df.iloc[:,1:10].values
y= df.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state = 0)

classifier = RandomForestClassifier()
classifier.fit(x_train,y_train)
prediction = classifier.predict(x_test)
print('Kidney = ',accuracy_score(y_test,prediction))
filename = 'F:/Coding/Python_program/Project2/Kidney_model.h5'
joblib.dump(classifier, filename)

# # Creating a model for liver Disease----------------------------------------------------------

df = pd.read_csv('F:\Coding\Python_program\Project2\Dataset\indian_liver_patient.csv')
df = df.dropna()
df.replace(to_replace='Male', value=1 ,inplace=True)
df.replace(to_replace='Female', value=0 ,inplace=True)
x= df.iloc[:,0:-1].values
y= df.iloc[:,-1].values ## 2 means no-disease
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1, random_state = 0)

classifier = RandomForestClassifier()
classifier.fit(x_train,y_train)
prediction = classifier.predict(x_test)
print('Liver = ', accuracy_score(y_test,prediction))
filename = 'F:/Coding/Python_program/Project2/Liver_model.h5'
joblib.dump(classifier, filename)