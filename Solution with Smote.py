# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 11:40:04 2019

@author: Puneet_Dubey
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 20:57:54 2019

@author: Puneet_Dubey
"""
#import basic packages

import numpy as np
import pandas as pd

#read data

dataset = pd.read_csv('Challenge Dataset_ML.csv')

out_data= dataset

dataset.info()

dataset_dt=dataset

#Total chrarges is type objects..needs conversion

dataset['TotalCharges'] = pd.to_numeric(dataset.TotalCharges, errors='coerce')

dataset.head()

#customer Id has high cardinality, hence wont be of use
dataset.drop('customerID',axis=1 ,inplace = True)

dataset.head()

dataset.describe(include=[np.object])

column_nm=list(dataset.columns)


column_nm.remove('tenure')
column_nm.remove('MonthlyCharges')
column_nm.remove('TotalCharges')


for i in column_nm:
    j=dataset[i].value_counts()
    print('__________________')
    print(j)

dataset.describe(include=[np.number])


#Checking for missing values

dataset.isnull().sum()  

#filling missing values with median

dataset.TotalCharges.fillna(dataset.TotalCharges.median(), inplace= True)


dataset.isnull().sum() 

"""correlation analysis from tableau it is clear that there is a strong correlation within 
Total charges and tenure """

print(dataset[['MonthlyCharges','TotalCharges','tenure']].corr())

dataset.drop(['TotalCharges'],axis=1,inplace= True)

dataset.head()

print(dataset.corr()) #no strong co-relations

#create dummy variables

copy_dataset = dataset

dummy=pd.get_dummies(copy_dataset,drop_first=True)

dummy.head()

len(dummy.columns)

#Building predictive models
X=dummy.iloc[:,0:22]
y=dummy.iloc[:,-1]

X_New = X

#Balancing the dataset

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))



from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train_res=sc.fit_transform(X_train_res)

X_test=sc.transform(X_test)

#Create model

from sklearn.linear_model import LogisticRegression


logreg = LogisticRegression()
logreg.fit(X_train_res, y_train_res)

y_pred = logreg.predict(X_test)

print('Accuracy on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

y_pred1 = logreg.predict(X_train)
print('Accuracy  on train set: {:.2f}'.format(logreg.score(X_train_res, y_train_res)))


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))



y_pred_list=list(y_pred)

y_prob=logreg.predict_proba(X_test)

y_prob_list=list(y_prob)


X_New=sc.fit_transform(X_New)

Y_pred_Final = logreg.predict(X_New)

Y_Pred_Final_List = list(Y_pred_Final)

dataset['Prediction']=Y_Pred_Final_List

'''
dataset_final= dataset



#decision tree

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print('Accuracy on test set: {:.2f}'.format(dt.score(X_test, y_test)))

y_pred1_dt = logreg.predict(X_train)

print('Accuracy  on train set: {:.2f}'.format(logreg.score(X_train, y_train)))

X_New=sc.fit_transform(X_New)

Y_pred_Final_dt= dt.predict(X_New)

Y_Pred_Final_List_dt = list(Y_pred_Final)


dataset_dt['Prediction']=Y_Pred_Final_List_dt'''