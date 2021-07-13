# -*- coding: utf-8 -*-
"""
Created on Tue May 11 12:01:03 2021

@author: onero
"""

##############################################################
# importing libraries
##############################################################


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

##############################################################
# Loading dataset
##############################################################

print('\n Reading....')
df=pd.read_excel('data.xlsx')
print('\n Done !')

pd.set_option('display.max_columns',None) ## display all columns 

##############################################################
# Checking null values and zero
##############################################################

df.head()


df.info()

## checking null values
df.isnull().sum()

## drop na values
df.dropna(inplace=True)
df.isnull().sum()


df['Duration'].value_counts() ## duration contain string and model cannot understand such data 2h 50m and all


##############################################################
# Exploring the data
##############################################################
df['Duration'].value_counts() ## duration contain string and model cannot understand such data 2h 50m and all

## from data we can see that Date_of_journey is object data type
## so we have to convert this into timestamp so as to use this column properly for prediction
## for this we require pandas (pd.to_datetime) to convert object into datetime dtype
##  .dt.day - extract only day of that date
## .dt.month - extract only month of that date



df['journey_day']=pd.to_datetime(df['Date_of_Journey'],format='%d/%m/%Y').dt.day  ## extracting the day since year is same

df['journey_month']=pd.to_datetime(df['Date_of_Journey'],format='%d/%m/%Y').dt.month ## extracting the month since year is sam


df.head()

## dropping Date_of_Journey because we already extracted the day and month

df.drop(['Date_of_Journey'],axis=1,inplace=True)

##############################################################
# Extraction of hour and minute form Dep_time
##############################################################

## extracting hours
df['dep_hours']=pd.to_datetime(df['Dep_Time']).dt.hour

## extracting minute

df['dep_min']=pd.to_datetime(df['Dep_Time']).dt.minute

## dropping the Dep_time

df.drop(['Dep_Time'],axis=1,inplace=True)

df.head()

##############################################################
# Extraction of hour and minute form 
##############################################################

## extracting hours

df['arr_hour']=pd.to_datetime(df['Arrival_Time']).dt.hour
df['arr_min']=pd.to_datetime(df['Arrival_Time']).dt.minute

## dropping the Arrival_Time
df1=df.drop(['Arrival_Time'],axis=1,inplace=False)

df1.head()

##############################################################
# Duration
##############################################################

# time taken by plane to reach destination is called duration
# it is the difference between departure time and arrival time


# assigning and converting duration column into list

duration=list(df1['Duration'])

for i in range(len(duration)):
    if len(duration[i].split()) !=2:
        if 'h' in duration[i]:
            duration[i]=duration[i].strip() +" "+ '0m'
        else:
            duration[i]="0h" +" " + duration[i]


'2h 50m'.split()

print(len('2h 50m'.split()))



duration_hour=[]
duration_min=[]

for i in range(len(duration)):
    duration_hour.append(int(duration[i].split(sep= 'h')[0]))
    duration_min.append(int(duration[i].split(sep= 'm')[0].split()[-1]))
    
    
df1['Duration_hour']=duration_hour    
df1['Duration_min']=duration_min    
    
    
df1.head()    
    

## dropping the duration
df2=df1.drop(['Duration'],axis=1,inplace=False)
df2.head()


##############################################################
# Handling categorical data
##############################################################

# nomianl data - data is not is order - one hot encoder used

# ordinal data -  data are in order - label encoder used

df2['Airline'].value_counts()


## airways vs price

sns.catplot(y='Price',x='Airline',data=df2.sort_values('Price',ascending=False),kind='boxen',height=5,aspect=2.5)


## as airline is nominal data we use one hot encoding

Airlines=df2['Airline']
Airlines=pd.get_dummies(Airlines,drop_first=True)

# pd.get_dummies - converts categorical into dummies variable
# drop_first - after converting that into dummies we drop first column

Airlines.head()




## source is also a nominal data we will perform one hot encoding

df2['Source'].value_counts()
sns.catplot(y='Price',x='Source',data=df2.sort_values('Price',ascending=False),kind='boxen',height=5,aspect=2.5)


Sources=df2['Source']
Sources=pd.get_dummies(Sources,drop_first=True)

Sources.head()



## for destination also 



df2['Destination'].value_counts()
sns.catplot(y='Price',x='Destination',data=df2.sort_values('Price',ascending=False),kind='boxen',height=5,aspect=2.5)


Destinations=df2['Destination']
Destinations=pd.get_dummies(Destinations,drop_first=True)

Destinations.head()


## route

df2['Route']
df2['Additional_Info'].value_counts()

## additional info contain 80% of no_info
## route and total_stop are related to each other
## we can drop the additional_info and route

df3=df2.drop(['Route','Additional_Info'],axis=1,inplace=False)
df3.head()

## Total_stop 

df3['Total_Stops'].value_counts()

## Total_stop is of ordinal data type we use label encoder
# here values are assign with corresponding keys

df4=df3.replace({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4},inplace=False)

df4.head()


## concatenate the dataframe -- df3 + Airline + source + destinatio

df5=pd.concat([df4,Airlines,Sources,Destinations],axis=1)

df5.head()

# dropping airline , source and destination as we have already converted that

df6=df5.drop(['Airline','Source','Destination'],axis=1,inplace=False)

df6.head()

df6.shape


##############################################################
# feature selections
##############################################################

## it is based on types of algo we use

# 1) heatmap
# 2) feature_importance
# 3) selectKbase



df6.shape

df6.columns


# independet features
X=df6.loc[:,['Total_Stops', 'journey_day', 'journey_month', 'dep_hours', 'dep_min', 'arr_hour', 'arr_min', 'Duration_hour', 'Duration_min','Air India','Go FIRST', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business','Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet', 'Trujet', 'Vistara', 'Vistara Premium economy','Chennai', 'Kolkata', 'Mumbai', 'Cochin', 'Delhi', 'Hyderabad','Lucknow','New Delhi']]
       
X.head()      
        
      
## dependent features

Y=df6.iloc[:,1]
Y.head()


df7=df6.iloc[:,0:10]

df7.isnull().sum()


df7.corr()
## finding correlation between dependent and indepedent

plt.figure(figsize=(18,18))
sns.heatmap(df7.corr(),annot=True,cmap="RdYlGn")
plt.show()


## important feature using ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesRegressor

selection=ExtraTreesRegressor()
selection.fit(X,Y)


# # plotting graph of feature imp 

plt.figure(figsize=(18,16))
feat_imp=pd.Series(selection.feature_importances_,index=X.columns)
feat_imp.nlargest(20).plot(kind='bar')



##############################################################
# model fitting
##############################################################

## Fitting model using random forest regression because the output is continuos 
# 1) splitting the data into train and test
# 2) import model
# 3) fit the model
# 4) predict 
# 5) check RMSE score
# 6) plot graph

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=313)



from sklearn.ensemble import RandomForestRegressor

RF=RandomForestRegressor()
RF.fit(X_train,Y_train)


Y_pred=RF.predict(X_test)


RF.score(X_train,Y_train)



RF.score(X_test,Y_test)


## distribution of error

sns.distplot(Y_test-Y_pred)
plt.show()

sns.kdeplot(Y_test-Y_pred,shade=True)

## scatter plot of y_test and y_pred
plt.scatter(Y_test, Y_pred)
plt.xlabel('Y_test')
plt.ylabel('Y_pred')
plt.show()



##############################################################
# metrics
##############################################################
# metrics

from sklearn import metrics

print('MAE:',metrics.mean_absolute_error(Y_test, Y_pred))
print('MSE:',metrics.mean_squared_error(Y_test, Y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))



metrics.r2_score(Y_test, Y_pred)

##############################################################
# Hyper parameter tuning
##############################################################
from sklearn.model_selection import RandomizedSearchCV

#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}



# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
RF_random = RandomizedSearchCV(estimator =RF, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

RF_random.fit(X_train,Y_train)


RF_random.best_params_

prediction=RF_random.predict(X_test)



print('MAE:', metrics.mean_absolute_error(Y_test, prediction))
print('MSE:', metrics.mean_squared_error(Y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, prediction)))


##############################################################
# save the model to reuse it again
##############################################################

import pickle
# open a file where you want to store the data
file=open('Flight.pkl','wb')

# dump information to that file 
pickle.dump(RF_random,file)


model=open('Flight.pkl','rb')
forest=pickle.load(model)

y_prediction=forest.predict(X_test)
metrics.r2_score(Y_test,y_prediction)


     
        
        




