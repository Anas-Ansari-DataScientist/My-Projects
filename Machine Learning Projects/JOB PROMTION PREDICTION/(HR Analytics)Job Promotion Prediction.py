#!/usr/bin/env python
# coding: utf-8

# # Job Promotion Prediction

# Importing Necessary libraries  

# In[1]:


import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt


# 
# # Importing the dataset 

# In[428]:


df=pd.read_csv('HR Analytics train data.csv')
df.head()


# # Understanding the data

# We have 54808 records with 14 different features.

# In[6]:


df.shape


# In[7]:


df.dtypes


# In[8]:


df.describe()


# Cheking if there is any nan values in our dataset.

# In[9]:


df.isna().sum()


# As we see some of the data is missing in education and previous_year_rating column, So we will replace the nan values witg mode of the column.

# In[10]:


df['education'].value_counts()


# In[11]:


df['previous_year_rating'].value_counts()


# In[301]:


df.head()


# # EDA

# In[299]:


# Let do some plotting
#Checking for the number of department in Department column
department_count = df["department"].value_counts()
plt.figure(figsize=(14,5))
plt.bar(department_count.index, department_count.values)
plt.xlabel("Departments")
plt.ylabel("Frequency")
plt.title("Department")
plt.show()


# In[308]:


# acoording the data the company has more numbers of bachelor's as compared to others
department_count = df["education"].value_counts()
plt.figure(figsize=(14,5))
plt.barh(department_count.index, department_count.values)
plt.xlabel("Education")
plt.ylabel("Frequency")
plt.title("Education")
plt.show()


# In[316]:


#The highest number of promotion are from Sales and marketing department.
depart_count = df.groupby(["department", "is_promoted"])["is_promoted"].count()
depart_count.unstack().plot(kind='bar', stacked=True)


# In[313]:


#The highest number of promotion are from Males as compared to females.
depart_count = df.groupby(["gender", "is_promoted"])["is_promoted"].count()
depart_count.unstack().plot(kind='bar', stacked=True)


# In[322]:


depart_count = df.groupby(["department", "education"])["department"].count()
depart_count.unstack().plot(kind='barh', stacked=True)


# In[323]:


#The dataset Contains imbalancing in the dependent variables.
[sns.countplot(training['is_promoted'])]


# # Data Manipulation

# In[429]:


#removing nan values with  the most repeated values
df['education'] = df['education'].fillna("Bachelor's")
df['previous_year_rating'] = df['previous_year_rating'].fillna('4.0')


# Converting categorical to numerical variable using one hot encoding

# In[430]:


Department = pd.get_dummies(df['department'])
Education = pd.get_dummies(df['education'])
Gender = pd.get_dummies(df['gender'],drop_first=True)
Recruitment_Channel = pd.get_dummies(df['recruitment_channel'])  
Region = pd.get_dummies(df['region'])


# In[431]:


df.drop(['department','education','gender','recruitment_channel','region'],axis=1,inplace=True)


# In[432]:


training = pd.concat([Department,Education,Gender,Recruitment_Channel,Region, df],axis=1)
training.head(3)


# In[327]:


corrmat = training.corr() 
  
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu",linewidths = 0.1) 


# In[328]:


#Selecting highly correlated features
cor_target = abs(corrmat["is_promoted"])
relevant_features = cor_target[cor_target>=0.0001]
relevant_features.sort_values(ascending=False)


# Dropping columns having correlation less than 0.001

# In[433]:


training.drop(['region_1', 'region_10', 'region_11', 'region_12','region_13', 'region_14', 'region_15', 'region_16', 'region_17','region_18', 'region_19', 'region_2', 'region_20', 'region_21','region_22', 'region_23', 'region_24', 'region_25', 'region_26','region_27', 'region_28', 'region_29', 'region_3', 'region_30','region_31', 'region_32', 'region_33', 'region_34', 'region_4','region_5', 'region_6', 'region_7', 'region_8', 'region_9','length_of_service','employee_id','Operations','R&D','other','Finance','Below Secondary','sourcing'],axis=1,inplace=True)


# #### Splitting the dependent and independent variable

# In[434]:


Y = training['is_promoted'].values
X = training.drop(['is_promoted'],axis=1).values


# **As we have seen that the dataset is imbalance so now we will balance our dataset and will try diferent ML models.**

# In[435]:


#Get the Promoted and the not-promoted dataset 

Promoted = training[training['is_promoted']==1]

notpromoted = training[training['is_promoted']== 0]


# In[436]:


print(Promoted.shape,notpromoted.shape)


# In[425]:


from imblearn.under_sampling import NearMiss


# In[437]:


# Implementing Undersampling for Handling Imbalanced 
nm = NearMiss()
X_res,y_res=nm.fit_sample(X,Y)


# In[438]:


X_res.shape,y_res.shape


# In[439]:


from collections import Counter
print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))


# In[491]:


X_train,x_test,Y_train,y_test = train_test_split(X_res,y_res,test_size = 0.7)


# In[492]:


print(X_train.shape)
print(x_test.shape)
print(Y_train.shape)
print(y_test.shape)


# # Implement Machine Learning Models

# ### Logistic Regression

# In[472]:


from sklearn.linear_model import LogisticRegression
logist = LogisticRegression()
logist.fit(X_train,Y_train)


# In[473]:


predictions = logist.predict(x_test)


# ### Random forest

# In[467]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier().fit(X_res,y_res)


# In[468]:


# predict on test set
rfc_pred = rfc.predict(x_test)


# ### DecisionTreeClassifier

# In[477]:


from sklearn.tree import DecisionTreeClassifier
# Create decision tree classifer object
clf = DecisionTreeClassifier(class_weight='balanced')
# Train model
modeldt = clf.fit(X_train,Y_train)


# In[478]:


# predict on test set
dt_pred = rfc.predict(x_test)


# # Model Evaluation

# ### Logistic Regression

# >Using score method on test data to test the accuracy of our logistic regression model
# 

# In[474]:


logist.score(x_test, y_test)


# In[444]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# *F1 SCORE*

# In[475]:


from sklearn.metrics import f1_score
f1_score(y_test, predictions)


# *CONFUSION MATRIX*

# In[476]:


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# *K-FOLD CROSS VALIDATION*

# In[447]:


from sklearn.model_selection import cross_val_score
a=cross_val_score(logist,x_test,y_test ,cv=10, scoring="accuracy")
a.mean()


# ### Random forest

# **Using score method on test data to test the accuracy of our Random forest model**

# In[469]:


rfc.score(x_test, y_test)


# *F1 SCORE*

# In[470]:


f1_score(y_test, rfc_pred)


# *CONFUSION MATRIX*

# In[471]:


print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))


# *K-FOLD CROSS VALIDATION*

# In[483]:


from sklearn.model_selection import cross_val_score
a=cross_val_score(rfc,x_test,y_test ,cv=10, scoring="accuracy")
a.mean()


# ### Decision Tree

# **Using score method on test data to test the accuracy of our Decision Tree model**

# In[479]:


clf.score(x_test, y_test)


# *F1 SCORE*

# In[480]:


f1_score(y_test, dt_pred)


# *CONFUSION MATRIX*

# In[481]:


print(classification_report(y_test,dt_pred))
print(confusion_matrix(y_test,dt_pred))


# *K-FOLD CROSS VALIDATION*

# In[482]:


from sklearn.model_selection import cross_val_score
a=cross_val_score(clf,x_test,y_test ,cv=10, scoring="accuracy")
a.mean()


# ## Conclusion

# **Our 3 models that is Random forest, Logistic Regression, Decision tree does not performed very well due to less amount of data .If we use the imbalance dataset Xgboost algorithm works best for imblanced dataset.**
# 
# *Hence I have learnt a lot in this project like how to balance an imbalance dataset and many more things.*
# >Thanks You
