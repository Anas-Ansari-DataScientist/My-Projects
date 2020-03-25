#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[10]:


# Part 1 - Data Preprocessing
# Importing the dataset
from google.colab import files
uploaded = files.upload()


# In[17]:


import io
dataset = pd.read_csv(io.BytesIO(uploaded['Churn_Modelling.csv']))
dataset


# In[ ]:


X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]


# In[ ]:


#Create dummy variables
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)


# In[ ]:


## Concatenate the Data Frames

X=pd.concat([X,geography,gender],axis=1)


# In[ ]:


## Drop Unnecessary columns
X=X.drop(['Geography','Gender'],axis=1)


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential     #responsible for creating any kid of NN like ANN, CNN, RNN
from keras.layers import Dense          #for creating hidden layers
#from keras.layers import LeakyReLU,PReLU,ELU      #Diffrent types of activation functions
from keras.layers import Dropout        #A regularization parameter used for regularization of dataset


# In[37]:


# Initialising the ANN
classifier = Sequential()


# In[ ]:


# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 11))    #output_dim means the number of neurons in 1st hidden layer output dimention got updated as units
                                                                                                        #init means the weight initialization techniques that we are using but init is got updated as kernel_initializer
                                                                                                        #activation means the activatio function that we are using
                                                                                                        #input_dim means how many inupt features are connected to the hidden layer


# In[ ]:


# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))


# In[ ]:


# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))


# In[44]:


classifier.summary()


# In[ ]:


# Compiling the ANN
classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])   #optimizer Adam is best for compiling
                                                                                               #Loss is binary_crossentropy beacause we are using categorical output as 0 and 1 if a regression problem we should use categorical_crossentropy


# In[47]:


# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, nb_epoch = 100)      #Advantages of using a batch size < number of all samples: It requires less memory. Since you train the network using fewer samples, the overall training procedure requires less memory. and uses less time and less space in Ram
                                                                                                           #nb_epoch is number of epoch its gonna tak while training


# In[48]:


# list all data in history

print(model_history.history.keys())
# summarize history for accuracy
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[49]:


# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[62]:


# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred


# In[54]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[53]:



# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
score

