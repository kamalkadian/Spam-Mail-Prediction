#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


#Data collection and Pre-Processing


# In[3]:


raw_mail_data=pd.read_csv("mail_data.csv")


# In[4]:


raw_mail_data


# In[5]:


#replace the null values with a null string


# In[6]:


mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[7]:


mail_data.head()


# In[8]:


#label encoding:spam mail as 0; ham mail as 1;


# In[9]:


mail_data.loc[mail_data["Category"]=="spam","Category"]=0
mail_data.loc[mail_data["Category"]=="ham","Category"]=1


# In[10]:


X=mail_data["Message"]
y=mail_data["Category"]


# In[11]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=.85)


# In[12]:


#feature extraction
#transform the text data to feature vectors 
feature_extraction=TfidfVectorizer(min_df=1,stop_words="english",lowercase="True")

X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)


# In[13]:


y_train=y_train.astype("int")
y_test=y_test.astype("int")


# In[14]:


#Training the model


# In[15]:


model=LogisticRegression()


# In[16]:


#training the logistic regression model with the training data


# In[17]:


model.fit(X_train_features, y_train)


# In[18]:


#evaluating the trained model


# In[19]:


#prediction on training data


# In[20]:


prediction=model.predict(X_train_features)
accuracy=accuracy_score(y_train,prediction)


# In[21]:


accuracy


# In[22]:


#prediction on test data
prediction=model.predict(X_test_features)
accuracy=accuracy_score(y_test,prediction)


# In[23]:


accuracy


# In[24]:


#building a predictive system


# In[25]:


input_mail=["I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."]


# In[26]:


#converting text to numerical


# In[27]:


input_data_feature=feature_extraction.transform(input_mail)


# In[28]:


#predicton


# In[32]:


prediction=model.predict(input_data_feature)
print(prediction)
if prediction[0]==1:
    print("ham mail")
else:
    print("spam mail")

