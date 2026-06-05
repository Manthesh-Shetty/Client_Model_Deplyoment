#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[5]:


df=pd.read_csv('claimants.csv')
df

df.shape
# In[6]:


df.shape


# In[7]:


df.head()


# In[8]:


df.drop(columns=['CASENUM'],inplace=True)


# In[9]:


df.head()


# In[10]:


df.isnull().sum()


# In[11]:


df.fillna({'CLMSEX':df.CLMSEX.mode()[0],'CLMINSUR':df.CLMINSUR.mode()[0],
           'SEATBELT':df.SEATBELT.mode()[0],'CLMAGE':df.CLMAGE.median()},inplace=True)


# In[12]:


df.isnull().sum()


# In[13]:


df.duplicated().sum()


# In[14]:


df.drop_duplicates(inplace=True)


# In[15]:


df.duplicated().sum()


# In[16]:


df.boxplot()


# In[17]:


def outlier_capping(df,column):
    Q1=df[column].quantile(0.25)
    Q3=df[column].quantile(0.75)
    IQR=Q3-Q1
    lower_extreme=Q1-1.5*IQR
    upper_extreme=Q3+1.5*IQR
    df[column]=df[column].apply (lambda x:lower_extreme if x<lower_extreme else upper_extreme if x>upper_extreme else x)
for col in df.select_dtypes(['int','float']).columns:
    outlier_capping(df,col)


# In[18]:


df.boxplot()
plt.show()


# In[19]:


features= df.drop (columns=['ATTORNEY'])
features.head()


# In[33]:


target =df[['ATTORNEY']]


# In[34]:


x_train,x_test,y_train,y_test= train_test_split(features,target,train_size=0.8,random_state=100,stratify=target.ATTORNEY)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[35]:


x_train.head()


# In[37]:


std_sca=StandardScaler()
x_train[['CLMAGE','LOSS']]=std_sca.fit_transform(x_train[['CLMAGE','LOSS']])
x_test[['CLMAGE','LOSS']]=std_sca.fit_transform(x_test[['CLMAGE','LOSS']])


# In[38]:


x_test.head()


# In[41]:


log_model=LogisticRegressionCV(cv=3)
log_model.fit(x_train,y_train)
y_pred=log_model.predict(x_test)
accuracy_score(y_test,y_pred)


# In[43]:


import pickle


# In[49]:


file='log12.pkl'


# In[45]:


pickle.dump(log_model,open(file,'wb'))


# In[48]:


file1='std11.pkl'
pickle.dump(std_sca,open(file1,'wb'))


# In[ ]:




