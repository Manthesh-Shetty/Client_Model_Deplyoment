#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings ('ignore')
import pickle


# In[6]:


model=pickle.load(open('log12.pkl','rb'))


# In[9]:


std_scaler=pickle.load(open('std11.pkl', 'rb'))


# In[10]:


st.title('Model Deployment using Logistic Regressiion')


# In[12]:


def user_input_parameters():
    Gender = st.sidebar.radio('Select your Gender', ('Male', 'Female'))
    Insurance = st.sidebar.selectbox('Enter Insurance Details, Yes-1,No-0', [0, 1])
    Seatbelt = st.sidebar.selectbox('Enter Seatbelt Details, Yes-1, No-0', [0, 1])
    Age = st.sidebar.slider('Select your Age', 0, 100)
    Loss = st.sidebar.number_input('Enter the loss')
    dict1 = {'CLMSEX': Gender, 'CLMINSUR': Insurance, 'SEATBELT': Seatbelt, 'CLMAGE': Age, 'LOSS': Loss}
    features = pd.DataFrame(dict1, index=[0])
    features['CLMSEX'] = features['CLMSEX'].map({'Male': 1, 'Female': 0})
    features[['CLMAGE', 'LOSS']] = std_scaler.transform(features[['CLMAGE', 'LOSS']])
    return features
df = user_input_parameters()
pred = model.predict(df)
pred_prob = model.predict_proba(df)
button = st.button('Predict')
if button is True:
    st.subheader('Predict')
    st.write(pred_prob)


# In[ ]:




