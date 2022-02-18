#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[11]:


from sklearn.datasets import load_breast_cancer


# In[13]:


cancer = load_breast_cancer()


# In[16]:


X =cancer.data
y = cancer.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# In[17]:


from sklearn.svm import SVC


# In[18]:


model = SVC()
model.fit(X_train,y_train)


# In[19]:


predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))


# In[20]:


print(classification_report(y_test,predictions))


# In[ ]:




