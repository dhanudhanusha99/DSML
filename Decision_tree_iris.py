#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import train_test_split


# In[2]:


data=load_iris()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=50,test_size=0.25)


# In[4]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)


# In[5]:


y_pred=clf.predict(x_test)
print("Train data accuracy : ",accuracy_score(y_true=y_train,y_pred=clf.predict(x_train)))
print("Test data accuracy : ",accuracy_score(y_true=y_test,y_pred=y_pred))


# In[6]:


tree.plot_tree(clf)


# In[ ]:




