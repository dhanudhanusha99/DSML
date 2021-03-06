#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[32]:


cell_df = pd.read_csv("cell_samples.csv")
cell_df.head()


# In[33]:


ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()


# In[34]:


cell_df.dtypes


# In[35]:


cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes


# In[36]:


feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
X[0:5]


# In[37]:


cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
y [0:5]


# In[38]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[39]:


from sklearn import svm
s = svm.SVC(kernel='rbf')
s.fit(X_train, y_train) 


# In[40]:


pred = clf.predict(X_test)
pred [0:5]


# In[41]:


from sklearn.metrics import classification_report, confusion_matrix


# In[43]:


print(classification_report(y_test,pred))


# In[44]:


print(confusion_matrix(y_test,pred))


# In[ ]:




