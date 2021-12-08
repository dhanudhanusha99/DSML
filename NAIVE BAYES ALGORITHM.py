#!/usr/bin/env python
# coding: utf-8

# In[29]:


from sklearn.datasets import load_diabetes
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


# In[30]:


d = load_diabetes()
x = d.data
y = d.target


# In[31]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)


# In[8]:


gnb = GaussianNB()
gnb.fit(x_train, y_train)
prediction = gnb.predict(x_test)
print(prediction)


# In[35]:


print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, prediction)*100)


# In[36]:


cm=np.array(confusion_matrix(y_test,prediction))
print(cm)


# In[37]:


plt.plot(x,y)
plt.show()


# In[ ]:





# In[ ]:




