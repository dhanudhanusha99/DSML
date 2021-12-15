#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score


# In[27]:


dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]
x


# In[28]:


y


# In[29]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 42)


# In[30]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[33]:


y_pred = regressor.predict(x_test)
y_pred



# In[35]:


score=r2_score(y_test,y_pred)
print(f'R2 score: {score}')


# In[36]:


z=regressor.predict([[12]])
z


# In[37]:


from sklearn.metrics import r2_score

score=r2_score(y_test,y_pred)
print(f'R2 score: {score}')


# In[38]:


plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience {Test set}')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:




