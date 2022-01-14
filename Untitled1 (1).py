#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from numpy import reshape
import seaborn as sns
import pandas as pd


# In[5]:


iris=load_iris()
x=iris.data
y=iris.target


# In[6]:


tsne=TSNE(n_components=2,verbose=1,random_state=123)
z=tsne.fit_transform(x)
df=pd.DataFrame()
df["y"]=y
df["comp-1"]=z[:,0]
df["comp-2"]=z[:,1]
sns.scatterplot(x="comp-1",y="comp-2",hue=df.y.tolist(),palette=sns.color_palette("hls",3),data=df).set(title="Iris data T-SNE projectopn")


# In[ ]:




