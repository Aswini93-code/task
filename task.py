#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


data=pd.read_csv("Desktop\Iris (1).csv")


# In[9]:


data.head()


# In[11]:


data.tail()


# In[12]:


data.shape


# In[13]:


data.dtypes


# In[14]:


data.describe()


# In[16]:


data.info()


# In[17]:


data.corr()


# In[19]:


data.isnull().sum()


# In[21]:


data.columns


# In[23]:


plt.scatter(data.SepalLengthCm,data['SepalWidthCm'])
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')


# In[24]:


km = KMeans(n_clusters=3)
km


# In[26]:


y_predicted = km.fit_predict(data[['SepalLengthCm','SepalWidthCm']])
y_predicted


# In[27]:


data['cluster']=y_predicted
data.head()


# In[28]:


km.cluster_centers_


# In[29]:


data1 = data[data.cluster==0] #make 3 data frame and each blongs to different cluster
data2 = data[data.cluster==1]
data3 = data[data.cluster==2]
plt.scatter(data1.SepalLengthCm,data1['SepalWidthCm'],color='green')
plt.scatter(data2.SepalLengthCm,data2['SepalWidthCm'],color='red')
plt.scatter(data3.SepalLengthCm,data3['SepalWidthCm'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.legend()


# In[31]:


sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(data[['SepalLengthCm','SepalWidthCm']])
    sse.append(km.inertia_)


# In[32]:


plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)


# In[ ]:


# the number of clusters:3

