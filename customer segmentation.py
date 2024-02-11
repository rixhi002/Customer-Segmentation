#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# In[9]:


customers_data = pd.read_csv("customers-data - customers-data.csv")


# In[10]:


customers_data.head()


# In[11]:


kmeans_model = KMeans(init='k-means++',  max_iter=400, random_state=42)


# In[13]:


kmeans_model.fit(customers_data[['products_purchased','complains',
'money_spent']])


# In[14]:


def try_different_clusters(K, data):

    cluster_values = list(range(1, K+1))
    inertias=[]

    for c in cluster_values:
        model = KMeans(n_clusters = c,init='k-means++',max_iter=400,random_state=42)
        model.fit(data)
        inertias.append(model.inertia_)

    return inertias


# In[16]:


outputs = try_different_clusters(12, customers_data[['products_purchased','complains','money_spent']])
distances = pd.DataFrame({"clusters": list(range(1, 13)),"sum of squared distances": outputs})


# In[17]:


figure = go.Figure()
figure.add_trace(go.Scatter(x=distances["clusters"], y=distances["sum of squared distances"]))

figure.update_layout(xaxis = dict(tick0 = 1,dtick = 1,tickmode = 'linear'),
                  xaxis_title="Number of clusters",
                  yaxis_title="Sum of squared distances",
                  title_text="Finding optimal number of clusters using elbow method")
figure.show()


# In[19]:


kmeans_model_new = KMeans(n_clusters = 5,init='k-means++',max_iter=400,random_state=42)

kmeans_model_new.fit_predict(customers_data[['products_purchased','complains','money_spent']])


# In[20]:


cluster_centers = kmeans_model_new.cluster_centers_
data = np.expm1(cluster_centers)
points = np.append(data, cluster_centers, axis=1)
points


# In[22]:


points = np.append(points, [[0], [1], [2], [3], [4]], axis=1)
customers_data["clusters"] = kmeans_model_new.labels_


# In[23]:


customers_data.head()


# In[25]:


figure = px.scatter_3d(customers_data,
                    color='clusters',
                    x="products_purchased",
                    y="complains",
                    z="money_spent",
                    category_orders = {"clusters": ["0", "1", "2", "3", "4"]}
                    )
figure.update_layout()
figure.show()


# In[ ]:




