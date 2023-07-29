#!/usr/bin/env python
# coding: utf-8

# ## Import Library

# In[44]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot

import warnings
warnings.filterwarnings('ignore')


# ## Baca Data

# In[45]:


df_customer = pd.read_csv('Customer.csv', delimiter=';')
df_product = pd.read_csv('Product.csv', delimiter=';')
df_store = pd.read_csv('Store.csv', delimiter=';')
df_transaction = pd.read_csv('Transaction.csv', delimiter=';')

df_customer.shape, df_product.shape, df_store.shape, df_transaction.shape


# ## Data Cleansing

# In[46]:


df_customer.head()


# In[47]:


df_product.head()


# In[48]:


df_store.head()


# In[49]:


df_transaction.head()


# In[50]:


#data cleansing df_customer
df_customer['Income'] = df_customer['Income'].replace('[,]','.',regex=True).astype('float')


# In[51]:


#data cleansing df_store
df_store['Latitude'] = df_store['Latitude'].replace('[,]','.',regex=True).astype('float')
df_store['Longitude'] = df_store['Longitude'].replace('[,]','.',regex=True).astype('float')


# In[52]:


#data cleansing df_transaction
df_transaction['Date'] = pd.to_datetime(df_transaction['Date'])


# ## Gabungkan Semua Data

# In[53]:


df_merge = pd.merge(df_transaction, df_customer, on=['CustomerID'])
df_merge = pd.merge(df_merge, df_product.drop(columns=['Price']), on=['ProductID'])
df_merge = pd.merge(df_merge, df_store, on=['StoreID'])


# In[54]:


df_merge.head()


# ## Model Machine Learning Regresi

# In[55]:


df_regresi = df_merge.groupby(['Date']).agg({
        'Qty' : 'sum'
}).reset_index()


# In[56]:


df_regresi


# In[57]:


decomposed = seasonal_decompose(df_regresi.set_index('Date'))

plt.figure(figsize=(8, 8))

plt.subplot(311)
decomposed.trend.plot(ax=plt.gca())
plt.title('Trend')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Seasonality')
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Residuals')

plt.tight_layout()


# In[58]:


cut_off = round(df_regresi.shape[0] * 0.9)
df_train = df_regresi[:cut_off]
df_test = df_regresi[cut_off:].reset_index(drop=True)
df_train.shape, df_test.shape


# In[59]:


df_train


# In[60]:


df_test


# In[61]:


plt.figure(figsize=(20,5))
sns.lineplot(data=df_train, x=df_train['Date'], y=df_train['Qty']);
sns.lineplot(data=df_test, x=df_test['Date'], y=df_test['Qty']);


# In[62]:


autocorrelation_plot(df_regresi['Qty']);


# In[63]:


# function to calculate RMSE
def rmse(y_actual, y_pred):
    print(f'RMSE value {mean_squared_error(y_actual, y_pred)**0.5}')
    
# function to eval machine learnig modelling
def eval(y_actual, y_pred):
    rmse(y_actual, y_pred)
    print(f'MAE value {mean_absolute_error(y_actual, y_pred)}')


# In[64]:


# ARIMA
df_train = df_train.set_index('Date')
df_test = df_test.set_index('Date')

y = df_train['Qty']

ARIMAmodel = ARIMA(y, order = (40, 2, 1))
ARIMAmodel = ARIMAmodel.fit()

y_pred = ARIMAmodel.get_forecast(len(df_test))

y_pred_df = y_pred.conf_int()
y_pred_df['predictions'] = ARIMAmodel.predict(start =y_pred_df.index[0], end =y_pred_df.index[-1])
y_pred_df.index = df_test.index
y_pred_out = y_pred_df['predictions']
eval(df_test['Qty'], y_pred_out)

plt.figure(figsize=(20,5))
plt.plot(df_train['Qty'])
plt.plot(df_test['Qty'], color='red')
plt.plot(y_pred_out, color='black', label = 'ARIMA Predictions')
plt.legend()


# ## Clustering

# In[65]:


df_merge.head()


# In[66]:


df_cluster = df_merge.groupby(['CustomerID']).agg({
    'TransactionID' : 'count',
    'Qty' : 'sum',
    'TotalAmount' : 'sum'
}).reset_index()


# In[67]:


df_cluster.head()


# In[68]:


df_cluster


# In[69]:


data_cluster = df_cluster.drop(columns=['CustomerID'])

data_cluster_normalize = preprocessing.normalize(data_cluster)


# In[70]:


data_cluster_normalize


# In[72]:


K = range(2, 8)
fits = []
score = []

for k in K:
    model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(data_cluster_normalize)
    
    fits.append(model)
    
    score.append(silhouette_score(data_cluster_normalize, model.labels_, metric='euclidean'))


# In[73]:


# choose 4 cluster
sns.lineplot(x = K, y =score);


# In[74]:


df_cluster['cluster_label'] = fits[2].labels_


# In[75]:


df_cluster.groupby(['cluster_label']).agg({
    'CustomerID' : 'count',
    'TransactionID' : 'mean',
    'Qty' : 'mean',
    'TotalAmount' : 'mean'
})


# cluster 0 -> TotalAmount ke-1 & Qty ke-3,
# cluster 2 -> Qty ke-1 & TotalAmount ke-2
