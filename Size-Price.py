#!/usr/bin/env python
# coding: utf-8

# ## Importing Library 

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()


# ## Load Data Set 

# In[11]:


data=pd.read_csv("real_estate_price_size.csv")
data.describe()
data.head(5)


# ## Defining the dependant and independant variables

# In[6]:


##we have to define the x and y axis as dependant and independant variables
x1=data["size"]
y=data["price"]


# ## The Priamary plot of the data 

# In[7]:


plt.scatter(x1,y)
plt.xlabel("size",fontsize=15)
plt.ylabel("price",fontsize=15)
plt.show()


# ## Using statsmodel and OLS fitting 

# In[9]:


x=sm.add_constant(x1)
results=sm.OLS(y,x).fit()
results.summary()


# ## Forming Regression Plot 

# In[10]:


yhat=223.1787*x1+101900
plt.scatter(x1,y)
plt.plot(x1,yhat,lw=4,c="red",label="regression line")
plt.xlabel("size")
plt.ylabel("price")
plt.show()


# In[ ]:




