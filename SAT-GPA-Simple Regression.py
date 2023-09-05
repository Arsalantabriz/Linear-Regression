#!/usr/bin/env python
# coding: utf-8

# # Linear Regression

# ## Importing The Library

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()


# ## Loading the Data

# In[6]:


data=pd.read_csv("1.01. Simple linear regression.csv")
data.head(5)


# In[7]:


data.describe()


# ## Defining the Dependant and Independant Variable

# In[9]:


x1=data["SAT"]
y=data["GPA"]


# ## Explore the Data 

# In[10]:


plt.scatter(x1,y)
plt.xlabel("SAT",fontsize=20)
plt.ylabel("GPA",fontsize=20)
plt.show()


# ## Regression Analysis 

# ### understanding the constant and coefficient 

# In[11]:


x=sm.add_constant(x1)
results=sm.OLS(y,x).fit()
results.summary()


# ### ploting the regression analysis

# In[13]:


plt.scatter(x1,y)
yhat=0.0017*x1+0.2750
fig=plt.plot(x1,yhat,lw=4,c="orange",label="regression line")
plt.xlabel("SAT",fontsize=20)
plt.ylabel("GPA",fontsize=20)
plt.show()


# In[ ]:




