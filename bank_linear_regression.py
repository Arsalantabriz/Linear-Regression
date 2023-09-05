#!/usr/bin/env python
# coding: utf-8

# # Calculating the Accuracy of the Model

# Using the same dataset, expand the model by including all other features into the regression. 
# 
# Moreover, calculate the accuracy of the model and create a confusion matrix

# ## Import the relevant libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


# ## Load the data

# Load the ‘Bank_data.csv’ dataset.

# In[5]:


raw_data = pd.read_csv('Bank_data.csv')
raw_data


# In[6]:


# We make sure to create a copy of the data before we start altering it. Note that we don't change the original data we loaded.
data = raw_data.copy()
# Removes the index column thata comes with the data
data = data.drop(['Unnamed: 0'], axis = 1)
# We use the map function to change any 'yes' values to 1 and 'no'values to 0. 
data['y'] = data['y'].map({'yes':1, 'no':0})
data


# In[ ]:





# ### Declare the dependent and independent variables

# Use 'duration' as the independet variable.

# In[7]:


estimators=['interest_rate','march','credit','previous','duration']

X1 = data[estimators]
y = data['y']


# ### Simple Logistic Regression

# Run the regression and graph the scatter plot.

# In[8]:


X = sm.add_constant(X1)
reg_logit = sm.Logit(y,X)
results_logit = reg_logit.fit()
results_logit.summary2()


# In[ ]:





# ## Expand the model

# We can be omitting many causal factors in our simple logistic model, so we instead switch to a multivariate logistic regression model. Add the ‘interest_rate’, ‘march’, ‘credit’ and ‘previous’ estimators to our model and run the regression again. 

# ### Declare the independent variable(s)

# In[ ]:





# In[ ]:





# ### Confusion Matrix

# Create the confusion matrix of the model and estimate its accuracy. 

# <i> For convenience we have already provided you with a function that finds the confusion matrix and the model accuracy.</i>

# In[9]:


def confusion_matrix(data,actual_values,model):
        
        # Confusion matrix 
        
        # Parameters
        # ----------
        # data: data frame or array
            # data is a data frame formatted in the same way as your input data (without the actual values)
            # e.g. const, var1, var2, etc. Order is very important!
        # actual_values: data frame or array
            # These are the actual values from the test_data
            # In the case of a logistic regression, it should be a single column with 0s and 1s
            
        # model: a LogitResults object
            # this is the variable where you have the fitted model 
            # e.g. results_log in this course
        # ----------
        
        #Predict the values using the Logit model
        pred_values = model.predict(data)
        # Specify the bins 
        bins=np.array([0,0.5,1])
        # Create a histogram, where if values are between 0 and 0.5 tell will be considered 0
        # if they are between 0.5 and 1, they will be considered 1
        cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
        # Calculate the accuracy
        accuracy = (cm[0,0]+cm[1,1])/cm.sum()
        # Return the confusion matrix and 
        return cm, accuracy


# In[10]:


confusion_matrix(X,y,results_logit)


# In[ ]:





# In[ ]:




