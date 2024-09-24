#!/usr/bin/env python
# coding: utf-8

# ## **first ML project**

# ### **Load data**

# In[2]:


import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
df


# ## **Data Preparation**

# ### data separation as X and Y

# In[4]:


y = df['logS']
y


# In[7]:


X = df.drop ('logS', axis = 1 )
X


# ### Data Splitting

# In[8]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size =0.2, random_state = 100)


# In[11]:


X_train #80% of the data


# In[12]:


X_test #20% of the data


# ## **Model Building**

# ### **Linear Regression**

# **Training the Model**

# In[14]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)


# **Applying the Model to make a predicition**

# In[16]:


y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)


# **Evaluate Model Performance**

# In[17]:


from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score (y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error (y_test, y_lr_test_pred)
lr_test_r2 = r2_score (y_test, y_lr_test_pred)


# In[22]:


print('LR MSE (Train): ', lr_train_mse)
print('LR R2 (Train): ', lr_train_r2)
print('LR MSE (Test): ', lr_test_mse)
print('LR R2 (Test): ', lr_test_r2)


# In[25]:


lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']


# In[26]:


lr_results


# ## **Random Forest**

# **Training the Model**

# In[28]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor( max_depth = 2, random_state = 100)
rf.fit(X_train, y_train)


# **Applying the Model to make a Prediction**

# In[29]:


y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)


# **Evalute Model Performance**

# In[30]:


from sklearn.metrics import mean_squared_error, r2_score

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score (y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error (y_test, y_rf_test_pred)
rf_test_r2 = r2_score (y_test, y_rf_test_pred)


# In[33]:


rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
rf_results


# ## **Model Comparison**

# In[34]:


df_models = pd.concat([lr_results, rf_results], axis =0)
df_models


# In[36]:


df_models.reset_index(drop = True)


# # **Data Visualization of prediction results**

# In[41]:


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5,5))
plt.scatter (x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.3)
z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), '#F8766D')
plt.ylabel('Predict LogS')
plt.xlabel('Experimental LogS')


# In[ ]:




