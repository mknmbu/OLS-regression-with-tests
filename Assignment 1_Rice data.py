#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Get relevant python libraries

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns


# In[2]:


## Read datafile
df = pd.read_csv("./rice2.csv")


# In[3]:


## useful Commands
df.info()
df.head()


# In[4]:


## Short Summary of the dataset
df.describe()


# In[5]:


## log transformation of the variables
lnq=np.log(df['prod'])
lna=np.log(df['area'])
lnl=np.log(df['labor'])
lnf=np.log(df['fert'])


# In[6]:


# estimate Cobb-Douglas production function
cdmod = smf.ols(formula='lnq ~ lna + lnl + lnf', data=df)
cdres = cdmod.fit()
print(cdres.summary())


# In[7]:


## Residuals from the model
lnpres = cdres.resid
print('Mean:    {}'.format(np.mean(lnpres)))
print('Std dev: {}'.format(np.std(lnpres)))


# In[9]:


## Histogram of the residuals
fig, ax = plt.subplots(figsize=(8,6))
#
# histogram
num_bins=15
ax.hist(lnpres, num_bins, density=True)
ax.set_xlabel('Regression residuals')
ax.set_ylabel('Probability density')
ax.set_axisbelow(True)
ax.xaxis.grid(linestyle='dashed')
ax.yaxis.grid(linestyle='dashed')
ax.grid(True)


# In[10]:


# Get different Variables for diagnostic
residuals = cdres.resid
fitted_value = cdres.fittedvalues
stand_resids = cdres.resid_pearson
influence = cdres.get_influence()
leverage = influence.hat_matrix_diag
norm_residuals = cdres.get_influence().resid_studentized_internal
norm_residuals_abs_sqrt = np.sqrt(np.abs(norm_residuals))


# In[11]:


## Ploting residuals against the predicted values of lnq

residuals = cdres.resid
fitted_value = cdres.fittedvalues

fig, ax = plt.subplots(figsize=(8,6))
fig.suptitle("Predicted values vs residuals", fontsize=16)
ax.scatter(fitted_value,residuals)
ax.set_xlabel('Predicted value')
ax.set_ylabel('Residuals')
ax.set_ylim((-1.5,2.0))
ax.grid(True)
fig.tight_layout()


# In[12]:


#Normalized residuals
norm_residuals = cdres.get_influence().resid_studentized_internal

#Leverage
lnqleverage = cdres.get_influence().hat_matrix_diag

#Plot of leverage vs normalized residuals

fig, ax = plt.subplots(figsize=(8,6))
fig.suptitle("Leverage vs normalized residuals",fontsize=16)
ax.scatter(lnqleverage,norm_residuals)
ax.set_xlabel('Leverage')
ax.set_ylabel('Normalized residuals')
ax.set_ylim(-1.5,2.0)
ax.grid(True)
fig.tight_layout()


# In[14]:


##Ramsey Reset Test

df['y2'] = cdres.fittedvalues ** 2
df['y3'] = cdres.fittedvalues ** 3

rrtmod = smf.ols(formula='np.log(prod) ~ np.log(area) + np.log(labor) + np.log(area) + np.log(labor) + np.log(fert)+ y2 + y3', data=df)
rrtres = rrtmod.fit()

rrhyp = ['y2=0', 'y3=0']
 
f0test = rrtres.f_test(rrhyp)
f0stat = f0test.statistic
f0pval = f0test.pvalue
print()
print('Ramsey RESET test')
print('F-stat (HC0 errors): {}'.format(f0stat))
print('            p-value: {}'.format(f0pval))


# In[17]:


## Heteroskedasticity test

## QQ plot
fitted_value = cdres.fittedvalues
norm_residuals = cdres.get_influence().resid_studentized_internal
norm_residuals_abs_sqrt = np.sqrt(np.abs(norm_residuals))


plot_LM = plt.figure()
plt.scatter(fitted_value, norm_residuals_abs_sqrt, alpha=0.5);
sns.regplot(fitted_value, norm_residuals_abs_sqrt, 
            scatter=False, 
            ci=False, 
            lowess=True, 
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
plot_LM.axes[0].set_title('Scale Location')
plot_LM.axes[0].set_xlabel('Fitted values')
plot_LM.asex[0].set_ylable('$\sqrt{│Standardized Residuals│}$');


# In[24]:


import scipy.stats as stats


# In[25]:


df['u_sqr'] = cdres.resid ** 2
bphmod = smf.ols(formula='u_sqr ~ np.log(area) + np.log(labor) + np.log(fert)', data=df)
bphres = bphmod.fit()
bph_lm = bphres.nobs * bphres.rsquared
bph_pv = 1 - stats.chi2.cdf(x=bph_lm,df= bphres.df_model)

print()
print('Testing for heteroskedasticity')
print('Aux regression, R-squared = {}'.format(bphres.rsquared))
print('Aux regression, d of free = {}'.format(bphres.df_model))
print('Number of observations    = {}'.format(bphres.nobs))
print('Breusch-Pagan LM-test     = {}'.format(round(bph_lm,2)))
print('              p-value     = {}'.format(round(bph_pv,4)))
print()


# In[27]:


#Generalized Cobb-Douglas Production Function

#getting var from datafile

x1 = df ['area'].values
x2 = df ['labor'].values
x3 = df ['fert'].values

#Do log trans

lnx1 = np.log(x1)
lnx2 = np.log(x2)
lnx3 = np.log(x3)

#Making new variables

lnx1sqr = lnx1**2
lnx2sqr = lnx2**2
lnx3sqr = lnx3**2

lnx1sqr_div = lnx1sqr/2
lnx2sqr_div = lnx2sqr/2
lnx3sqr_div = lnx3sqr/2

lnx1lnx2 = lnx1 * lnx2
lnx1lnx3 = lnx1 * lnx3
lnx2lnx3 = lnx2 * lnx3

#Estimating translog function

transmod = smf.ols(formula='np.log(prod) ~ lnx1 + lnx2 + lnx3 + lnx1sqr_div + lnx2sqr_div + lnx3sqr_div + lnx1lnx2 + lnx1lnx3 + lnx2lnx3', data=df)
transres = transmod.fit()
print(transres.summary())


# In[31]:


## Ramsey RESET Test
df['y4'] = transres.fittedvalues **2
df['y5'] = transres.fittedvalues **3

rrtmod2 = smf.ols(formula='np.log(prod) ~ lnx1 + lnx2 + lnx3 + lnx1sqr_div + lnx2sqr_div + lnx3sqr_div + lnx1lnx2 + lnx1lnx3 + lnx2lnx3 + y4 + y5', data=df)
rrtres2 = rrtmod2.fit()

rrhyp2 = ['y4=0', 'y5=0']

f0test = rrtres2.f_test(rrhyp2)
f0stat = f0test.statistic
f0pval = f0test.pvalue
print()
print('Ramsey RESET test')
print('F-stat (HC0 errors: {}'.format(f0stat))
print('           p-value: {}'.format(f0pval))


# In[37]:


#Testing for heteroskedasticity

df['u_sqr2'] = transres.resid ** 2
bphmod2 = smf.ols(formula='u_sqr2 ~ lnx1 + lnx2 + lnx3 + lnx1sqr_div + lnx2sqr_div + lnx3sqr_div + lnx1lnx2 + lnx1lnx3 + lnx2lnx3', data=df)
bphres2 = bphmod2.fit()
bph_lm2 = bphres2.nobs * bphres2.rsquared
bph_pv2 = 1 - stats.chi2.cdf(x=bph_lm2, df= bphmod2.df_model)
print()
print('Aux regression, R-squared = {}'.format(bphres.rsquared))
print('Aux regressionm d of free = {}'.format(bphmod2.df_model))
print('Number of observations    = {}'.format(bphres.nobs))
print()
print('Breusch-Pagan LM-test: {}, p-value={}'.format(round(bph_lm,2), round(bph_pv,4)))


# In[ ]:




