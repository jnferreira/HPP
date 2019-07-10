import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import eli5
from sklearn.impute import SimpleImputer
from eli5.sklearn import PermutationImportance
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn import metrics
from xgboost import XGBRegressor
from scipy.stats import norm, skew

sns.set()

pd.set_option('display.max_colwidth', 900)
pd.set_option('display.max_rows', 500)

X_full = pd.read_csv('./train.csv')
X_test_full = pd.read_csv('test.csv')
#print('Train shape:', X_full.shape)
#print('Test shape:', X_test_full.shape)

y = X_full['SalePrice']

selected = ['GrLivArea',
 'LotArea',
 'BsmtUnfSF',
 '1stFlrSF',
 'TotalBsmtSF',
 'GarageArea',
 'BsmtFinSF1',
 'LotFrontage',
 'YearBuilt',
 'Neighborhood',
 'GarageYrBlt',
 'OpenPorchSF',
 'YearRemodAdd',
 'WoodDeckSF',
 'MoSold',
 '2ndFlrSF',
 'OverallCond',
 'Exterior1st',
 'YrSold',
 'OverallQual']

#sns.distplot(X_full['SalePrice'])

############## NUMERICAL AND CATEGORICAL VALUES ###############

print('\nNumerical columns ---> {} \nCategorical columns ---> {}'.format(list(X_full.select_dtypes(include=[np.number]).shape)[1], list(X_full.select_dtypes(include=['object']).shape)[1]))

################################################################


############## Break down the columns by their type ###############

print('\nColumns by their type \n {}'.format(X_full.columns.to_series().groupby(X_full.dtypes).groups))

################################################################

print("\nColumns have missing values: {}".format(len(X_full.columns[X_full.isna().any()].tolist())))
print('\n', X_full.columns[X_full.isna().any()].tolist())

################################################################

print('\n--------------------------------------------------------------------------------------')
print('\n', X_full.describe())
print('--------------------------------------------------------------------------------------')

#X_full.hist(figsize=(18,18))

corr = X_full.corr(method='spearman')
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

#plt.figure(figsize=(15, 10))
#sns.heatmap(corr, vmax=.5, mask=mask, fmt='.2f', linewidths=.2)

correlations = X_full.corr(method='spearman')['SalePrice'].sort_values(ascending=False)
correlations_abs = correlations.abs()
print('\nTop 10 correlations (absolute):\n', correlations_abs.head(11))


##########  Data Preparation  #######################
"""
    Data Cleaning: Outliers and Missing Values, 
    Feature Selection, 
    Feature Engineering, 
    Label Encoding and Feature Scaling
"""

target_tr = np.log1p(y)

#sns.distplot(target_tr , fit=norm);
#(mu, sigma) = norm.fit(target_tr)
#plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
#plt.title('Price (Log)')
#print("Skewness: %f" % target_tr.skew())


perc_na = (X_full.isnull().sum()/len(X_full))*100
ratio_na = perc_na.sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Values Ratio' :ratio_na})
print(missing_data.shape)
missing_data.head(20)






