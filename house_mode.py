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

sns.set()

pd.set_option('display.max_colwidth', 900)
pd.set_option('display.max_rows', 500)

X_full = pd.read_csv('train.csv')
X_test_full = pd.read_csv('test.csv')
#print('Train shape:', X_full.shape)
#print('Test shape:', X_test_full.shape)

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

all_data_na = (X_full.isnull().sum() / len(X_full)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)

f, ax = plt.subplots(figsize=(10, 8))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


corrmat = X_full.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(X_full[top_corr_features].corr(),annot=True)


var = 'GrLivArea'
data = pd.concat([X_full['SalePrice'], X_full[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))



var = 'OverallQual'
data = pd.concat([X_full['SalePrice'], X_full[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)



X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = np.log(X_full.SalePrice)
X_full.drop(['SalePrice'], axis=1, inplace=True)

ind = X_test_full.Id

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.7, random_state=0)

X_train = X_train_full[selected].copy()
X_valid = X_valid_full[selected].copy()
X_test = X_test_full[selected].copy()
X_full = X_full[selected].copy()

missing_val_count_by_column = (X_train.isnull().sum())
print('\nMissing Values')
    
print(missing_val_count_by_column[missing_val_count_by_column > 0])

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

good_label_cols = [col for col in object_cols if set(X_train[col]) == set(X_valid[col])]
        
bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
print('\nCategorical columns that will be label encoded:', good_label_cols)
print('Categorical columns that will be dropped from the dataset:', bad_label_cols)


X_train = X_train.drop(bad_label_cols, axis=1)
X_valid = X_valid.drop(bad_label_cols, axis=1)
X_test = X_test.drop(bad_label_cols, axis=1)
X_full = X_full.drop(bad_label_cols, axis=1)

label_encoder = LabelEncoder()
for col in good_label_cols:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_valid[col] = label_encoder.transform(X_valid[col])
    X_test[col] = label_encoder.fit_transform(X_test[col])
    X_full[col] = label_encoder.fit_transform(X_full[col])



my_imputer = SimpleImputer() 
X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
X_valid = pd.DataFrame(my_imputer.transform(X_valid))
X_test = pd.DataFrame(my_imputer.fit_transform(X_test))
X_full = pd.DataFrame(my_imputer.fit_transform(X_full))

X_train.columns = X_train.columns
X_valid.columns = X_valid.columns
X_test.columns = X_test.columns
X_full.columns = X_full.columns

print('\nAfter preprocessing: ', X_train.shape)
print('After preprocessing X_test: ', X_train.shape)

model = XGBRegressor(n_estimators=350, learning_rate=0.09, max_depth=6, random_state=42) 
model.fit(X_train, y_train) 

predictions = model.predict(X_valid)

rmse = np.sqrt(metrics.mean_squared_error(predictions, y_valid))

print('\n************************************')
print("Mean Absolute Error:", rmse)
print('**************************************') 
