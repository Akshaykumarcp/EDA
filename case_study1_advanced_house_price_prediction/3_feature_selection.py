import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso ## for feature slection
from sklearn.feature_selection import SelectFromModel ## for feature slection

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)

dataset=pd.read_csv('case_study1_advanced_house_price_prediction/X_train.csv')
dataset.head()
""" 
   Id  SalePrice  MSSubClass  MSZoning  LotFrontage   LotArea  Street  Alley  \
0   1  12.247694    0.235294      0.75     0.418208  0.366344     1.0    1.0
1   2  12.109011    0.000000      0.75     0.495064  0.391317     1.0    1.0
2   3  12.317167    0.235294      0.75     0.434909  0.422359     1.0    1.0
3   4  11.849398    0.294118      0.75     0.388581  0.390295     1.0    1.0
4   5  12.429216    0.235294      0.75     0.513123  0.468761     1.0    1.0

   LotShape  LandContour  Utilities  LotConfig  LandSlope  Neighborhood  \
0  0.000000     0.333333        1.0       0.00        0.0      0.636364
1  0.000000     0.333333        1.0       0.50        0.0      0.500000
2  0.333333     0.333333        1.0       0.00        0.0      0.636364
3  0.333333     0.333333        1.0       0.25        0.0      0.727273
4  0.333333     0.333333        1.0       0.50        0.0      1.000000

   Condition1  Condition2  BldgType  HouseStyle  OverallQual  OverallCond  \
0         0.4         1.0      0.75         1.0     0.666667        0.500
1         0.2         1.0      0.75         0.6     0.555556        0.875
2         0.4         1.0      0.75         1.0     0.666667        0.500
3         0.4         1.0      0.75         1.0     0.666667        0.500
4         0.4         1.0      0.75         1.0     0.777778        0.500

   YearBuilt  YearRemodAdd  RoofStyle  RoofMatl  Exterior1st  Exterior2nd  \
0   0.036765      0.098361        0.0       0.0          1.0          1.0
1   0.227941      0.524590        0.0       0.0          0.4          0.3
2   0.051471      0.114754        0.0       0.0          1.0          1.0
3   0.669118      0.606557        0.0       0.0          0.2          0.4
4   0.058824      0.147541        0.0       0.0          1.0          1.0

   MasVnrType  MasVnrArea  ExterQual  ExterCond  Foundation  BsmtQual  \
0        0.50     0.12250   0.666667        1.0        1.00      0.75
1        0.25     0.00000   0.333333        1.0        0.50      0.75
2        0.50     0.10125   0.666667        1.0        1.00      0.75
3        0.25     0.00000   0.333333        1.0        0.25      0.50
4        0.50     0.21875   0.666667        1.0        1.00      0.75

   BsmtCond  BsmtExposure  BsmtFinType1  BsmtFinSF1  BsmtFinType2  BsmtFinSF2  \
0      0.75          0.25      1.000000    0.125089      0.833333         0.0
1      0.75          1.00      0.666667    0.173281      0.833333         0.0
2      0.75          0.50      1.000000    0.086109      0.833333         0.0
3      1.00          0.25      0.666667    0.038271      0.833333         0.0
4      0.75          0.75      1.000000    0.116052      0.833333         0.0

   BsmtUnfSF  TotalBsmtSF  Heating  HeatingQC  CentralAir  Electrical  \
0   0.064212     0.140098      1.0       1.00         1.0         1.0
1   0.121575     0.206547      1.0       1.00         1.0         1.0
2   0.185788     0.150573      1.0       1.00         1.0         1.0
3   0.231164     0.123732      1.0       0.75         1.0         1.0
4   0.209760     0.187398      1.0       1.00         1.0         1.0

   1stFlrSF  2ndFlrSF  LowQualFinSF  GrLivArea  BsmtFullBath  BsmtHalfBath  \
0  0.356155  0.413559           0.0   0.577712      0.333333           0.0
1  0.503056  0.000000           0.0   0.470245      0.000000           0.5
2  0.383441  0.419370           0.0   0.593095      0.333333           0.0
3  0.399941  0.366102           0.0   0.579157      0.333333           0.0
4  0.466237  0.509927           0.0   0.666523      0.333333           0.0

   FullBath  HalfBath  BedroomAbvGr  KitchenAbvGr  KitchenQual  TotRmsAbvGrd  \
0  0.666667       0.5         0.375      0.333333     0.666667      0.500000
1  0.666667       0.0         0.375      0.333333     0.333333      0.333333
2  0.666667       0.5         0.375      0.333333     0.666667      0.333333
3  0.333333       0.0         0.375      0.333333     0.666667      0.416667
4  0.666667       0.5         0.500      0.333333     0.666667      0.583333

   Functional  Fireplaces  FireplaceQu  GarageType  GarageYrBlt  GarageFinish  \
0         1.0    0.000000          0.2         0.8     0.046729      0.666667
1         1.0    0.333333          0.6         0.8     0.289720      0.666667
2         1.0    0.333333          0.6         0.8     0.065421      0.666667
3         1.0    0.333333          0.8         0.4     0.074766      0.333333
4         1.0    0.333333          0.6         0.8     0.074766      0.666667

   GarageCars  GarageArea  GarageQual  GarageCond  PavedDrive  WoodDeckSF  \
0        0.50    0.386460    0.666667         1.0         1.0    0.000000
1        0.50    0.324401    0.666667         1.0         1.0    0.347725
2        0.50    0.428773    0.666667         1.0         1.0    0.000000
3        0.75    0.452750    0.666667         1.0         1.0    0.000000
4        0.75    0.589563    0.666667         1.0         1.0    0.224037

   OpenPorchSF  EnclosedPorch  3SsnPorch  ScreenPorch  PoolArea  PoolQC  \
0     0.111517       0.000000        0.0          0.0       0.0     0.0
1     0.000000       0.000000        0.0          0.0       0.0     0.0
2     0.076782       0.000000        0.0          0.0       0.0     0.0
3     0.063985       0.492754        0.0          0.0       0.0     0.0
4     0.153565       0.000000        0.0          0.0       0.0     0.0

   Fence  MiscFeature  MiscVal    MoSold  YrSold  SaleType  SaleCondition  \
0    1.0          1.0      0.0  0.090909    0.50  0.666667           0.75
1    1.0          1.0      0.0  0.363636    0.25  0.666667           0.75
2    1.0          1.0      0.0  0.727273    0.50  0.666667           0.75
3    1.0          1.0      0.0  0.090909    0.00  0.666667           0.00
4    1.0          1.0      0.0  1.000000    0.50  0.666667           0.75

   LotFrontagenan  MasVnrAreanan  GarageYrBltnan
0             0.0            0.0             0.0
1             0.0            0.0             0.0
2             0.0            0.0             0.0
3             0.0            0.0             0.0
4             0.0            0.0             0.0 """

## Capture the dependent feature
y_train=dataset[['SalePrice']]

## drop dependent feature from dataset
X_train=dataset.drop(['Id','SalePrice'],axis=1)

### Apply Feature Selection
# first, I specify the Lasso Regression model, and I
# select a suitable alpha (equivalent of penalty).
# The bigger the alpha the less features that will be selected.

# Then I use the selectFromModel object from sklearn, which
# will select the features which coefficients are non-zero

feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
feature_sel_model.fit(X_train, y_train)
""" SelectFromModel(estimator=Lasso(alpha=0.005, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=0,
   selection='cyclic', tol=0.0001, warm_start=False),
        max_features=None, norm_order=1, prefit=False, threshold=None) """

feature_sel_model.get_support()
""" array([ True,  True, False, False, False, False, False, False, False,
       False, False,  True, False, False, False, False,  True, False,
       False,  True,  True, False, False, False, False, False, False,
       False, False,  True, False,  True, False, False, False, False,
       False, False, False,  True,  True, False,  True, False, False,
        True,  True, False, False, False, False, False,  True, False,
       False,  True,  True,  True, False,  True,  True, False, False,
       False,  True, False, False, False, False, False, False, False,
       False, False, False, False, False, False,  True, False, False,
       False]) """

# let's print the number of total and selected features

# this is how we can make a list of the selected features
selected_feat = X_train.columns[(feature_sel_model.get_support())]

# let's print some stats
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(np.sum(feature_sel_model.estimator_.coef_ == 0)))
""" total features: 82
selected features: 21
features with coefficients shrank to zero: 61 """

selected_feat
""" Index(['MSSubClass', 'MSZoning', 'Neighborhood', 'OverallQual', 'YearRemodAdd',
       'RoofStyle', 'BsmtQual', 'BsmtExposure', 'HeatingQC', 'CentralAir',
       '1stFlrSF', 'GrLivArea', 'BsmtFullBath', 'KitchenQual', 'Fireplaces',
       'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCars', 'PavedDrive',
       'SaleCondition'],
      dtype='object') """

X_train=X_train[selected_feat]
X_train.head()
""" 
   MSSubClass  MSZoning  Neighborhood  OverallQual  YearRemodAdd  RoofStyle  \
0    0.235294      0.75      0.636364     0.666667      0.098361        0.0
1    0.000000      0.75      0.500000     0.555556      0.524590        0.0
2    0.235294      0.75      0.636364     0.666667      0.114754        0.0
3    0.294118      0.75      0.727273     0.666667      0.606557        0.0
4    0.235294      0.75      1.000000     0.777778      0.147541        0.0

   BsmtQual  BsmtExposure  HeatingQC  CentralAir  1stFlrSF  GrLivArea  \
0      0.75          0.25       1.00         1.0  0.356155   0.577712
1      0.75          1.00       1.00         1.0  0.503056   0.470245
2      0.75          0.50       1.00         1.0  0.383441   0.593095
3      0.50          0.25       0.75         1.0  0.399941   0.579157
4      0.75          0.75       1.00         1.0  0.466237   0.666523

   BsmtFullBath  KitchenQual  Fireplaces  FireplaceQu  GarageType  \
0      0.333333     0.666667    0.000000          0.2         0.8
1      0.000000     0.333333    0.333333          0.6         0.8
2      0.333333     0.666667    0.333333          0.6         0.8
3      0.333333     0.666667    0.333333          0.8         0.4
4      0.333333     0.666667    0.333333          0.6         0.8

   GarageFinish  GarageCars  PavedDrive  SaleCondition
0      0.666667        0.50         1.0           0.75
1      0.666667        0.50         1.0           0.75
2      0.666667        0.50         1.0           0.75
3      0.333333        0.75         1.0           0.00
4      0.666667        0.75         1.0           0.75 """