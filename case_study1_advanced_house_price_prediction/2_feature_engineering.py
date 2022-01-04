""" 
We will be performing all the below steps in Feature Engineering:
1. Missing values
2. Temporal variables
3. Categorical variables: remove rare labels
4. Standarise the values of the variables to the same range """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)

dataset=pd.read_csv('case_study1_advanced_house_price_prediction/train.csv')

dataset.head()
""" 
   Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \
0   1          60       RL         65.0     8450   Pave   NaN      Reg
1   2          20       RL         80.0     9600   Pave   NaN      Reg
2   3          60       RL         68.0    11250   Pave   NaN      IR1
3   4          70       RL         60.0     9550   Pave   NaN      IR1
4   5          60       RL         84.0    14260   Pave   NaN      IR1

  LandContour Utilities LotConfig LandSlope Neighborhood Condition1  \
0         Lvl    AllPub    Inside       Gtl      CollgCr       Norm
1         Lvl    AllPub       FR2       Gtl      Veenker      Feedr
2         Lvl    AllPub    Inside       Gtl      CollgCr       Norm
3         Lvl    AllPub    Corner       Gtl      Crawfor       Norm
4         Lvl    AllPub       FR2       Gtl      NoRidge       Norm

  Condition2 BldgType HouseStyle  OverallQual  OverallCond  YearBuilt  \
0       Norm     1Fam     2Story            7            5       2003
1       Norm     1Fam     1Story            6            8       1976
2       Norm     1Fam     2Story            7            5       2001
3       Norm     1Fam     2Story            7            5       1915
4       Norm     1Fam     2Story            8            5       2000

   YearRemodAdd RoofStyle RoofMatl Exterior1st Exterior2nd MasVnrType  \
0          2003     Gable  CompShg     VinylSd     VinylSd    BrkFace
1          1976     Gable  CompShg     MetalSd     MetalSd       None
2          2002     Gable  CompShg     VinylSd     VinylSd    BrkFace
3          1970     Gable  CompShg     Wd Sdng     Wd Shng       None
4          2000     Gable  CompShg     VinylSd     VinylSd    BrkFace

   MasVnrArea ExterQual ExterCond Foundation BsmtQual BsmtCond BsmtExposure  \
0       196.0        Gd        TA      PConc       Gd       TA           No
1         0.0        TA        TA     CBlock       Gd       TA           Gd
2       162.0        Gd        TA      PConc       Gd       TA           Mn
3         0.0        TA        TA     BrkTil       TA       Gd           No
4       350.0        Gd        TA      PConc       Gd       TA           Av

  BsmtFinType1  BsmtFinSF1 BsmtFinType2  BsmtFinSF2  BsmtUnfSF  TotalBsmtSF  \
0          GLQ         706          Unf           0        150          856
1          ALQ         978          Unf           0        284         1262
2          GLQ         486          Unf           0        434          920
3          ALQ         216          Unf           0        540          756
4          GLQ         655          Unf           0        490         1145

  Heating HeatingQC CentralAir Electrical  1stFlrSF  2ndFlrSF  LowQualFinSF  \
0    GasA        Ex          Y      SBrkr       856       854             0
1    GasA        Ex          Y      SBrkr      1262         0             0
2    GasA        Ex          Y      SBrkr       920       866             0
3    GasA        Gd          Y      SBrkr       961       756             0
4    GasA        Ex          Y      SBrkr      1145      1053             0

   GrLivArea  BsmtFullBath  BsmtHalfBath  FullBath  HalfBath  BedroomAbvGr  \
0       1710             1             0         2         1             3
1       1262             0             1         2         0             3
2       1786             1             0         2         1             3
3       1717             1             0         1         0             3
4       2198             1             0         2         1             4

   KitchenAbvGr KitchenQual  TotRmsAbvGrd Functional  Fireplaces FireplaceQu  \
0             1          Gd             8        Typ           0         NaN
1             1          TA             6        Typ           1          TA
2             1          Gd             6        Typ           1          TA
3             1          Gd             7        Typ           1          Gd
4             1          Gd             9        Typ           1          TA

  GarageType  GarageYrBlt GarageFinish  GarageCars  GarageArea GarageQual  \
0     Attchd       2003.0          RFn           2         548         TA
1     Attchd       1976.0          RFn           2         460         TA
2     Attchd       2001.0          RFn           2         608         TA
3     Detchd       1998.0          Unf           3         642         TA
4     Attchd       2000.0          RFn           3         836         TA

  GarageCond PavedDrive  WoodDeckSF  OpenPorchSF  EnclosedPorch  3SsnPorch  \
0         TA          Y           0           61              0          0
1         TA          Y         298            0              0          0
2         TA          Y           0           42              0          0
3         TA          Y           0           35            272          0
4         TA          Y         192           84              0          0

   ScreenPorch  PoolArea PoolQC Fence MiscFeature  MiscVal  MoSold  YrSold  \
0            0         0    NaN   NaN         NaN        0       2    2008
1            0         0    NaN   NaN         NaN        0       5    2007
2            0         0    NaN   NaN         NaN        0       9    2008
3            0         0    NaN   NaN         NaN        0       2    2006
4            0         0    NaN   NaN         NaN        0      12    2008

  SaleType SaleCondition  SalePrice
0       WD        Normal     208500
1       WD        Normal     181500
2       WD        Normal     223500
3       WD       Abnorml     140000
4       WD        Normal     250000 """

## Always remember there way always be a chance of data leakage so we need to split the data first and then apply 
# feature Engineering
X_train,X_test,y_train,y_test=train_test_split(dataset,dataset['SalePrice'],test_size=0.1,random_state=0)
X_train.shape, X_test.shape
# ((1314, 81), (146, 81))

""" 1. Missing Values """

## Let us capture all the nan values
## First lets handle Categorical features which are missing
features_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes=='O']

for feature in features_nan:
    print("{}: {}% missing values".format(feature,np.round(dataset[feature].isnull().mean(),4)))
""" 
Alley: 0.9377% missing values
MasVnrType: 0.0055% missing values
BsmtQual: 0.0253% missing values
BsmtCond: 0.0253% missing values
BsmtExposure: 0.026% missing values
BsmtFinType1: 0.0253% missing values
BsmtFinType2: 0.026% missing values
FireplaceQu: 0.4726% missing values
GarageType: 0.0555% missing values
GarageFinish: 0.0555% missing values
GarageQual: 0.0555% missing values
GarageCond: 0.0555% missing values
PoolQC: 0.9952% missing values
Fence: 0.8075% missing values
MiscFeature: 0.963% missing values """

## Replace missing value with a new label
def replace_cat_feature(dataset,features_nan):
    data=dataset.copy()
    data[features_nan]=data[features_nan].fillna('Missing')
    return data

dataset=replace_cat_feature(dataset,features_nan)

dataset[features_nan].isnull().sum()
""" Alley           0
MasVnrType      0
BsmtQual        0
BsmtCond        0
BsmtExposure    0
BsmtFinType1    0
BsmtFinType2    0
FireplaceQu     0
GarageType      0
GarageFinish    0
GarageQual      0
GarageCond      0
PoolQC          0
Fence           0
MiscFeature     0
dtype: int64 """

dataset.head()
""" 
   Id  MSSubClass MSZoning  LotFrontage  LotArea Street    Alley LotShape  \
0   1          60       RL         65.0     8450   Pave  Missing      Reg
1   2          20       RL         80.0     9600   Pave  Missing      Reg
2   3          60       RL         68.0    11250   Pave  Missing      IR1
3   4          70       RL         60.0     9550   Pave  Missing      IR1
4   5          60       RL         84.0    14260   Pave  Missing      IR1

  LandContour Utilities LotConfig LandSlope Neighborhood Condition1  \
0         Lvl    AllPub    Inside       Gtl      CollgCr       Norm
1         Lvl    AllPub       FR2       Gtl      Veenker      Feedr
2         Lvl    AllPub    Inside       Gtl      CollgCr       Norm
3         Lvl    AllPub    Corner       Gtl      Crawfor       Norm
4         Lvl    AllPub       FR2       Gtl      NoRidge       Norm

  Condition2 BldgType HouseStyle  OverallQual  OverallCond  YearBuilt  \
0       Norm     1Fam     2Story            7            5       2003
1       Norm     1Fam     1Story            6            8       1976
2       Norm     1Fam     2Story            7            5       2001
3       Norm     1Fam     2Story            7            5       1915
4       Norm     1Fam     2Story            8            5       2000

   YearRemodAdd RoofStyle RoofMatl Exterior1st Exterior2nd MasVnrType  \
0          2003     Gable  CompShg     VinylSd     VinylSd    BrkFace
1          1976     Gable  CompShg     MetalSd     MetalSd       None
2          2002     Gable  CompShg     VinylSd     VinylSd    BrkFace
3          1970     Gable  CompShg     Wd Sdng     Wd Shng       None
4          2000     Gable  CompShg     VinylSd     VinylSd    BrkFace

   MasVnrArea ExterQual ExterCond Foundation BsmtQual BsmtCond BsmtExposure  \
0       196.0        Gd        TA      PConc       Gd       TA           No
1         0.0        TA        TA     CBlock       Gd       TA           Gd
2       162.0        Gd        TA      PConc       Gd       TA           Mn
3         0.0        TA        TA     BrkTil       TA       Gd           No
4       350.0        Gd        TA      PConc       Gd       TA           Av

  BsmtFinType1  BsmtFinSF1 BsmtFinType2  BsmtFinSF2  BsmtUnfSF  TotalBsmtSF  \
0          GLQ         706          Unf           0        150          856
1          ALQ         978          Unf           0        284         1262
2          GLQ         486          Unf           0        434          920
3          ALQ         216          Unf           0        540          756
4          GLQ         655          Unf           0        490         1145

  Heating HeatingQC CentralAir Electrical  1stFlrSF  2ndFlrSF  LowQualFinSF  \
0    GasA        Ex          Y      SBrkr       856       854             0
1    GasA        Ex          Y      SBrkr      1262         0             0
2    GasA        Ex          Y      SBrkr       920       866             0
3    GasA        Gd          Y      SBrkr       961       756             0
4    GasA        Ex          Y      SBrkr      1145      1053             0

   GrLivArea  BsmtFullBath  BsmtHalfBath  FullBath  HalfBath  BedroomAbvGr  \
0       1710             1             0         2         1             3
1       1262             0             1         2         0             3
2       1786             1             0         2         1             3
3       1717             1             0         1         0             3
4       2198             1             0         2         1             4

   KitchenAbvGr KitchenQual  TotRmsAbvGrd Functional  Fireplaces FireplaceQu  \
0             1          Gd             8        Typ           0     Missing
1             1          TA             6        Typ           1          TA
2             1          Gd             6        Typ           1          TA
3             1          Gd             7        Typ           1          Gd
4             1          Gd             9        Typ           1          TA

  GarageType  GarageYrBlt GarageFinish  GarageCars  GarageArea GarageQual  \
0     Attchd       2003.0          RFn           2         548         TA
1     Attchd       1976.0          RFn           2         460         TA
2     Attchd       2001.0          RFn           2         608         TA
3     Detchd       1998.0          Unf           3         642         TA
4     Attchd       2000.0          RFn           3         836         TA

  GarageCond PavedDrive  WoodDeckSF  OpenPorchSF  EnclosedPorch  3SsnPorch  \
0         TA          Y           0           61              0          0
1         TA          Y         298            0              0          0
2         TA          Y           0           42              0          0
3         TA          Y           0           35            272          0
4         TA          Y         192           84              0          0

   ScreenPorch  PoolArea   PoolQC    Fence MiscFeature  MiscVal  MoSold  \
0            0         0  Missing  Missing     Missing        0       2
1            0         0  Missing  Missing     Missing        0       5
2            0         0  Missing  Missing     Missing        0       9
3            0         0  Missing  Missing     Missing        0       2
4            0         0  Missing  Missing     Missing        0      12

   YrSold SaleType SaleCondition  SalePrice
0    2008       WD        Normal     208500
1    2007       WD        Normal     181500
2    2008       WD        Normal     223500
3    2006       WD       Abnorml     140000
4    2008       WD        Normal     250000
>>> dataset.head()
   Id  MSSubClass MSZoning  LotFrontage  LotArea Street    Alley LotShape  \
0   1          60       RL         65.0     8450   Pave  Missing      Reg
1   2          20       RL         80.0     9600   Pave  Missing      Reg
2   3          60       RL         68.0    11250   Pave  Missing      IR1
3   4          70       RL         60.0     9550   Pave  Missing      IR1
4   5          60       RL         84.0    14260   Pave  Missing      IR1

  LandContour Utilities LotConfig LandSlope Neighborhood Condition1  \
0         Lvl    AllPub    Inside       Gtl      CollgCr       Norm
1         Lvl    AllPub       FR2       Gtl      Veenker      Feedr
2         Lvl    AllPub    Inside       Gtl      CollgCr       Norm
3         Lvl    AllPub    Corner       Gtl      Crawfor       Norm
4         Lvl    AllPub       FR2       Gtl      NoRidge       Norm

  Condition2 BldgType HouseStyle  OverallQual  OverallCond  YearBuilt  \
0       Norm     1Fam     2Story            7            5       2003
1       Norm     1Fam     1Story            6            8       1976
2       Norm     1Fam     2Story            7            5       2001
3       Norm     1Fam     2Story            7            5       1915
4       Norm     1Fam     2Story            8            5       2000

   YearRemodAdd RoofStyle RoofMatl Exterior1st Exterior2nd MasVnrType  \
0          2003     Gable  CompShg     VinylSd     VinylSd    BrkFace
1          1976     Gable  CompShg     MetalSd     MetalSd       None
2          2002     Gable  CompShg     VinylSd     VinylSd    BrkFace
3          1970     Gable  CompShg     Wd Sdng     Wd Shng       None
4          2000     Gable  CompShg     VinylSd     VinylSd    BrkFace

   MasVnrArea ExterQual ExterCond Foundation BsmtQual BsmtCond BsmtExposure  \
0       196.0        Gd        TA      PConc       Gd       TA           No
1         0.0        TA        TA     CBlock       Gd       TA           Gd
2       162.0        Gd        TA      PConc       Gd       TA           Mn
3         0.0        TA        TA     BrkTil       TA       Gd           No
4       350.0        Gd        TA      PConc       Gd       TA           Av

  BsmtFinType1  BsmtFinSF1 BsmtFinType2  BsmtFinSF2  BsmtUnfSF  TotalBsmtSF  \
0          GLQ         706          Unf           0        150          856
1          ALQ         978          Unf           0        284         1262
2          GLQ         486          Unf           0        434          920
3          ALQ         216          Unf           0        540          756
4          GLQ         655          Unf           0        490         1145

  Heating HeatingQC CentralAir Electrical  1stFlrSF  2ndFlrSF  LowQualFinSF  \
0    GasA        Ex          Y      SBrkr       856       854             0
1    GasA        Ex          Y      SBrkr      1262         0             0
2    GasA        Ex          Y      SBrkr       920       866             0
3    GasA        Gd          Y      SBrkr       961       756             0
4    GasA        Ex          Y      SBrkr      1145      1053             0

   GrLivArea  BsmtFullBath  BsmtHalfBath  FullBath  HalfBath  BedroomAbvGr  \
0       1710             1             0         2         1             3
1       1262             0             1         2         0             3
2       1786             1             0         2         1             3
3       1717             1             0         1         0             3
4       2198             1             0         2         1             4

   KitchenAbvGr KitchenQual  TotRmsAbvGrd Functional  Fireplaces FireplaceQu  \
0             1          Gd             8        Typ           0     Missing
1             1          TA             6        Typ           1          TA
2             1          Gd             6        Typ           1          TA
3             1          Gd             7        Typ           1          Gd
4             1          Gd             9        Typ           1          TA

  GarageType  GarageYrBlt GarageFinish  GarageCars  GarageArea GarageQual  \
0     Attchd       2003.0          RFn           2         548         TA
1     Attchd       1976.0          RFn           2         460         TA
2     Attchd       2001.0          RFn           2         608         TA
3     Detchd       1998.0          Unf           3         642         TA
4     Attchd       2000.0          RFn           3         836         TA

  GarageCond PavedDrive  WoodDeckSF  OpenPorchSF  EnclosedPorch  3SsnPorch  \
0         TA          Y           0           61              0          0
1         TA          Y         298            0              0          0
2         TA          Y           0           42              0          0
3         TA          Y           0           35            272          0
4         TA          Y         192           84              0          0

   ScreenPorch  PoolArea   PoolQC    Fence MiscFeature  MiscVal  MoSold  \
0            0         0  Missing  Missing     Missing        0       2
1            0         0  Missing  Missing     Missing        0       5
2            0         0  Missing  Missing     Missing        0       9
3            0         0  Missing  Missing     Missing        0       2
4            0         0  Missing  Missing     Missing        0      12

   YrSold SaleType SaleCondition  SalePrice
0    2008       WD        Normal     208500
1    2007       WD        Normal     181500
2    2008       WD        Normal     223500
3    2006       WD       Abnorml     140000
4    2008       WD        Normal     250000 """

## Now lets check for numerical variables the contains missing values
numerical_with_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes!='O']

## We will print the numerical nan variables and percentage of missing values

for feature in numerical_with_nan:
    print("{}: {}% missing value".format(feature,np.around(dataset[feature].isnull().mean(),4)))
""" 
LotFrontage: 0.1774% missing value
MasVnrArea: 0.0055% missing value
GarageYrBlt: 0.0555% missing value """

## Replacing the numerical Missing Values

for feature in numerical_with_nan:
    ## We will replace by using median since there are outliers
    median_value=dataset[feature].median()
    
    ## create a new feature to capture nan values
    dataset[feature+'nan']=np.where(dataset[feature].isnull(),1,0)
    dataset[feature].fillna(median_value,inplace=True)
    
dataset[numerical_with_nan].isnull().sum()
""" 
LotFrontage    0
MasVnrArea     0
GarageYrBlt    0
dtype: int64  """

dataset.head(50)
""" 
    Id  MSSubClass MSZoning  LotFrontage  LotArea Street    Alley LotShape  \
0    1          60       RL         65.0     8450   Pave  Missing      Reg
1    2          20       RL         80.0     9600   Pave  Missing      Reg
2    3          60       RL         68.0    11250   Pave  Missing      IR1
3    4          70       RL         60.0     9550   Pave  Missing      IR1
4    5          60       RL         84.0    14260   Pave  Missing      IR1
5    6          50       RL         85.0    14115   Pave  Missing      IR1
6    7          20       RL         75.0    10084   Pave  Missing      Reg
7    8          60       RL         69.0    10382   Pave  Missing      IR1
8    9          50       RM         51.0     6120   Pave  Missing      Reg
9   10         190       RL         50.0     7420   Pave  Missing      Reg
10  11          20       RL         70.0    11200   Pave  Missing      Reg
11  12          60       RL         85.0    11924   Pave  Missing      IR1
12  13          20       RL         69.0    12968   Pave  Missing      IR2
13  14          20       RL         91.0    10652   Pave  Missing      IR1
14  15          20       RL         69.0    10920   Pave  Missing      IR1
15  16          45       RM         51.0     6120   Pave  Missing      Reg
16  17          20       RL         69.0    11241   Pave  Missing      IR1
17  18          90       RL         72.0    10791   Pave  Missing      Reg
18  19          20       RL         66.0    13695   Pave  Missing      Reg
19  20          20       RL         70.0     7560   Pave  Missing      Reg
20  21          60       RL        101.0    14215   Pave  Missing      IR1
21  22          45       RM         57.0     7449   Pave     Grvl      Reg
22  23          20       RL         75.0     9742   Pave  Missing      Reg
23  24         120       RM         44.0     4224   Pave  Missing      Reg
24  25          20       RL         69.0     8246   Pave  Missing      IR1
25  26          20       RL        110.0    14230   Pave  Missing      Reg
26  27          20       RL         60.0     7200   Pave  Missing      Reg
27  28          20       RL         98.0    11478   Pave  Missing      Reg
28  29          20       RL         47.0    16321   Pave  Missing      IR1
29  30          30       RM         60.0     6324   Pave  Missing      IR1
30  31          70  C (all)         50.0     8500   Pave     Pave      Reg
31  32          20       RL         69.0     8544   Pave  Missing      IR1
32  33          20       RL         85.0    11049   Pave  Missing      Reg
33  34          20       RL         70.0    10552   Pave  Missing      IR1
34  35         120       RL         60.0     7313   Pave  Missing      Reg
35  36          60       RL        108.0    13418   Pave  Missing      Reg
36  37          20       RL        112.0    10859   Pave  Missing      Reg
37  38          20       RL         74.0     8532   Pave  Missing      Reg
38  39          20       RL         68.0     7922   Pave  Missing      Reg
39  40          90       RL         65.0     6040   Pave  Missing      Reg
40  41          20       RL         84.0     8658   Pave  Missing      Reg
41  42          20       RL        115.0    16905   Pave  Missing      Reg
42  43          85       RL         69.0     9180   Pave  Missing      IR1
43  44          20       RL         69.0     9200   Pave  Missing      IR1
44  45          20       RL         70.0     7945   Pave  Missing      Reg
45  46         120       RL         61.0     7658   Pave  Missing      Reg
46  47          50       RL         48.0    12822   Pave  Missing      IR1
47  48          20       FV         84.0    11096   Pave  Missing      Reg
48  49         190       RM         33.0     4456   Pave  Missing      Reg
49  50          20       RL         66.0     7742   Pave  Missing      Reg

   LandContour Utilities LotConfig LandSlope Neighborhood Condition1  \
0          Lvl    AllPub    Inside       Gtl      CollgCr       Norm
1          Lvl    AllPub       FR2       Gtl      Veenker      Feedr
2          Lvl    AllPub    Inside       Gtl      CollgCr       Norm
3          Lvl    AllPub    Corner       Gtl      Crawfor       Norm
4          Lvl    AllPub       FR2       Gtl      NoRidge       Norm
5          Lvl    AllPub    Inside       Gtl      Mitchel       Norm
6          Lvl    AllPub    Inside       Gtl      Somerst       Norm
7          Lvl    AllPub    Corner       Gtl       NWAmes       PosN
8          Lvl    AllPub    Inside       Gtl      OldTown     Artery
9          Lvl    AllPub    Corner       Gtl      BrkSide     Artery
10         Lvl    AllPub    Inside       Gtl       Sawyer       Norm
11         Lvl    AllPub    Inside       Gtl      NridgHt       Norm
12         Lvl    AllPub    Inside       Gtl       Sawyer       Norm
13         Lvl    AllPub    Inside       Gtl      CollgCr       Norm
14         Lvl    AllPub    Corner       Gtl        NAmes       Norm
15         Lvl    AllPub    Corner       Gtl      BrkSide       Norm
16         Lvl    AllPub   CulDSac       Gtl        NAmes       Norm
17         Lvl    AllPub    Inside       Gtl       Sawyer       Norm
18         Lvl    AllPub    Inside       Gtl      SawyerW       RRAe
19         Lvl    AllPub    Inside       Gtl        NAmes       Norm
20         Lvl    AllPub    Corner       Gtl      NridgHt       Norm
21         Bnk    AllPub    Inside       Gtl       IDOTRR       Norm
22         Lvl    AllPub    Inside       Gtl      CollgCr       Norm
23         Lvl    AllPub    Inside       Gtl      MeadowV       Norm
24         Lvl    AllPub    Inside       Gtl       Sawyer       Norm
25         Lvl    AllPub    Corner       Gtl      NridgHt       Norm
26         Lvl    AllPub    Corner       Gtl        NAmes       Norm
27         Lvl    AllPub    Inside       Gtl      NridgHt       Norm
28         Lvl    AllPub   CulDSac       Gtl        NAmes       Norm
29         Lvl    AllPub    Inside       Gtl      BrkSide      Feedr
30         Lvl    AllPub    Inside       Gtl       IDOTRR      Feedr
31         Lvl    AllPub   CulDSac       Gtl       Sawyer       Norm
32         Lvl    AllPub    Corner       Gtl      CollgCr       Norm
33         Lvl    AllPub    Inside       Gtl        NAmes       Norm
34         Lvl    AllPub    Inside       Gtl      NridgHt       Norm
35         Lvl    AllPub    Inside       Gtl      NridgHt       Norm
36         Lvl    AllPub    Corner       Gtl      CollgCr       Norm
37         Lvl    AllPub    Inside       Gtl        NAmes       Norm
38         Lvl    AllPub    Inside       Gtl        NAmes       Norm
39         Lvl    AllPub    Inside       Gtl      Edwards       Norm
40         Lvl    AllPub    Inside       Gtl        NAmes       Norm
41         Lvl    AllPub    Inside       Gtl       Timber       Norm
42         Lvl    AllPub   CulDSac       Gtl      SawyerW       Norm
43         Lvl    AllPub   CulDSac       Gtl      CollgCr       Norm
44         Lvl    AllPub    Inside       Gtl        NAmes       Norm
45         Lvl    AllPub    Inside       Gtl      NridgHt       Norm
46         Lvl    AllPub   CulDSac       Gtl      Mitchel       Norm
47         Lvl    AllPub    Inside       Gtl      Somerst       Norm
48         Lvl    AllPub    Inside       Gtl      OldTown       Norm
49         Lvl    AllPub    Inside       Gtl       Sawyer       Norm

   Condition2 BldgType HouseStyle  OverallQual  OverallCond  YearBuilt  \
0        Norm     1Fam     2Story            7            5       2003
1        Norm     1Fam     1Story            6            8       1976
2        Norm     1Fam     2Story            7            5       2001
3        Norm     1Fam     2Story            7            5       1915
4        Norm     1Fam     2Story            8            5       2000
5        Norm     1Fam     1.5Fin            5            5       1993
6        Norm     1Fam     1Story            8            5       2004
7        Norm     1Fam     2Story            7            6       1973
8        Norm     1Fam     1.5Fin            7            5       1931
9      Artery   2fmCon     1.5Unf            5            6       1939
10       Norm     1Fam     1Story            5            5       1965
11       Norm     1Fam     2Story            9            5       2005
12       Norm     1Fam     1Story            5            6       1962
13       Norm     1Fam     1Story            7            5       2006
14       Norm     1Fam     1Story            6            5       1960
15       Norm     1Fam     1.5Unf            7            8       1929
16       Norm     1Fam     1Story            6            7       1970
17       Norm   Duplex     1Story            4            5       1967
18       Norm     1Fam     1Story            5            5       2004
19       Norm     1Fam     1Story            5            6       1958
20       Norm     1Fam     2Story            8            5       2005
21       Norm     1Fam     1.5Unf            7            7       1930
22       Norm     1Fam     1Story            8            5       2002
23       Norm   TwnhsE     1Story            5            7       1976
24       Norm     1Fam     1Story            5            8       1968
25       Norm     1Fam     1Story            8            5       2007
26       Norm     1Fam     1Story            5            7       1951
27       Norm     1Fam     1Story            8            5       2007
28       Norm     1Fam     1Story            5            6       1957
29       RRNn     1Fam     1Story            4            6       1927
30       Norm     1Fam     2Story            4            4       1920
31       Norm     1Fam     1Story            5            6       1966
32       Norm     1Fam     1Story            8            5       2007
33       Norm     1Fam     1Story            5            5       1959
34       Norm   TwnhsE     1Story            9            5       2005
35       Norm     1Fam     2Story            8            5       2004
36       Norm     1Fam     1Story            5            5       1994
37       Norm     1Fam     1Story            5            6       1954
38       Norm     1Fam     1Story            5            7       1953
39       Norm   Duplex     1Story            4            5       1955
40       Norm     1Fam     1Story            6            5       1965
41       Norm     1Fam     1Story            5            6       1959
42       Norm     1Fam     SFoyer            5            7       1983
43       Norm     1Fam     1Story            5            6       1975
44       Norm     1Fam     1Story            5            6       1959
45       Norm   TwnhsE     1Story            9            5       2005
46       Norm     1Fam     1.5Fin            7            5       2003
47       Norm     1Fam     1Story            8            5       2006
48       Norm   2fmCon     2Story            4            5       1920
49       Norm     1Fam     1Story            5            7       1966

    YearRemodAdd RoofStyle RoofMatl Exterior1st Exterior2nd MasVnrType  \
0           2003     Gable  CompShg     VinylSd     VinylSd    BrkFace
1           1976     Gable  CompShg     MetalSd     MetalSd       None
2           2002     Gable  CompShg     VinylSd     VinylSd    BrkFace
3           1970     Gable  CompShg     Wd Sdng     Wd Shng       None
4           2000     Gable  CompShg     VinylSd     VinylSd    BrkFace
5           1995     Gable  CompShg     VinylSd     VinylSd       None
6           2005     Gable  CompShg     VinylSd     VinylSd      Stone
7           1973     Gable  CompShg     HdBoard     HdBoard      Stone
8           1950     Gable  CompShg     BrkFace     Wd Shng       None
9           1950     Gable  CompShg     MetalSd     MetalSd       None
10          1965       Hip  CompShg     HdBoard     HdBoard       None
11          2006       Hip  CompShg     WdShing     Wd Shng      Stone
12          1962       Hip  CompShg     HdBoard     Plywood       None
13          2007     Gable  CompShg     VinylSd     VinylSd      Stone
14          1960       Hip  CompShg     MetalSd     MetalSd    BrkFace
15          2001     Gable  CompShg     Wd Sdng     Wd Sdng       None
16          1970     Gable  CompShg     Wd Sdng     Wd Sdng    BrkFace
17          1967     Gable  CompShg     MetalSd     MetalSd       None
18          2004     Gable  CompShg     VinylSd     VinylSd       None
19          1965       Hip  CompShg     BrkFace     Plywood       None
20          2006     Gable  CompShg     VinylSd     VinylSd    BrkFace
21          1950     Gable  CompShg     Wd Sdng     Wd Sdng       None
22          2002       Hip  CompShg     VinylSd     VinylSd    BrkFace   
23          1976     Gable  CompShg     CemntBd     CmentBd       None
24          2001     Gable  CompShg     Plywood     Plywood       None
25          2007     Gable  CompShg     VinylSd     VinylSd      Stone
26          2000     Gable  CompShg     Wd Sdng     Wd Sdng       None
27          2008     Gable  CompShg     VinylSd     VinylSd      Stone
28          1997     Gable  CompShg     MetalSd     MetalSd       None
29          1950     Gable  CompShg     MetalSd     MetalSd       None
30          1950   Gambrel  CompShg     BrkFace     BrkFace       None
31          2006     Gable  CompShg     HdBoard     HdBoard       None
32          2007     Gable  CompShg     VinylSd     VinylSd       None
33          1959       Hip  CompShg     BrkFace     BrkFace       None
34          2005       Hip  CompShg     MetalSd     MetalSd    BrkFace
35          2005     Gable  CompShg     VinylSd     VinylSd      Stone
36          1995     Gable  CompShg     VinylSd     VinylSd       None
37          1990       Hip  CompShg     Wd Sdng     Wd Sdng    BrkFace
38          2007     Gable  CompShg     VinylSd     VinylSd       None
39          1955     Gable  CompShg     AsbShng     Plywood       None
40          1965     Gable  CompShg     Wd Sdng     Wd Sdng    BrkFace
41          1959     Gable  CompShg     VinylSd     VinylSd       None
42          1983     Gable  CompShg     HdBoard     HdBoard       None
43          1980       Hip  CompShg     VinylSd     VinylSd       None
44          1959     Gable  CompShg     BrkFace     Wd Sdng       None
45          2005       Hip  CompShg     MetalSd     MetalSd    BrkFace
46          2003     Gable  CompShg     VinylSd     VinylSd       None
47          2006     Gable  CompShg     VinylSd     VinylSd       None
48          2008     Gable  CompShg     MetalSd     MetalSd       None
49          1966     Gable  CompShg     HdBoard     HdBoard       None

    MasVnrArea ExterQual ExterCond Foundation BsmtQual BsmtCond BsmtExposure  \
0        196.0        Gd        TA      PConc       Gd       TA           No
1          0.0        TA        TA     CBlock       Gd       TA           Gd
2        162.0        Gd        TA      PConc       Gd       TA           Mn
3          0.0        TA        TA     BrkTil       TA       Gd           No
4        350.0        Gd        TA      PConc       Gd       TA           Av
5          0.0        TA        TA       Wood       Gd       TA           No
6        186.0        Gd        TA      PConc       Ex       TA           Av
7        240.0        TA        TA     CBlock       Gd       TA           Mn
8          0.0        TA        TA     BrkTil       TA       TA           No
9          0.0        TA        TA     BrkTil       TA       TA           No
10         0.0        TA        TA     CBlock       TA       TA           No
11       286.0        Ex        TA      PConc       Ex       TA           No
12         0.0        TA        TA     CBlock       TA       TA           No
13       306.0        Gd        TA      PConc       Gd       TA           Av
14       212.0        TA        TA     CBlock       TA       TA           No
15         0.0        TA        TA     BrkTil       TA       TA           No
16       180.0        TA        TA     CBlock       TA       TA           No
17         0.0        TA        TA       Slab  Missing  Missing      Missing
18         0.0        TA        TA      PConc       TA       TA           No
19         0.0        TA        TA     CBlock       TA       TA           No
20       380.0        Gd        TA      PConc       Ex       TA           Av
21         0.0        TA        TA      PConc       TA       TA           No
22       281.0        Gd        TA      PConc       Gd       TA           No
23         0.0        TA        TA      PConc       Gd       TA           No
24         0.0        TA        Gd     CBlock       TA       TA           Mn
25       640.0        Gd        TA      PConc       Gd       TA           No
26         0.0        TA        TA     CBlock       TA       TA           Mn
27       200.0        Gd        TA      PConc       Ex       TA           No
28         0.0        TA        TA     CBlock       TA       TA           Gd
29         0.0        TA        TA     BrkTil       TA       TA           No
30         0.0        TA        Fa     BrkTil       TA       TA           No
31         0.0        TA        TA     CBlock       TA       TA           No
32         0.0        Gd        TA      PConc       Ex       TA           Av
33         0.0        TA        TA     CBlock       TA       TA           No
34       246.0        Ex        TA      PConc       Ex       TA           No
35       132.0        Gd        TA      PConc       Ex       TA           Av
36         0.0        TA        TA      PConc       Gd       TA           No
37       650.0        TA        TA     CBlock       TA       TA           No
38         0.0        TA        Gd     CBlock       TA       TA           No
39         0.0        TA        TA      PConc  Missing  Missing      Missing
40       101.0        TA        TA     CBlock       TA       TA           No
41         0.0        TA        Gd     CBlock       TA       TA           Gd
42         0.0        TA        TA     CBlock       Gd       TA           Av
43         0.0        TA        TA     CBlock       Gd       TA           Av
44         0.0        TA        TA     CBlock       TA       TA           No
45       412.0        Ex        TA      PConc       Ex       TA           No
46         0.0        Gd        TA      PConc       Ex       TA           No
47         0.0        Gd        TA      PConc       Gd       TA           Av
48         0.0        TA        TA     BrkTil       TA       TA           No
49         0.0        TA        TA     CBlock       TA       TA           No

   BsmtFinType1  BsmtFinSF1 BsmtFinType2  BsmtFinSF2  BsmtUnfSF  TotalBsmtSF  \
0           GLQ         706          Unf           0        150          856
1           ALQ         978          Unf           0        284         1262
2           GLQ         486          Unf           0        434          920
3           ALQ         216          Unf           0        540          756
4           GLQ         655          Unf           0        490         1145
5           GLQ         732          Unf           0         64          796
6           GLQ        1369          Unf           0        317         1686
7           ALQ         859          BLQ          32        216         1107
8           Unf           0          Unf           0        952          952
9           GLQ         851          Unf           0        140          991
10          Rec         906          Unf           0        134         1040
11          GLQ         998          Unf           0        177         1175
12          ALQ         737          Unf           0        175          912
13          Unf           0          Unf           0       1494         1494
14          BLQ         733          Unf           0        520         1253
15          Unf           0          Unf           0        832          832
16          ALQ         578          Unf           0        426         1004
17      Missing           0      Missing           0          0            0
18          GLQ         646          Unf           0        468         1114
19          LwQ         504          Unf           0        525         1029
20          Unf           0          Unf           0       1158         1158
21          Unf           0          Unf           0        637          637
22          Unf           0          Unf           0       1777         1777
23          GLQ         840          Unf           0        200         1040
24          Rec         188          ALQ         668        204         1060
25          Unf           0          Unf           0       1566         1566
26          BLQ         234          Rec         486        180          900
27          GLQ        1218          Unf           0        486         1704
28          BLQ        1277          Unf           0        207         1484
29          Unf           0          Unf           0        520          520
30          Unf           0          Unf           0        649          649
31          Unf           0          Unf           0       1228         1228
32          Unf           0          Unf           0       1234         1234
33          Rec        1018          Unf           0        380         1398
34          GLQ        1153          Unf           0        408         1561
35          Unf           0          Unf           0       1117         1117
36          Unf           0          Unf           0       1097         1097
37          Rec        1213          Unf           0         84         1297
38          GLQ         731          Unf           0        326         1057
39      Missing           0      Missing           0          0            0
40          Rec         643          Unf           0        445         1088
41          BLQ         967          Unf           0        383         1350
42          ALQ         747          LwQ          93          0          840
43          LwQ         280          BLQ         491        167          938
44          ALQ         179          BLQ         506        465         1150
45          GLQ         456          Unf           0       1296         1752
46          GLQ        1351          Unf           0         83         1434
47          GLQ          24          Unf           0       1632         1656
48          Unf           0          Unf           0        736          736
49          BLQ         763          Unf           0        192          955

   Heating HeatingQC CentralAir Electrical  1stFlrSF  2ndFlrSF  LowQualFinSF  \
0     GasA        Ex          Y      SBrkr       856       854             0
1     GasA        Ex          Y      SBrkr      1262         0             0
2     GasA        Ex          Y      SBrkr       920       866             0
3     GasA        Gd          Y      SBrkr       961       756             0
4     GasA        Ex          Y      SBrkr      1145      1053             0
5     GasA        Ex          Y      SBrkr       796       566             0
6     GasA        Ex          Y      SBrkr      1694         0             0
7     GasA        Ex          Y      SBrkr      1107       983             0
8     GasA        Gd          Y      FuseF      1022       752             0
9     GasA        Ex          Y      SBrkr      1077         0             0
10    GasA        Ex          Y      SBrkr      1040         0             0
11    GasA        Ex          Y      SBrkr      1182      1142             0
12    GasA        TA          Y      SBrkr       912         0             0
13    GasA        Ex          Y      SBrkr      1494         0             0
14    GasA        TA          Y      SBrkr      1253         0             0
15    GasA        Ex          Y      FuseA       854         0             0
16    GasA        Ex          Y      SBrkr      1004         0             0
17    GasA        TA          Y      SBrkr      1296         0             0
18    GasA        Ex          Y      SBrkr      1114         0             0
19    GasA        TA          Y      SBrkr      1339         0             0
20    GasA        Ex          Y      SBrkr      1158      1218             0
21    GasA        Ex          Y      FuseF      1108         0             0
22    GasA        Ex          Y      SBrkr      1795         0             0
23    GasA        TA          Y      SBrkr      1060         0             0
24    GasA        Ex          Y      SBrkr      1060         0             0
25    GasA        Ex          Y      SBrkr      1600         0             0
26    GasA        TA          Y      SBrkr       900         0             0
27    GasA        Ex          Y      SBrkr      1704         0             0
28    GasA        TA          Y      SBrkr      1600         0             0
29    GasA        Fa          N      SBrkr       520         0             0
30    GasA        TA          N      SBrkr       649       668             0
31    GasA        Gd          Y      SBrkr      1228         0             0
32    GasA        Ex          Y      SBrkr      1234         0             0
33    GasA        Gd          Y      SBrkr      1700         0             0
34    GasA        Ex          Y      SBrkr      1561         0             0
35    GasA        Ex          Y      SBrkr      1132      1320             0
36    GasA        Ex          Y      SBrkr      1097         0             0
37    GasA        Gd          Y      SBrkr      1297         0             0
38    GasA        TA          Y      SBrkr      1057         0             0
39    GasA        TA          N      FuseP      1152         0             0
40    GasA        Ex          Y      SBrkr      1324         0             0
41    GasA        Gd          Y      SBrkr      1328         0             0
42    GasA        Gd          Y      SBrkr       884         0             0
43    GasA        TA          Y      SBrkr       938         0             0
44    GasA        Ex          Y      FuseA      1150         0             0
45    GasA        Ex          Y      SBrkr      1752         0             0
46    GasA        Ex          Y      SBrkr      1518       631             0
47    GasA        Ex          Y      SBrkr      1656         0             0
48    GasA        Gd          Y      SBrkr       736       716             0
49    GasA        Ex          Y      SBrkr       955         0             0

    GrLivArea  BsmtFullBath  BsmtHalfBath  FullBath  HalfBath  BedroomAbvGr  \
0        1710             1             0         2         1             3
1        1262             0             1         2         0             3
2        1786             1             0         2         1             3
3        1717             1             0         1         0             3
4        2198             1             0         2         1             4
5        1362             1             0         1         1             1
6        1694             1             0         2         0             3
7        2090             1             0         2         1             3
8        1774             0             0         2         0             2
9        1077             1             0         1         0             2
10       1040             1             0         1         0             3
11       2324             1             0         3         0             4
12        912             1             0         1         0             2
13       1494             0             0         2         0             3
14       1253             1             0         1         1             2
15        854             0             0         1         0             2
16       1004             1             0         1         0             2
17       1296             0             0         2         0             2
18       1114             1             0         1         1             3
19       1339             0             0         1         0             3
20       2376             0             0         3         1             4
21       1108             0             0         1         0             3
22       1795             0             0         2         0             3
23       1060             1             0         1         0             3
24       1060             1             0         1         0             3
25       1600             0             0         2         0             3
26        900             0             1         1         0             3
27       1704             1             0         2         0             3
28       1600             1             0         1         0             2
29        520             0             0         1         0             1
30       1317             0             0         1         0             3
31       1228             0             0         1         1             3
32       1234             0             0         2         0             3
33       1700             0             1         1         1             4
34       1561             1             0         2         0             2
35       2452             0             0         3         1             4
36       1097             0             0         1         1             3
37       1297             0             1         1         0             3
38       1057             1             0         1         0             3
39       1152             0             0         2         0             2
40       1324             0             0         2         0             3
41       1328             0             1         1         1             2
42        884             1             0         1         0             2
43        938             1             0         1         0             3
44       1150             1             0         1         0             3
45       1752             1             0         2         0             2
46       2149             1             0         1         1             1
47       1656             0             0         2         0             3
48       1452             0             0         2         0             2
49        955             1             0         1         0             3

    KitchenAbvGr KitchenQual  TotRmsAbvGrd Functional  Fireplaces FireplaceQu  \
0              1          Gd             8        Typ           0     Missing
1              1          TA             6        Typ           1          TA
2              1          Gd             6        Typ           1          TA
3              1          Gd             7        Typ           1          Gd
4              1          Gd             9        Typ           1          TA
5              1          TA             5        Typ           0     Missing
6              1          Gd             7        Typ           1          Gd
7              1          TA             7        Typ           2          TA
8              2          TA             8       Min1           2          TA
9              2          TA             5        Typ           2          TA
10             1          TA             5        Typ           0     Missing
11             1          Ex            11        Typ           2          Gd
12             1          TA             4        Typ           0     Missing
13             1          Gd             7        Typ           1          Gd
14             1          TA             5        Typ           1          Fa
15             1          TA             5        Typ           0     Missing
16             1          TA             5        Typ           1          TA
17             2          TA             6        Typ           0     Missing
18             1          Gd             6        Typ           0     Missing
19             1          TA             6       Min1           0     Missing
20             1          Gd             9        Typ           1          Gd
21             1          Gd             6        Typ           1          Gd
22             1          Gd             7        Typ           1          Gd
23             1          TA             6        Typ           1          TA
24             1          Gd             6        Typ           1          TA
25             1          Gd             7        Typ           1          Gd
26             1          Gd             5        Typ           0     Missing
27             1          Gd             7        Typ           1          Gd
28             1          TA             6        Typ           2          Gd
29             1          Fa             4        Typ           0     Missing
30             1          TA             6        Typ           0     Missing
31             1          Gd             6        Typ           0     Missing
32             1          Gd             7        Typ           0     Missing
33             1          Gd             6        Typ           1          Gd
34             1          Ex             6        Typ           1          Gd
35             1          Gd             9        Typ           1          Gd
36             1          TA             6        Typ           0     Missing
37             1          TA             5        Typ           1          TA
38             1          Gd             5        Typ           0     Missing
39             2          Fa             6        Typ           0     Missing
40             1          TA             6        Typ           1          TA
41             1          TA             5        Typ           2          Gd
42             1          Gd             5        Typ           0     Missing
43             1          TA             5        Typ           0     Missing
44             1          TA             6        Typ           0     Missing
45             1          Ex             6        Typ           1          Gd
46             1          Gd             6        Typ           1          Ex
47             1          Gd             7        Typ           0     Missing
48             3          TA             8        Typ           0     Missing
49             1          TA             6        Typ           0     Missing

   GarageType  GarageYrBlt GarageFinish  GarageCars  GarageArea GarageQual  \
0      Attchd       2003.0          RFn           2         548         TA
1      Attchd       1976.0          RFn           2         460         TA
2      Attchd       2001.0          RFn           2         608         TA
3      Detchd       1998.0          Unf           3         642         TA
4      Attchd       2000.0          RFn           3         836         TA
5      Attchd       1993.0          Unf           2         480         TA
6      Attchd       2004.0          RFn           2         636         TA
7      Attchd       1973.0          RFn           2         484         TA
8      Detchd       1931.0          Unf           2         468         Fa
9      Attchd       1939.0          RFn           1         205         Gd
10     Detchd       1965.0          Unf           1         384         TA
11    BuiltIn       2005.0          Fin           3         736         TA
12     Detchd       1962.0          Unf           1         352         TA
13     Attchd       2006.0          RFn           3         840         TA
14     Attchd       1960.0          RFn           1         352         TA
15     Detchd       1991.0          Unf           2         576         TA
16     Attchd       1970.0          Fin           2         480         TA
17    CarPort       1967.0          Unf           2         516         TA
18     Detchd       2004.0          Unf           2         576         TA
19     Attchd       1958.0          Unf           1         294         TA
20    BuiltIn       2005.0          RFn           3         853         TA
21     Attchd       1930.0          Unf           1         280         TA
22     Attchd       2002.0          RFn           2         534         TA
23     Attchd       1976.0          Unf           2         572         TA
24     Attchd       1968.0          Unf           1         270         TA
25     Attchd       2007.0          RFn           3         890         TA
26     Detchd       2005.0          Unf           2         576         TA
27     Attchd       2008.0          RFn           3         772         TA   
28     Attchd       1957.0          RFn           1         319         TA
29     Detchd       1920.0          Unf           1         240         Fa
30     Detchd       1920.0          Unf           1         250         TA
31     Attchd       1966.0          Unf           1         271         TA
32     Attchd       2007.0          RFn           2         484         TA
33     Attchd       1959.0          RFn           2         447         TA
34     Attchd       2005.0          Fin           2         556         TA
35    BuiltIn       2004.0          Fin           3         691         TA
36     Attchd       1995.0          Unf           2         672         TA
37     Attchd       1954.0          Fin           2         498         TA
38     Detchd       1953.0          Unf           1         246         TA
39    Missing       1980.0      Missing           0           0    Missing
40     Attchd       1965.0          RFn           2         440         TA
41     Attchd       1959.0          RFn           1         308         TA
42     Attchd       1983.0          RFn           2         504         TA
43     Detchd       1977.0          Unf           1         308         TA
44     Attchd       1959.0          RFn           1         300         TA
45     Attchd       2005.0          RFn           2         576         TA
46     Attchd       2003.0          RFn           2         670         TA
47     Attchd       2006.0          RFn           3         826         TA
48    Missing       1980.0      Missing           0           0    Missing
49     Attchd       1966.0          Unf           1         386         TA

   GarageCond PavedDrive  WoodDeckSF  OpenPorchSF  EnclosedPorch  3SsnPorch  \
0          TA          Y           0           61              0          0
1          TA          Y         298            0              0          0
2          TA          Y           0           42              0          0
3          TA          Y           0           35            272          0
4          TA          Y         192           84              0          0
5          TA          Y          40           30              0        320
6          TA          Y         255           57              0          0
7          TA          Y         235          204            228          0
8          TA          Y          90            0            205          0
9          TA          Y           0            4              0          0
10         TA          Y           0            0              0          0
11         TA          Y         147           21              0          0
12         TA          Y         140            0              0          0
13         TA          Y         160           33              0          0
14         TA          Y           0          213            176          0
15         TA          Y          48          112              0          0
16         TA          Y           0            0              0          0
17         TA          Y           0            0              0          0
18         TA          Y           0          102              0          0
19         TA          Y           0            0              0          0
20         TA          Y         240          154              0          0
21         TA          N           0            0            205          0
22         TA          Y         171          159              0          0
23         TA          Y         100          110              0          0
24         TA          Y         406           90              0          0
25         TA          Y           0           56              0          0
26         TA          Y         222           32              0          0
27         TA          Y           0           50              0          0
28         TA          Y         288          258              0          0
29         TA          Y          49            0             87          0
30         Fa          N           0           54            172          0
31         TA          Y           0           65              0          0
32         TA          Y           0           30              0          0
33         TA          Y           0           38              0          0
34         TA          Y         203           47              0          0
35         TA          Y         113           32              0          0
36         TA          Y         392           64              0          0
37         TA          Y           0            0              0          0
38         TA          Y           0           52              0          0
39    Missing          N           0            0              0          0
40         TA          Y           0          138              0          0
41         TA          P           0          104              0          0
42         Gd          Y         240            0              0          0
43         TA          Y         145            0              0          0
44         TA          Y           0            0              0          0
45         TA          Y         196           82              0          0
46         TA          Y         168           43              0          0
47         TA          Y           0          146              0          0
48    Missing          N           0            0            102          0
49         TA          Y           0            0              0          0

    ScreenPorch  PoolArea   PoolQC    Fence MiscFeature  MiscVal  MoSold  \
0             0         0  Missing  Missing     Missing        0       2
1             0         0  Missing  Missing     Missing        0       5
2             0         0  Missing  Missing     Missing        0       9
3             0         0  Missing  Missing     Missing        0       2
4             0         0  Missing  Missing     Missing        0      12
5             0         0  Missing    MnPrv        Shed      700      10
6             0         0  Missing  Missing     Missing        0       8
7             0         0  Missing  Missing        Shed      350      11
8             0         0  Missing  Missing     Missing        0       4
9             0         0  Missing  Missing     Missing        0       1
10            0         0  Missing  Missing     Missing        0       2
11            0         0  Missing  Missing     Missing        0       7
12          176         0  Missing  Missing     Missing        0       9
13            0         0  Missing  Missing     Missing        0       8
14            0         0  Missing     GdWo     Missing        0       5
15            0         0  Missing    GdPrv     Missing        0       7
16            0         0  Missing  Missing        Shed      700       3
17            0         0  Missing  Missing        Shed      500      10
18            0         0  Missing  Missing     Missing        0       6
19            0         0  Missing    MnPrv     Missing        0       5
20            0         0  Missing  Missing     Missing        0      11
21            0         0  Missing    GdPrv     Missing        0       6
22            0         0  Missing  Missing     Missing        0       9
23            0         0  Missing  Missing     Missing        0       6
24            0         0  Missing    MnPrv     Missing        0       5
25            0         0  Missing  Missing     Missing        0       7
26            0         0  Missing  Missing     Missing        0       5
27            0         0  Missing  Missing     Missing        0       5
28            0         0  Missing  Missing     Missing        0      12
29            0         0  Missing  Missing     Missing        0       5
30            0         0  Missing    MnPrv     Missing        0       7
31            0         0  Missing    MnPrv     Missing        0       6
32            0         0  Missing  Missing     Missing        0       1
33            0         0  Missing  Missing     Missing        0       4
34            0         0  Missing  Missing     Missing        0       8
35            0         0  Missing  Missing     Missing        0       9
36            0         0  Missing  Missing     Missing        0       6
37            0         0  Missing  Missing     Missing        0      10
38            0         0  Missing  Missing     Missing        0       1
39            0         0  Missing  Missing     Missing        0       6
40            0         0  Missing     GdWo     Missing        0      12
41            0         0  Missing  Missing     Missing        0       7
42            0         0  Missing    MnPrv     Missing        0      12
43            0         0  Missing    MnPrv     Missing        0       7
44            0         0  Missing  Missing     Missing        0       5
45            0         0  Missing  Missing     Missing        0       2
46          198         0  Missing  Missing     Missing        0       8
47            0         0  Missing  Missing     Missing        0       7
48            0         0  Missing  Missing     Missing        0       6
49            0         0  Missing    MnPrv     Missing        0       1

    YrSold SaleType SaleCondition  SalePrice  LotFrontagenan  MasVnrAreanan  \
0     2008       WD        Normal     208500               0              0
1     2007       WD        Normal     181500               0              0
2     2008       WD        Normal     223500               0              0
3     2006       WD       Abnorml     140000               0              0
4     2008       WD        Normal     250000               0              0
5     2009       WD        Normal     143000               0              0
6     2007       WD        Normal     307000               0              0
7     2009       WD        Normal     200000               1              0
8     2008       WD       Abnorml     129900               0              0
9     2008       WD        Normal     118000               0              0
10    2008       WD        Normal     129500               0              0
11    2006      New       Partial     345000               0              0
12    2008       WD        Normal     144000               1              0
13    2007      New       Partial     279500               0              0
14    2008       WD        Normal     157000               1              0
15    2007       WD        Normal     132000               0              0
16    2010       WD        Normal     149000               1              0
17    2006       WD        Normal      90000               0              0
18    2008       WD        Normal     159000               0              0
19    2009      COD       Abnorml     139000               0              0
20    2006      New       Partial     325300               0              0
21    2007       WD        Normal     139400               0              0
22    2008       WD        Normal     230000               0              0
23    2007       WD        Normal     129900               0              0
24    2010       WD        Normal     154000               1              0
25    2009       WD        Normal     256300               0              0
26    2010       WD        Normal     134800               0              0
27    2010       WD        Normal     306000               0              0
28    2006       WD        Normal     207500               0              0
29    2008       WD        Normal      68500               0              0
30    2008       WD        Normal      40000               0              0
31    2008       WD        Normal     149350               1              0
32    2008       WD        Normal     179900               0              0
33    2010       WD        Normal     165500               0              0
34    2007       WD        Normal     277500               0              0
35    2006       WD        Normal     309000               0              0
36    2009       WD        Normal     145000               0              0
37    2009       WD        Normal     153000               0              0
38    2010       WD       Abnorml     109000               0              0
39    2008       WD       AdjLand      82000               0              0
40    2006       WD       Abnorml     160000               0              0
41    2007       WD        Normal     170000               0              0
42    2007       WD        Normal     144000               1              0
43    2008       WD        Normal     130250               1              0
44    2006       WD        Normal     141000               0              0
45    2010       WD        Normal     319900               0              0
46    2009       WD       Abnorml     239686               0              0
47    2007       WD        Normal     249700               0              0
48    2009      New       Partial     113000               0              0
49    2007       WD        Normal     127000               0              0

    GarageYrBltnan
0                0
1                0
2                0
3                0
4                0
5                0
6                0
7                0
8                0
9                0
10               0
11               0
12               0
13               0
14               0
15               0
16               0
17               0
18               0
19               0
20               0
21               0
22               0
23               0
24               0
25               0
26               0
27               0
28               0
29               0
30               0
31               0
32               0
33               0
34               0
35               0
36               0
37               0
38               0
39               1
40               0
41               0
42               0
43               0
44               0
45               0
46               0
47               0
48               1
49               0 """

""" 2. Temporal Variables (Date Time Variables) """

for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    dataset[feature]=dataset['YrSold']-dataset[feature]

dataset.head()
""" 
   Id  MSSubClass MSZoning  LotFrontage  LotArea Street    Alley LotShape  \
0   1          60       RL         65.0     8450   Pave  Missing      Reg
1   2          20       RL         80.0     9600   Pave  Missing      Reg
2   3          60       RL         68.0    11250   Pave  Missing      IR1
3   4          70       RL         60.0     9550   Pave  Missing      IR1
4   5          60       RL         84.0    14260   Pave  Missing      IR1

  LandContour Utilities LotConfig LandSlope Neighborhood Condition1  \
0         Lvl    AllPub    Inside       Gtl      CollgCr       Norm
1         Lvl    AllPub       FR2       Gtl      Veenker      Feedr
2         Lvl    AllPub    Inside       Gtl      CollgCr       Norm
3         Lvl    AllPub    Corner       Gtl      Crawfor       Norm
4         Lvl    AllPub       FR2       Gtl      NoRidge       Norm

  Condition2 BldgType HouseStyle  OverallQual  OverallCond  YearBuilt  \
0       Norm     1Fam     2Story            7            5          5
1       Norm     1Fam     1Story            6            8         31
2       Norm     1Fam     2Story            7            5          7
3       Norm     1Fam     2Story            7            5         91
4       Norm     1Fam     2Story            8            5          8

   YearRemodAdd RoofStyle RoofMatl Exterior1st Exterior2nd MasVnrType  \
0             5     Gable  CompShg     VinylSd     VinylSd    BrkFace
1            31     Gable  CompShg     MetalSd     MetalSd       None
2             6     Gable  CompShg     VinylSd     VinylSd    BrkFace
3            36     Gable  CompShg     Wd Sdng     Wd Shng       None
4             8     Gable  CompShg     VinylSd     VinylSd    BrkFace

   MasVnrArea ExterQual ExterCond Foundation BsmtQual BsmtCond BsmtExposure  \
0       196.0        Gd        TA      PConc       Gd       TA           No
1         0.0        TA        TA     CBlock       Gd       TA           Gd
2       162.0        Gd        TA      PConc       Gd       TA           Mn
3         0.0        TA        TA     BrkTil       TA       Gd           No
4       350.0        Gd        TA      PConc       Gd       TA           Av

  BsmtFinType1  BsmtFinSF1 BsmtFinType2  BsmtFinSF2  BsmtUnfSF  TotalBsmtSF  \
0          GLQ         706          Unf           0        150          856
1          ALQ         978          Unf           0        284         1262
2          GLQ         486          Unf           0        434          920
3          ALQ         216          Unf           0        540          756
4          GLQ         655          Unf           0        490         1145

  Heating HeatingQC CentralAir Electrical  1stFlrSF  2ndFlrSF  LowQualFinSF  \
0    GasA        Ex          Y      SBrkr       856       854             0
1    GasA        Ex          Y      SBrkr      1262         0             0
2    GasA        Ex          Y      SBrkr       920       866             0
3    GasA        Gd          Y      SBrkr       961       756             0
4    GasA        Ex          Y      SBrkr      1145      1053             0

   GrLivArea  BsmtFullBath  BsmtHalfBath  FullBath  HalfBath  BedroomAbvGr  \
0       1710             1             0         2         1             3
1       1262             0             1         2         0             3
2       1786             1             0         2         1             3
3       1717             1             0         1         0             3
4       2198             1             0         2         1             4

   KitchenAbvGr KitchenQual  TotRmsAbvGrd Functional  Fireplaces FireplaceQu  \
0             1          Gd             8        Typ           0     Missing
1             1          TA             6        Typ           1          TA
2             1          Gd             6        Typ           1          TA
3             1          Gd             7        Typ           1          Gd
4             1          Gd             9        Typ           1          TA

  GarageType  GarageYrBlt GarageFinish  GarageCars  GarageArea GarageQual  \
0     Attchd          5.0          RFn           2         548         TA
1     Attchd         31.0          RFn           2         460         TA
2     Attchd          7.0          RFn           2         608         TA
3     Detchd          8.0          Unf           3         642         TA
4     Attchd          8.0          RFn           3         836         TA

  GarageCond PavedDrive  WoodDeckSF  OpenPorchSF  EnclosedPorch  3SsnPorch  \
0         TA          Y           0           61              0          0
1         TA          Y         298            0              0          0
2         TA          Y           0           42              0          0
3         TA          Y           0           35            272          0
4         TA          Y         192           84              0          0

   ScreenPorch  PoolArea   PoolQC    Fence MiscFeature  MiscVal  MoSold  \
0            0         0  Missing  Missing     Missing        0       2
1            0         0  Missing  Missing     Missing        0       5
2            0         0  Missing  Missing     Missing        0       9
3            0         0  Missing  Missing     Missing        0       2
4            0         0  Missing  Missing     Missing        0      12

   YrSold SaleType SaleCondition  SalePrice  LotFrontagenan  MasVnrAreanan  \
0    2008       WD        Normal     208500               0              0
1    2007       WD        Normal     181500               0              0
2    2008       WD        Normal     223500               0              0
3    2006       WD       Abnorml     140000               0              0
4    2008       WD        Normal     250000               0              0

   GarageYrBltnan
0               0
1               0
2               0
3               0
4               0 """

dataset[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()
""" 
    YearBuilt  YearRemodAdd  GarageYrBlt
0          5             5          5.0
1         31            31         31.0
2          7             6          7.0
3         91            36          8.0
4          8             8          8.0 """

""" 
Numerical Variables
- Since the numerical variables are skewed we will perform log normal distribution
 """

import numpy as np
num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

for feature in num_features:
    dataset[feature]=np.log(dataset[feature])

dataset.head()
""" 
   Id  MSSubClass MSZoning  LotFrontage   LotArea Street    Alley LotShape  \
0   1          60       RL     4.174387  9.041922   Pave  Missing      Reg
1   2          20       RL     4.382027  9.169518   Pave  Missing      Reg
2   3          60       RL     4.219508  9.328123   Pave  Missing      IR1
3   4          70       RL     4.094345  9.164296   Pave  Missing      IR1
4   5          60       RL     4.430817  9.565214   Pave  Missing      IR1

  LandContour Utilities LotConfig LandSlope Neighborhood Condition1  \
0         Lvl    AllPub    Inside       Gtl      CollgCr       Norm
1         Lvl    AllPub       FR2       Gtl      Veenker      Feedr
2         Lvl    AllPub    Inside       Gtl      CollgCr       Norm
3         Lvl    AllPub    Corner       Gtl      Crawfor       Norm
4         Lvl    AllPub       FR2       Gtl      NoRidge       Norm

  Condition2 BldgType HouseStyle  OverallQual  OverallCond  YearBuilt  \
0       Norm     1Fam     2Story            7            5          5
1       Norm     1Fam     1Story            6            8         31
2       Norm     1Fam     2Story            7            5          7
3       Norm     1Fam     2Story            7            5         91
4       Norm     1Fam     2Story            8            5          8

   YearRemodAdd RoofStyle RoofMatl Exterior1st Exterior2nd MasVnrType  \
0             5     Gable  CompShg     VinylSd     VinylSd    BrkFace
1            31     Gable  CompShg     MetalSd     MetalSd       None
2             6     Gable  CompShg     VinylSd     VinylSd    BrkFace
3            36     Gable  CompShg     Wd Sdng     Wd Shng       None
4             8     Gable  CompShg     VinylSd     VinylSd    BrkFace

   MasVnrArea ExterQual ExterCond Foundation BsmtQual BsmtCond BsmtExposure  \
0       196.0        Gd        TA      PConc       Gd       TA           No
1         0.0        TA        TA     CBlock       Gd       TA           Gd
2       162.0        Gd        TA      PConc       Gd       TA           Mn
3         0.0        TA        TA     BrkTil       TA       Gd           No
4       350.0        Gd        TA      PConc       Gd       TA           Av

  BsmtFinType1  BsmtFinSF1 BsmtFinType2  BsmtFinSF2  BsmtUnfSF  TotalBsmtSF  \
0          GLQ         706          Unf           0        150          856
1          ALQ         978          Unf           0        284         1262
2          GLQ         486          Unf           0        434          920
3          ALQ         216          Unf           0        540          756
4          GLQ         655          Unf           0        490         1145

  Heating HeatingQC CentralAir Electrical  1stFlrSF  2ndFlrSF  LowQualFinSF  \
0    GasA        Ex          Y      SBrkr  6.752270       854             0
1    GasA        Ex          Y      SBrkr  7.140453         0             0
2    GasA        Ex          Y      SBrkr  6.824374       866             0
3    GasA        Gd          Y      SBrkr  6.867974       756             0
4    GasA        Ex          Y      SBrkr  7.043160      1053             0

   GrLivArea  BsmtFullBath  BsmtHalfBath  FullBath  HalfBath  BedroomAbvGr  \
0   7.444249             1             0         2         1             3
1   7.140453             0             1         2         0             3
2   7.487734             1             0         2         1             3
3   7.448334             1             0         1         0             3
4   7.695303             1             0         2         1             4

   KitchenAbvGr KitchenQual  TotRmsAbvGrd Functional  Fireplaces FireplaceQu  \
0             1          Gd             8        Typ           0     Missing
1             1          TA             6        Typ           1          TA
2             1          Gd             6        Typ           1          TA
3             1          Gd             7        Typ           1          Gd
4             1          Gd             9        Typ           1          TA

  GarageType  GarageYrBlt GarageFinish  GarageCars  GarageArea GarageQual  \
0     Attchd          5.0          RFn           2         548         TA
1     Attchd         31.0          RFn           2         460         TA
2     Attchd          7.0          RFn           2         608         TA
3     Detchd          8.0          Unf           3         642         TA
4     Attchd          8.0          RFn           3         836         TA

  GarageCond PavedDrive  WoodDeckSF  OpenPorchSF  EnclosedPorch  3SsnPorch  \
0         TA          Y           0           61              0          0
1         TA          Y         298            0              0          0
2         TA          Y           0           42              0          0
3         TA          Y           0           35            272          0
4         TA          Y         192           84              0          0

   ScreenPorch  PoolArea   PoolQC    Fence MiscFeature  MiscVal  MoSold  \
0            0         0  Missing  Missing     Missing        0       2
1            0         0  Missing  Missing     Missing        0       5
2            0         0  Missing  Missing     Missing        0       9
3            0         0  Missing  Missing     Missing        0       2
4            0         0  Missing  Missing     Missing        0      12

   YrSold SaleType SaleCondition  SalePrice  LotFrontagenan  MasVnrAreanan  \
0    2008       WD        Normal  12.247694               0              0
1    2007       WD        Normal  12.109011               0              0
2    2008       WD        Normal  12.317167               0              0
3    2006       WD       Abnorml  11.849398               0              0
4    2008       WD        Normal  12.429216               0              0

   GarageYrBltnan
0               0
1               0
2               0
3               0
4               0 """

""" 3. Handling Rare Categorical Feature
We will remove categorical variables that are present less than 1% of the observations
 """
categorical_features=[feature for feature in dataset.columns if dataset[feature].dtype=='O']
categorical_features
""" ['MSZoning',
 'Street',
 'Alley',
 'LotShape',
 'LandContour',
 'Utilities',
 'LotConfig',
 'LandSlope',
 'Neighborhood',
 'Condition1',
 'Condition2',
 'BldgType',
 'HouseStyle',
 'RoofStyle',
 'RoofMatl',
 'Exterior1st',
 'Exterior2nd',
 'MasVnrType',
 'ExterQual',
 'ExterCond',
 'Foundation',
 'BsmtQual',
 'BsmtCond',
 'BsmtExposure',
 'BsmtFinType1',
 'BsmtFinType2',
 'Heating',
 'HeatingQC',
 'CentralAir',
 'Electrical',
 'KitchenQual',
 'Functional',
 'FireplaceQu',
 'GarageType',
 'GarageFinish',
 'GarageQual',
 'GarageCond',
 'PavedDrive',
 'PoolQC',
 'Fence',
 'MiscFeature',
 'SaleType',
 'SaleCondition'] """

for feature in categorical_features:
    temp=dataset.groupby(feature)['SalePrice'].count()/len(dataset)
    temp_df=temp[temp>0.01].index
    dataset[feature]=np.where(dataset[feature].isin(temp_df),dataset[feature],'Rare_var')
    
dataset.head(100)
""" 
     Id  MSSubClass MSZoning  LotFrontage   LotArea Street    Alley LotShape  \
0     1          60       RL     4.174387  9.041922   Pave  Missing      Reg
1     2          20       RL     4.382027  9.169518   Pave  Missing      Reg
2     3          60       RL     4.219508  9.328123   Pave  Missing      IR1
3     4          70       RL     4.094345  9.164296   Pave  Missing      IR1
4     5          60       RL     4.430817  9.565214   Pave  Missing      IR1
..  ...         ...      ...          ...       ...    ...      ...      ...
95   96          60       RL     4.234107  9.186560   Pave  Missing      IR2
96   97          20       RL     4.356709  9.236398   Pave  Missing      IR1
97   98          20       RL     4.290459  9.298443   Pave  Missing      Reg
98   99          30       RL     4.442651  9.270965   Pave  Missing      Reg
99  100          20       RL     4.343805  9.139918   Pave  Missing      IR1

   LandContour Utilities LotConfig LandSlope Neighborhood Condition1  \
0          Lvl    AllPub    Inside       Gtl      CollgCr       Norm
1          Lvl    AllPub       FR2       Gtl     Rare_var      Feedr
2          Lvl    AllPub    Inside       Gtl      CollgCr       Norm
3          Lvl    AllPub    Corner       Gtl      Crawfor       Norm
4          Lvl    AllPub       FR2       Gtl      NoRidge       Norm
..         ...       ...       ...       ...          ...        ...
95         Lvl    AllPub    Corner       Gtl      Gilbert       Norm
96         Lvl    AllPub    Inside       Gtl      CollgCr       Norm
97         HLS    AllPub    Inside       Gtl      Edwards       Norm
98         Lvl    AllPub    Corner       Gtl      Edwards       Norm
99         Lvl    AllPub    Inside       Gtl        NAmes       Norm

   Condition2 BldgType HouseStyle  OverallQual  OverallCond  YearBuilt  \
0        Norm     1Fam     2Story            7            5          5
1        Norm     1Fam     1Story            6            8         31
2        Norm     1Fam     2Story            7            5          7
3        Norm     1Fam     2Story            7            5         91
4        Norm     1Fam     2Story            8            5          8
..        ...      ...        ...          ...          ...        ...
95       Norm     1Fam     2Story            6            8         16
96       Norm     1Fam     1Story            7            5          7
97       Norm     1Fam     1Story            4            5         42
98       Norm     1Fam     1Story            5            5         90
99       Norm     1Fam     1Story            4            5         51

    YearRemodAdd RoofStyle RoofMatl Exterior1st Exterior2nd MasVnrType  \
0              5     Gable  CompShg     VinylSd     VinylSd    BrkFace
1             31     Gable  CompShg     MetalSd     MetalSd       None
2              6     Gable  CompShg     VinylSd     VinylSd    BrkFace
3             36     Gable  CompShg     Wd Sdng     Wd Shng       None
4              8     Gable  CompShg     VinylSd     VinylSd    BrkFace
..           ...       ...      ...         ...         ...        ...
95            16     Gable  CompShg     VinylSd     VinylSd    BrkFace
96             7     Gable  CompShg     VinylSd     VinylSd    BrkFace
97            42       Hip  CompShg     HdBoard     HdBoard    BrkFace
98            60     Gable  CompShg     Wd Sdng     Wd Sdng       None
99            51     Gable  CompShg     Plywood     Plywood       None

    MasVnrArea ExterQual ExterCond Foundation BsmtQual BsmtCond BsmtExposure  \
0        196.0        Gd        TA      PConc       Gd       TA           No
1          0.0        TA        TA     CBlock       Gd       TA           Gd
2        162.0        Gd        TA      PConc       Gd       TA           Mn
3          0.0        TA        TA     BrkTil       TA       Gd           No
4        350.0        Gd        TA      PConc       Gd       TA           Av
..         ...       ...       ...        ...      ...      ...          ...
95        68.0        Ex        Gd      PConc       Gd       Gd           No
96       183.0        Gd        TA      PConc       Gd       TA           Av
97        48.0        TA        TA     CBlock       TA       TA           No
98         0.0        TA        TA     BrkTil       TA       TA           No
99         0.0        TA        TA     CBlock       TA       TA           No

   BsmtFinType1  BsmtFinSF1 BsmtFinType2  BsmtFinSF2  BsmtUnfSF  TotalBsmtSF  \
0           GLQ         706          Unf           0        150          856
1           ALQ         978          Unf           0        284         1262
2           GLQ         486          Unf           0        434          920
3           ALQ         216          Unf           0        540          756
4           GLQ         655          Unf           0        490         1145
..          ...         ...          ...         ...        ...          ...
95          ALQ         310          Unf           0        370          680
96          ALQ        1162          Unf           0        426         1588
97          Rec         520          Unf           0        440          960
98          ALQ         108          Unf           0        350          458
99          ALQ         569          Unf           0        381          950

   Heating HeatingQC CentralAir Electrical  1stFlrSF  2ndFlrSF  LowQualFinSF  \
0     GasA        Ex          Y      SBrkr  6.752270       854             0
1     GasA        Ex          Y      SBrkr  7.140453         0             0
2     GasA        Ex          Y      SBrkr  6.824374       866             0
3     GasA        Gd          Y      SBrkr  6.867974       756             0
4     GasA        Ex          Y      SBrkr  7.043160      1053             0
..     ...       ...        ...        ...       ...       ...           ...
95    GasA        Gd          Y      SBrkr  6.522093       790             0
96    GasA        Ex          Y      SBrkr  7.370231         0             0
97    GasA        TA          Y      FuseF  6.866933         0             0
98    GasA        Fa          N      SBrkr  6.727432         0             0
99    GasA        Fa          Y      SBrkr  7.110696         0             0

    GrLivArea  BsmtFullBath  BsmtHalfBath  FullBath  HalfBath  BedroomAbvGr  \
0    7.444249             1             0         2         1             3
1    7.140453             0             1         2         0             3
2    7.487734             1             0         2         1             3
3    7.448334             1             0         1         0             3
4    7.695303             1             0         2         1             4
..        ...           ...           ...       ...       ...           ...
95   7.293018             0             0         2         1             3
96   7.370231             0             0         2         0             3
97   6.866933             1             0         1         0             3
98   6.727432             0             0         1         0             2
99   7.110696             1             0         1         1             3

    KitchenAbvGr KitchenQual  TotRmsAbvGrd Functional  Fireplaces FireplaceQu  \
0              1          Gd             8        Typ           0     Missing
1              1          TA             6        Typ           1          TA
2              1          Gd             6        Typ           1          TA
3              1          Gd             7        Typ           1          Gd
4              1          Gd             9        Typ           1          TA
..           ...         ...           ...        ...         ...         ...
95             1          TA             6        Typ           1          TA
96             1          Gd             6        Typ           0     Missing
97             1          TA             6        Typ           0     Missing
98             1          TA             5        Typ           0     Missing
99             1          TA             6        Typ           0     Missing

   GarageType  GarageYrBlt GarageFinish  GarageCars  GarageArea GarageQual  \
0      Attchd          5.0          RFn           2         548         TA
1      Attchd         31.0          RFn           2         460         TA
2      Attchd          7.0          RFn           2         608         TA
3      Detchd          8.0          Unf           3         642         TA
4      Attchd          8.0          RFn           3         836         TA
..        ...          ...          ...         ...         ...        ...
95    BuiltIn         16.0          Fin           2         420         TA
96     Attchd          7.0          RFn           2         472         TA
97     Attchd         42.0          Fin           1         432         TA
98    Basment         90.0          Unf           1         366         Fa
99    Missing         30.0      Missing           0           0    Missing

   GarageCond PavedDrive  WoodDeckSF  OpenPorchSF  EnclosedPorch  3SsnPorch  \
0          TA          Y           0           61              0          0
1          TA          Y         298            0              0          0
2          TA          Y           0           42              0          0
3          TA          Y           0           35            272          0
4          TA          Y         192           84              0          0
..        ...        ...         ...          ...            ...        ...
95         TA          Y         232           63              0          0
96         TA          Y         158           29              0          0
97         TA          P         120            0              0          0
98         TA          Y           0            0             77          0
99    Missing          Y         352            0              0          0

    ScreenPorch  PoolArea   PoolQC    Fence MiscFeature  MiscVal  MoSold  \
0             0         0  Missing  Missing     Missing        0       2
1             0         0  Missing  Missing     Missing        0       5
2             0         0  Missing  Missing     Missing        0       9
3             0         0  Missing  Missing     Missing        0       2
4             0         0  Missing  Missing     Missing        0      12
..          ...       ...      ...      ...         ...      ...     ...
95            0         0  Missing  Missing        Shed      480       4
96            0         0  Missing  Missing     Missing        0       8
97            0         0  Missing  Missing     Missing        0       5
98            0         0  Missing  Missing        Shed      400       5
99            0         0  Missing  Missing        Shed      400       1

    YrSold SaleType SaleCondition  SalePrice  LotFrontagenan  MasVnrAreanan  \
0     2008       WD        Normal  12.247694               0              0
1     2007       WD        Normal  12.109011               0              0
2     2008       WD        Normal  12.317167               0              0
3     2006       WD       Abnorml  11.849398               0              0
4     2008       WD        Normal  12.429216               0              0
..     ...      ...           ...        ...             ...            ...
95    2009       WD        Normal  12.128111               1              0
96    2006       WD        Normal  12.273731               0              0
97    2007       WD        Normal  11.458997               0              0
98    2010      COD       Abnorml  11.326596               0              0
99    2010       WD        Normal  11.767180               0              0

    GarageYrBltnan
0                0
1                0
2                0
3                0
4                0
..             ...
95               0
96               0
97               0
98               0
99               1

[100 rows x 84 columns]
 """

for feature in categorical_features:
    labels_ordered=dataset.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    dataset[feature]=dataset[feature].map(labels_ordered)
    
dataset.head(10)
""" 
   Id  MSSubClass  MSZoning  LotFrontage   LotArea  Street  Alley  LotShape  \
0   1          60         3     4.174387  9.041922       1      2         0
1   2          20         3     4.382027  9.169518       1      2         0
2   3          60         3     4.219508  9.328123       1      2         1
3   4          70         3     4.094345  9.164296       1      2         1
4   5          60         3     4.430817  9.565214       1      2         1
5   6          50         3     4.442651  9.554993       1      2         1
6   7          20         3     4.317488  9.218705       1      2         0
7   8          60         3     4.234107  9.247829       1      2         1
8   9          50         1     3.931826  8.719317       1      2         0
9  10         190         3     3.912023  8.911934       1      2         0

   LandContour  Utilities  LotConfig  LandSlope  Neighborhood  Condition1  \
0            1          1          0          0            14           2
1            1          1          2          0            11           1
2            1          1          0          0            14           2
3            1          1          1          0            16           2
4            1          1          2          0            22           2
5            1          1          0          0             9           2
6            1          1          0          0            18           2
7            1          1          1          0            12           5
8            1          1          0          0             4           0
9            1          1          1          0             3           0

   Condition2  BldgType  HouseStyle  OverallQual  OverallCond  YearBuilt  \
0           1         3           5            7            5          5
1           1         3           3            6            8         31
2           1         3           5            7            5          7
3           1         3           5            7            5         91
4           1         3           5            8            5          8
5           1         3           1            5            5         16
6           1         3           3            8            5          3
7           1         3           5            7            6         36
8           1         3           1            7            5         77
9           0         0           2            5            6         69

   YearRemodAdd  RoofStyle  RoofMatl  Exterior1st  Exterior2nd  MasVnrType  \
0             5          0         0           10           10           2
1            31          0         0            4            3           1
2             6          0         0           10           10           2
3            36          0         0            2            4           1
4             8          0         0           10           10           2
5            14          0         0           10           10           1
6             2          0         0           10           10           4
7            36          0         0            6            5           4
8            58          0         0            8            4           1
9            58          0         0            4            3           1

   MasVnrArea  ExterQual  ExterCond  Foundation  BsmtQual  BsmtCond  \
0       196.0          2          3           4         3         3
1         0.0          1          3           2         3         3
2       162.0          2          3           4         3         3
3         0.0          1          3           1         2         4
4       350.0          2          3           4         3         3
5         0.0          1          3           3         3         3
6       186.0          2          3           4         4         3
7       240.0          1          3           2         3         3
8         0.0          1          3           1         2         3
9         0.0          1          3           1         2         3

   BsmtExposure  BsmtFinType1  BsmtFinSF1  BsmtFinType2  BsmtFinSF2  \
0             1             6         706             5           0
1             4             4         978             5           0
2             2             6         486             5           0
3             1             4         216             5           0
4             3             6         655             5           0
5             1             6         732             5           0
6             3             6        1369             5           0
7             2             4         859             1          32
8             1             5           0             5           0
9             1             6         851             5           0

   BsmtUnfSF  TotalBsmtSF  Heating  HeatingQC  CentralAir  Electrical  \
0        150          856        2          4           1           3
1        284         1262        2          4           1           3
2        434          920        2          4           1           3
3        540          756        2          3           1           3
4        490         1145        2          4           1           3
5         64          796        2          4           1           3
6        317         1686        2          4           1           3
7        216         1107        2          4           1           3
8        952          952        2          3           1           1
9        140          991        2          4           1           3

   1stFlrSF  2ndFlrSF  LowQualFinSF  GrLivArea  BsmtFullBath  BsmtHalfBath  \
0  6.752270       854             0   7.444249             1             0
1  7.140453         0             0   7.140453             0             1
2  6.824374       866             0   7.487734             1             0
3  6.867974       756             0   7.448334             1             0
4  7.043160      1053             0   7.695303             1             0
5  6.679599       566             0   7.216709             1             0
6  7.434848         0             0   7.434848             1             0
7  7.009409       983             0   7.644919             1             0
8  6.929517       752             0   7.480992             0             0
9  6.981935         0             0   6.981935             1             0

   FullBath  HalfBath  BedroomAbvGr  KitchenAbvGr  KitchenQual  TotRmsAbvGrd  \
0         2         1             3             1            2             8
1         2         0             3             1            1             6
2         2         1             3             1            2             6
3         1         0             3             1            2             7
4         2         1             4             1            2             9
5         1         1             1             1            1             5
6         2         0             3             1            2             7
7         2         1             3             1            1             7
8         2         0             2             2            1             8
9         1         0             2             2            1             5

   Functional  Fireplaces  FireplaceQu  GarageType  GarageYrBlt  GarageFinish  \
0           4           0            1           4          5.0             2
1           4           1            3           4         31.0             2
2           4           1            3           4          7.0             2
3           4           1            4           2          8.0             1
4           4           1            3           4          8.0             2
5           4           0            1           4         16.0             1
6           4           1            4           4          3.0             2
7           4           2            3           4         36.0             2
8           3           2            3           2         77.0             1
9           4           2            3           4         69.0             2

   GarageCars  GarageArea  GarageQual  GarageCond  PavedDrive  WoodDeckSF  \
0           2         548           2           3           2           0
1           2         460           2           3           2         298
2           2         608           2           3           2           0
3           3         642           2           3           2           0
4           3         836           2           3           2         192
5           2         480           2           3           2          40
6           2         636           2           3           2         255
7           2         484           2           3           2         235
8           2         468           1           3           2          90
9           1         205           3           3           2           0

   OpenPorchSF  EnclosedPorch  3SsnPorch  ScreenPorch  PoolArea  PoolQC  \
0           61              0          0            0         0       0
1            0              0          0            0         0       0
2           42              0          0            0         0       0
3           35            272          0            0         0       0
4           84              0          0            0         0       0
5           30              0        320            0         0       0
6           57              0          0            0         0       0
7          204            228          0            0         0       0
8            0            205          0            0         0       0
9            4              0          0            0         0       0

   Fence  MiscFeature  MiscVal  MoSold  YrSold  SaleType  SaleCondition  \
0      4            2        0       2    2008         2              3
1      4            2        0       5    2007         2              3
2      4            2        0       9    2008         2              3
3      4            2        0       2    2006         2              0
4      4            2        0      12    2008         2              3
5      2            1      700      10    2009         2              3
6      4            2        0       8    2007         2              3
7      4            1      350      11    2009         2              3
8      4            2        0       4    2008         2              0
9      4            2        0       1    2008         2              3

   SalePrice  LotFrontagenan  MasVnrAreanan  GarageYrBltnan
0  12.247694               0              0               0
1  12.109011               0              0               0
2  12.317167               0              0               0
3  11.849398               0              0               0
4  12.429216               0              0               0
5  11.870600               0              0               0
6  12.634603               0              0               0
7  12.206073               1              0               0
8  11.774520               0              0               0
9  11.678440               0              0               0 """

scaling_feature=[feature for feature in dataset.columns if feature not in ['Id','SalePerice'] ]
len(scaling_feature)
# 83

scaling_feature
""" 
['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 
'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 
'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 
'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 
'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 
'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 
'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 
'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice', 'LotFrontagenan', 'MasVnrAreanan', 
'GarageYrBltnan'] """

dataset.head()
""" 
   Id  MSSubClass  MSZoning  LotFrontage   LotArea  Street  Alley  LotShape  \
0   1          60         3     4.174387  9.041922       1      2         0
1   2          20         3     4.382027  9.169518       1      2         0
2   3          60         3     4.219508  9.328123       1      2         1
3   4          70         3     4.094345  9.164296       1      2         1
4   5          60         3     4.430817  9.565214       1      2         1

   LandContour  Utilities  LotConfig  LandSlope  Neighborhood  Condition1  \
0            1          1          0          0            14           2
1            1          1          2          0            11           1
2            1          1          0          0            14           2
3            1          1          1          0            16           2
4            1          1          2          0            22           2

   Condition2  BldgType  HouseStyle  OverallQual  OverallCond  YearBuilt  \
0           1         3           5            7            5          5
1           1         3           3            6            8         31
2           1         3           5            7            5          7
3           1         3           5            7            5         91
4           1         3           5            8            5          8

   YearRemodAdd  RoofStyle  RoofMatl  Exterior1st  Exterior2nd  MasVnrType  \
0             5          0         0           10           10           2
1            31          0         0            4            3           1
2             6          0         0           10           10           2
3            36          0         0            2            4           1
4             8          0         0           10           10           2

   MasVnrArea  ExterQual  ExterCond  Foundation  BsmtQual  BsmtCond  \
0       196.0          2          3           4         3         3
1         0.0          1          3           2         3         3
2       162.0          2          3           4         3         3
3         0.0          1          3           1         2         4
4       350.0          2          3           4         3         3

   BsmtExposure  BsmtFinType1  BsmtFinSF1  BsmtFinType2  BsmtFinSF2  \
0             1             6         706             5           0
1             4             4         978             5           0
2             2             6         486             5           0
3             1             4         216             5           0
4             3             6         655             5           0

   BsmtUnfSF  TotalBsmtSF  Heating  HeatingQC  CentralAir  Electrical  \
0        150          856        2          4           1           3
1        284         1262        2          4           1           3
2        434          920        2          4           1           3
3        540          756        2          3           1           3
4        490         1145        2          4           1           3

   1stFlrSF  2ndFlrSF  LowQualFinSF  GrLivArea  BsmtFullBath  BsmtHalfBath  \
0  6.752270       854             0   7.444249             1             0
1  7.140453         0             0   7.140453             0             1
2  6.824374       866             0   7.487734             1             0
3  6.867974       756             0   7.448334             1             0
4  7.043160      1053             0   7.695303             1             0

   FullBath  HalfBath  BedroomAbvGr  KitchenAbvGr  KitchenQual  TotRmsAbvGrd  \
0         2         1             3             1            2             8
1         2         0             3             1            1             6
2         2         1             3             1            2             6
3         1         0             3             1            2             7
4         2         1             4             1            2             9

   Functional  Fireplaces  FireplaceQu  GarageType  GarageYrBlt  GarageFinish  \
0           4           0            1           4          5.0             2
1           4           1            3           4         31.0             2
2           4           1            3           4          7.0             2
3           4           1            4           2          8.0             1
4           4           1            3           4          8.0             2

   GarageCars  GarageArea  GarageQual  GarageCond  PavedDrive  WoodDeckSF  \
0           2         548           2           3           2           0
1           2         460           2           3           2         298
2           2         608           2           3           2           0
3           3         642           2           3           2           0
4           3         836           2           3           2         192

   OpenPorchSF  EnclosedPorch  3SsnPorch  ScreenPorch  PoolArea  PoolQC  \
0           61              0          0            0         0       0
1            0              0          0            0         0       0
2           42              0          0            0         0       0
3           35            272          0            0         0       0
4           84              0          0            0         0       0

   Fence  MiscFeature  MiscVal  MoSold  YrSold  SaleType  SaleCondition  \
0      4            2        0       2    2008         2              3
1      4            2        0       5    2007         2              3
2      4            2        0       9    2008         2              3
3      4            2        0       2    2006         2              0
4      4            2        0      12    2008         2              3

   SalePrice  LotFrontagenan  MasVnrAreanan  GarageYrBltnan
0  12.247694               0              0               0
1  12.109011               0              0               0
2  12.317167               0              0               0
3  11.849398               0              0               0
4  12.429216               0              0               0 """

""" 4. Feature Scaling """
feature_scale=[feature for feature in dataset.columns if feature not in ['Id','SalePrice']]

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
scaler.fit(dataset[feature_scale])
MinMaxScaler(copy=True, feature_range=(0, 1))
scaler.transform(dataset[feature_scale])
""" 
array([[0.23529412, 0.75      , 0.41820812, ..., 0.        , 0.        ,
        0.        ],
       [0.        , 0.75      , 0.49506375, ..., 0.        , 0.        ,
        0.        ],
       [0.23529412, 0.75      , 0.434909  , ..., 0.        , 0.        ,
        0.        ],
       ...,
       [0.29411765, 0.75      , 0.42385922, ..., 0.        , 0.        ,
        0.        ],
       [0.        , 0.75      , 0.434909  , ..., 0.        , 0.        ,
        0.        ],
       [0.        , 0.75      , 0.47117546, ..., 0.        , 0.        ,
        0.        ]]) """

# transform the train and test set, and add on the Id and SalePrice variables
data = pd.concat([dataset[['Id', 'SalePrice']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(dataset[feature_scale]), columns=feature_scale)],
                    axis=1)

data.head()
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

data.to_csv('case_study1_advanced_house_price_prediction/X_train.csv',index=False)