#Importing Librarires
import numpy as np
import pandas as pd
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"] # defining the colour palette
flatui = sns.color_palette(flatui)
from wordcloud import WordCloud  
## Display all the columns of the dataframe
pd.pandas.set_option('display.max_columns',None)

df=pd.read_csv("case_study6_fifa/FIFA_data.csv") # reading the dataset

df.head(10) # having a look at the dataset, first 10 rows
""" 
   Unnamed: 0      ID               Name  Age  \
0           0  158023           L. Messi   31   
1           1   20801  Cristiano Ronaldo   33   
2           2  190871          Neymar Jr   26   
3           3  193080             De Gea   27
4           4  192985       K. De Bruyne   27
5           5  183277          E. Hazard   27
6           6  177003          L. Modrić   32
7           7  176580          L. Suárez   31
8           8  155862       Sergio Ramos   32
9           9  200389           J. Oblak   25

                                            Photo Nationality  \
0  https://cdn.sofifa.org/players/4/19/158023.png   Argentina
1   https://cdn.sofifa.org/players/4/19/20801.png    Portugal
2  https://cdn.sofifa.org/players/4/19/190871.png      Brazil
3  https://cdn.sofifa.org/players/4/19/193080.png       Spain
4  https://cdn.sofifa.org/players/4/19/192985.png     Belgium
5  https://cdn.sofifa.org/players/4/19/183277.png     Belgium
6  https://cdn.sofifa.org/players/4/19/177003.png     Croatia
7  https://cdn.sofifa.org/players/4/19/176580.png     Uruguay
8  https://cdn.sofifa.org/players/4/19/155862.png       Spain
9  https://cdn.sofifa.org/players/4/19/200389.png    Slovenia

                                  Flag  Overall  Potential  \
0  https://cdn.sofifa.org/flags/52.png       94         94
1  https://cdn.sofifa.org/flags/38.png       94         94
2  https://cdn.sofifa.org/flags/54.png       92         93
3  https://cdn.sofifa.org/flags/45.png       91         93
4   https://cdn.sofifa.org/flags/7.png       91         92
5   https://cdn.sofifa.org/flags/7.png       91         91
6  https://cdn.sofifa.org/flags/10.png       91         91
7  https://cdn.sofifa.org/flags/60.png       91         91
8  https://cdn.sofifa.org/flags/45.png       91         91
9  https://cdn.sofifa.org/flags/44.png       90         93

                  Club                                     Club Logo    Value  \
0         FC Barcelona  https://cdn.sofifa.org/teams/2/light/241.png  €110.5M
1             Juventus   https://cdn.sofifa.org/teams/2/light/45.png     €77M
2  Paris Saint-Germain   https://cdn.sofifa.org/teams/2/light/73.png  €118.5M
3    Manchester United   https://cdn.sofifa.org/teams/2/light/11.png     €72M
4      Manchester City   https://cdn.sofifa.org/teams/2/light/10.png    €102M
5              Chelsea    https://cdn.sofifa.org/teams/2/light/5.png     €93M
6          Real Madrid  https://cdn.sofifa.org/teams/2/light/243.png     €67M
7         FC Barcelona  https://cdn.sofifa.org/teams/2/light/241.png     €80M
8          Real Madrid  https://cdn.sofifa.org/teams/2/light/243.png     €51M
9      Atlético Madrid  https://cdn.sofifa.org/teams/2/light/240.png     €68M

    Wage  Special Preferred Foot  International Reputation  Weak Foot  \
0  €565K     2202           Left                       5.0        4.0
1  €405K     2228          Right                       5.0        4.0
2  €290K     2143          Right                       5.0        5.0
3  €260K     1471          Right                       4.0        3.0
4  €355K     2281          Right                       4.0        5.0
5  €340K     2142          Right                       4.0        4.0
6  €420K     2280          Right                       4.0        4.0
7  €455K     2346          Right                       5.0        4.0
8  €380K     2201          Right                       4.0        3.0
9   €94K     1331          Right                       3.0        3.0

   Skill Moves       Work Rate   Body Type Real Face Position  Jersey Number  \
0          4.0  Medium/ Medium       Messi       Yes       RF           10.0
1          5.0       High/ Low  C. Ronaldo       Yes       ST            7.0
2          5.0    High/ Medium      Neymar       Yes       LW           10.0
3          1.0  Medium/ Medium        Lean       Yes       GK            1.0
4          4.0      High/ High      Normal       Yes      RCM            7.0
5          4.0    High/ Medium      Normal       Yes       LF           10.0
6          4.0      High/ High        Lean       Yes      RCM           10.0
7          3.0    High/ Medium      Normal       Yes       RS            9.0
8          3.0    High/ Medium      Normal       Yes      RCB           15.0
9          1.0  Medium/ Medium      Normal       Yes       GK            1.0

         Joined Loaned From Contract Valid Until Height  Weight    LS    ST  \
0   Jul 1, 2004         NaN                 2021    5'7  159lbs  88+2  88+2
1  Jul 10, 2018         NaN                 2022    6'2  183lbs  91+3  91+3
2   Aug 3, 2017         NaN                 2022    5'9  150lbs  84+3  84+3
3   Jul 1, 2011         NaN                 2020    6'4  168lbs   NaN   NaN
4  Aug 30, 2015         NaN                 2023   5'11  154lbs  82+3  82+3
5   Jul 1, 2012         NaN                 2020    5'8  163lbs  83+3  83+3
6   Aug 1, 2012         NaN                 2020    5'8  146lbs  77+3  77+3
7  Jul 11, 2014         NaN                 2021    6'0  190lbs  87+5  87+5
8   Aug 1, 2005         NaN                 2020    6'0  181lbs  73+3  73+3
9  Jul 16, 2014         NaN                 2021    6'2  192lbs   NaN   NaN

     RS    LW    LF    CF    RF    RW   LAM   CAM   RAM    LM   LCM    CM  \
0  88+2  92+2  93+2  93+2  93+2  92+2  93+2  93+2  93+2  91+2  84+2  84+2
1  91+3  89+3  90+3  90+3  90+3  89+3  88+3  88+3  88+3  88+3  81+3  81+3
2  84+3  89+3  89+3  89+3  89+3  89+3  89+3  89+3  89+3  88+3  81+3  81+3
3   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN
4  82+3  87+3  87+3  87+3  87+3  87+3  88+3  88+3  88+3  88+3  87+3  87+3
5  83+3  89+3  88+3  88+3  88+3  89+3  89+3  89+3  89+3  89+3  82+3  82+3
6  77+3  85+3  84+3  84+3  84+3  85+3  87+3  87+3  87+3  86+3  88+3  88+3
7  87+5  86+5  87+5  87+5  87+5  86+5  85+5  85+5  85+5  84+5  79+5  79+5
8  73+3  70+3  71+3  71+3  71+3  70+3  71+3  71+3  71+3  72+3  75+3  75+3
9   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN

    RCM    RM   LWB   LDM   CDM   RDM   RWB    LB   LCB    CB   RCB    RB  \
0  84+2  91+2  64+2  61+2  61+2  61+2  64+2  59+2  47+2  47+2  47+2  59+2
1  81+3  88+3  65+3  61+3  61+3  61+3  65+3  61+3  53+3  53+3  53+3  61+3
2  81+3  88+3  65+3  60+3  60+3  60+3  65+3  60+3  47+3  47+3  47+3  60+3
3   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN
4  87+3  88+3  77+3  77+3  77+3  77+3  77+3  73+3  66+3  66+3  66+3  73+3
5  82+3  89+3  66+3  63+3  63+3  63+3  66+3  60+3  49+3  49+3  49+3  60+3
6  88+3  86+3  82+3  81+3  81+3  81+3  82+3  79+3  71+3  71+3  71+3  79+3
7  79+5  84+5  69+5  68+5  68+5  68+5  69+5  66+5  63+5  63+5  63+5  66+5
8  75+3  72+3  81+3  84+3  84+3  84+3  81+3  84+3  87+3  87+3  87+3  84+3
9   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN

   Crossing  Finishing  HeadingAccuracy  ShortPassing  Volleys  Dribbling  \
0      84.0       95.0             70.0          90.0     86.0       97.0
1      84.0       94.0             89.0          81.0     87.0       88.0
2      79.0       87.0             62.0          84.0     84.0       96.0
3      17.0       13.0             21.0          50.0     13.0       18.0
4      93.0       82.0             55.0          92.0     82.0       86.0
5      81.0       84.0             61.0          89.0     80.0       95.0
6      86.0       72.0             55.0          93.0     76.0       90.0
7      77.0       93.0             77.0          82.0     88.0       87.0
8      66.0       60.0             91.0          78.0     66.0       63.0
9      13.0       11.0             15.0          29.0     13.0       12.0

   Curve  FKAccuracy  LongPassing  BallControl  Acceleration  SprintSpeed  \
0   93.0        94.0         87.0         96.0          91.0         86.0
1   81.0        76.0         77.0         94.0          89.0         91.0
2   88.0        87.0         78.0         95.0          94.0         90.0
3   21.0        19.0         51.0         42.0          57.0         58.0
4   85.0        83.0         91.0         91.0          78.0         76.0
5   83.0        79.0         83.0         94.0          94.0         88.0
6   85.0        78.0         88.0         93.0          80.0         72.0
7   86.0        84.0         64.0         90.0          86.0         75.0
8   74.0        72.0         77.0         84.0          76.0         75.0
9   13.0        14.0         26.0         16.0          43.0         60.0

   Agility  Reactions  Balance  ShotPower  Jumping  Stamina  Strength  \
0     91.0       95.0     95.0       85.0     68.0     72.0      59.0
1     87.0       96.0     70.0       95.0     95.0     88.0      79.0
2     96.0       94.0     84.0       80.0     61.0     81.0      49.0
3     60.0       90.0     43.0       31.0     67.0     43.0      64.0
4     79.0       91.0     77.0       91.0     63.0     90.0      75.0
5     95.0       90.0     94.0       82.0     56.0     83.0      66.0
6     93.0       90.0     94.0       79.0     68.0     89.0      58.0
7     82.0       92.0     83.0       86.0     69.0     90.0      83.0
8     78.0       85.0     66.0       79.0     93.0     84.0      83.0
9     67.0       86.0     49.0       22.0     76.0     41.0      78.0

   LongShots  Aggression  Interceptions  Positioning  Vision  Penalties  \
0       94.0        48.0           22.0         94.0    94.0       75.0
1       93.0        63.0           29.0         95.0    82.0       85.0
2       82.0        56.0           36.0         89.0    87.0       81.0
3       12.0        38.0           30.0         12.0    68.0       40.0
4       91.0        76.0           61.0         87.0    94.0       79.0
5       80.0        54.0           41.0         87.0    89.0       86.0
6       82.0        62.0           83.0         79.0    92.0       82.0
7       85.0        87.0           41.0         92.0    84.0       85.0
8       59.0        88.0           90.0         60.0    63.0       75.0
9       12.0        34.0           19.0         11.0    70.0       11.0

   Composure  Marking  StandingTackle  SlidingTackle  GKDiving  GKHandling  \
0       96.0     33.0            28.0           26.0       6.0        11.0
1       95.0     28.0            31.0           23.0       7.0        11.0
2       94.0     27.0            24.0           33.0       9.0         9.0
3       68.0     15.0            21.0           13.0      90.0        85.0
4       88.0     68.0            58.0           51.0      15.0        13.0
5       91.0     34.0            27.0           22.0      11.0        12.0
6       84.0     60.0            76.0           73.0      13.0         9.0
7       85.0     62.0            45.0           38.0      27.0        25.0
8       82.0     87.0            92.0           91.0      11.0         8.0
9       70.0     27.0            12.0           18.0      86.0        92.0

   GKKicking  GKPositioning  GKReflexes Release Clause
0       15.0           14.0         8.0        €226.5M
1       15.0           14.0        11.0        €127.1M
2       15.0           15.0        11.0        €228.1M
3       87.0           88.0        94.0        €138.6M
4        5.0           10.0        13.0        €196.4M
5        6.0            8.0         8.0        €172.1M
6        7.0           14.0         9.0        €137.4M
7       31.0           33.0        37.0          €164M
8        9.0            7.0        11.0        €104.6M
9       78.0           88.0        89.0        €144.5M"""

df.shape # checking the number of rows and columns in the dataset
# (18207, 89)

df.info() #Printing a concise summary of the DataFrame.
""" <class 'pandas.core.frame.DataFrame'>
RangeIndex: 18207 entries, 0 to 18206
Data columns (total 89 columns):
Unnamed: 0                  18207 non-null int64
ID                          18207 non-null int64
Name                        18207 non-null object
Age                         18207 non-null int64
Photo                       18207 non-null object
Nationality                 18207 non-null object
Flag                        18207 non-null object
Overall                     18207 non-null int64
Potential                   18207 non-null int64
Club                        17966 non-null object
Club Logo                   18207 non-null object
Value                       18207 non-null object
Wage                        18207 non-null object
Special                     18207 non-null int64
Preferred Foot              18159 non-null object
International Reputation    18159 non-null float64
Weak Foot                   18159 non-null float64
Skill Moves                 18159 non-null float64
Work Rate                   18159 non-null object
Body Type                   18159 non-null object
Real Face                   18159 non-null object
Position                    18147 non-null object
Jersey Number               18147 non-null float64
Joined                      16654 non-null object
Loaned From                 1264 non-null object
Contract Valid Until        17918 non-null object
Height                      18159 non-null object
Weight                      18159 non-null object
LS                          16122 non-null object
ST                          16122 non-null object
RS                          16122 non-null object
LW                          16122 non-null object
LF                          16122 non-null object
CF                          16122 non-null object
RF                          16122 non-null object
RW                          16122 non-null object
LAM                         16122 non-null object
CAM                         16122 non-null object
RAM                         16122 non-null object
LM                          16122 non-null object
LCM                         16122 non-null object
CM                          16122 non-null object
RCM                         16122 non-null object
RM                          16122 non-null object
LWB                         16122 non-null object
LDM                         16122 non-null object
CDM                         16122 non-null object
RDM                         16122 non-null object
RWB                         16122 non-null object
LB                          16122 non-null object
LCB                         16122 non-null object
CB                          16122 non-null object
RCB                         16122 non-null object
RB                          16122 non-null object
Crossing                    18159 non-null float64
Finishing                   18159 non-null float64
HeadingAccuracy             18159 non-null float64
ShortPassing                18159 non-null float64
Volleys                     18159 non-null float64
Dribbling                   18159 non-null float64
Curve                       18159 non-null float64
FKAccuracy                  18159 non-null float64
LongPassing                 18159 non-null float64
BallControl                 18159 non-null float64
Acceleration                18159 non-null float64
SprintSpeed                 18159 non-null float64
Agility                     18159 non-null float64
Reactions                   18159 non-null float64
Balance                     18159 non-null float64
ShotPower                   18159 non-null float64
Jumping                     18159 non-null float64
Stamina                     18159 non-null float64
Strength                    18159 non-null float64
LongShots                   18159 non-null float64
Aggression                  18159 non-null float64
Interceptions               18159 non-null float64
Positioning                 18159 non-null float64
Vision                      18159 non-null float64
Penalties                   18159 non-null float64
Composure                   18159 non-null float64
Marking                     18159 non-null float64
StandingTackle              18159 non-null float64
SlidingTackle               18159 non-null float64
GKDiving                    18159 non-null float64
GKHandling                  18159 non-null float64
GKKicking                   18159 non-null float64
GKPositioning               18159 non-null float64
GKReflexes                  18159 non-null float64
Release Clause              16643 non-null object
dtypes: float64(38), int64(6), object(45)
memory usage: 12.4+ MB """

df.isnull().sum() # checking the count of the missing values in each column
""" Unnamed: 0                      0
ID                              0
Name                            0
Age                             0
Photo                           0
Nationality                     0
Flag                            0
Overall                         0
Potential                       0
Club                          241
Club Logo                       0
Value                           0
Wage                            0
Special                         0
Preferred Foot                 48
International Reputation       48
Weak Foot                      48
Skill Moves                    48
Work Rate                      48
Body Type                      48
Real Face                      48
Position                       60
Jersey Number                  60
Joined                       1553
Loaned From                 16943
Contract Valid Until          289
Height                         48
Weight                         48
LS                           2085
ST                           2085
                            ...  
Dribbling                      48
Curve                          48
FKAccuracy                     48
LongPassing                    48
BallControl                    48
Acceleration                   48
SprintSpeed                    48
Agility                        48
Reactions                      48
Balance                        48
ShotPower                      48
Jumping                        48
Stamina                        48
Strength                       48
LongShots                      48
Aggression                     48
Interceptions                  48
Positioning                    48
Vision                         48
Penalties                      48
Composure                      48
Marking                        48
StandingTackle                 48
SlidingTackle                  48
GKDiving                       48
GKHandling                     48
GKKicking                      48
GKPositioning                  48
GKReflexes                     48
Release Clause               1564
Length: 89, dtype: int64 """

df.columns # listing the columns
""" Index(['Unnamed: 0', 'ID', 'Name', 'Age', 'Photo', 'Nationality', 'Flag',
       'Overall', 'Potential', 'Club', 'Club Logo', 'Value', 'Wage', 'Special',
       'Preferred Foot', 'International Reputation', 'Weak Foot',
       'Skill Moves', 'Work Rate', 'Body Type', 'Real Face', 'Position',
       'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until',
       'Height', 'Weight', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Crossing',
       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause'],
      dtype='object') """

# Plotting the Heatmap of the columns using correlation matrix
f,ax = plt.subplots(figsize=(25, 15))
sns.heatmap(df.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()

#Nationality Text Size = Nationality Player Count
# Ploting the wordcloud for the Nationalit column
plt.subplots(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='white',
                          width=1920,
                          height=1080
                         ).generate(" ".join(df.Nationality))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')
plt.show()

""" 
- In the next few steps we'll be imputing the missing values from the dataset. 
As the dataset containes a lot of rows, we won't repeatedly show all the imputations. 
Instead, we will show the final dataset after all the imputations to establish that we have achived a dataset
which doesn't have any missing values
 """

#Imputing the missing values for the columns Club and Position
df['Club'].fillna('No Club', inplace = True)
df['Position'].fillna('ST', inplace = True)

# selecting columns to impute the missing values by mean
to_impute_by_mean = df.loc[:, ['Crossing', 'Finishing', 'HeadingAccuracy',
                                 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',
                                 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',
                                 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping',
                                 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',
                                 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',
                                 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                                 'GKKicking', 'GKPositioning', 'GKReflexes']]

# replacing the missing values with mean
for i in to_impute_by_mean.columns:
    df[i].fillna(df[i].mean(), inplace = True)

'''These are categorical variables and will be imputed by mode.'''
to_impute_by_mode = df.loc[:, ['Body Type','International Reputation', 'Height', 'Weight', 'Preferred Foot','Jersey Number']]
for i in to_impute_by_mode.columns:
    df[i].fillna(df[i].mode()[0], inplace = True)

'''The following variables are either discrete numerical or continuous numerical variables.
So the will be imputed by median.'''
to_impute_by_median = df.loc[:, ['Weak Foot', 'Skill Moves', ]]
for i in to_impute_by_median.columns:
    df[i].fillna(df[i].median(), inplace = True)

'''Columns remaining to be imputed'''
df.columns[df.isna().any()]
""" Index(['Work Rate', 'Real Face', 'Joined', 'Loaned From',
       'Contract Valid Until', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Release Clause'],
      dtype='object') """

df.fillna(0, inplace = True) # Filling the remaining  missing values with zero

df.head(10)
""" 
   Unnamed: 0      ID               Name  Age  \
0           0  158023           L. Messi   31
1           1   20801  Cristiano Ronaldo   33
2           2  190871          Neymar Jr   26
3           3  193080             De Gea   27
4           4  192985       K. De Bruyne   27
5           5  183277          E. Hazard   27
6           6  177003          L. Modrić   32
7           7  176580          L. Suárez   31
8           8  155862       Sergio Ramos   32
9           9  200389           J. Oblak   25

                                            Photo Nationality  \
0  https://cdn.sofifa.org/players/4/19/158023.png   Argentina
1   https://cdn.sofifa.org/players/4/19/20801.png    Portugal
2  https://cdn.sofifa.org/players/4/19/190871.png      Brazil
3  https://cdn.sofifa.org/players/4/19/193080.png       Spain
4  https://cdn.sofifa.org/players/4/19/192985.png     Belgium
5  https://cdn.sofifa.org/players/4/19/183277.png     Belgium
6  https://cdn.sofifa.org/players/4/19/177003.png     Croatia
7  https://cdn.sofifa.org/players/4/19/176580.png     Uruguay
8  https://cdn.sofifa.org/players/4/19/155862.png       Spain
9  https://cdn.sofifa.org/players/4/19/200389.png    Slovenia

                                  Flag  Overall  Potential  \
0  https://cdn.sofifa.org/flags/52.png       94         94
1  https://cdn.sofifa.org/flags/38.png       94         94
2  https://cdn.sofifa.org/flags/54.png       92         93
3  https://cdn.sofifa.org/flags/45.png       91         93
4   https://cdn.sofifa.org/flags/7.png       91         92
5   https://cdn.sofifa.org/flags/7.png       91         91
6  https://cdn.sofifa.org/flags/10.png       91         91
7  https://cdn.sofifa.org/flags/60.png       91         91
8  https://cdn.sofifa.org/flags/45.png       91         91
9  https://cdn.sofifa.org/flags/44.png       90         93

                  Club                                     Club Logo    Value  \
0         FC Barcelona  https://cdn.sofifa.org/teams/2/light/241.png  €110.5M
1             Juventus   https://cdn.sofifa.org/teams/2/light/45.png     €77M
2  Paris Saint-Germain   https://cdn.sofifa.org/teams/2/light/73.png  €118.5M
3    Manchester United   https://cdn.sofifa.org/teams/2/light/11.png     €72M
4      Manchester City   https://cdn.sofifa.org/teams/2/light/10.png    €102M
5              Chelsea    https://cdn.sofifa.org/teams/2/light/5.png     €93M
6          Real Madrid  https://cdn.sofifa.org/teams/2/light/243.png     €67M
7         FC Barcelona  https://cdn.sofifa.org/teams/2/light/241.png     €80M
8          Real Madrid  https://cdn.sofifa.org/teams/2/light/243.png     €51M
9      Atlético Madrid  https://cdn.sofifa.org/teams/2/light/240.png     €68M

    Wage  Special Preferred Foot  International Reputation  Weak Foot  \
0  €565K     2202           Left                       5.0        4.0
1  €405K     2228          Right                       5.0        4.0
2  €290K     2143          Right                       5.0        5.0
3  €260K     1471          Right                       4.0        3.0
4  €355K     2281          Right                       4.0        5.0
5  €340K     2142          Right                       4.0        4.0
6  €420K     2280          Right                       4.0        4.0
7  €455K     2346          Right                       5.0        4.0
8  €380K     2201          Right                       4.0        3.0
9   €94K     1331          Right                       3.0        3.0

   Skill Moves       Work Rate   Body Type Real Face Position  Jersey Number  \
0          4.0  Medium/ Medium       Messi       Yes       RF           10.0
1          5.0       High/ Low  C. Ronaldo       Yes       ST            7.0
2          5.0    High/ Medium      Neymar       Yes       LW           10.0
3          1.0  Medium/ Medium        Lean       Yes       GK            1.0
4          4.0      High/ High      Normal       Yes      RCM            7.0
5          4.0    High/ Medium      Normal       Yes       LF           10.0
6          4.0      High/ High        Lean       Yes      RCM           10.0
7          3.0    High/ Medium      Normal       Yes       RS            9.0
8          3.0    High/ Medium      Normal       Yes      RCB           15.0
9          1.0  Medium/ Medium      Normal       Yes       GK            1.0

         Joined Loaned From Contract Valid Until Height  Weight    LS    ST  \
0   Jul 1, 2004         NaN                 2021    5'7  159lbs  88+2  88+2
1  Jul 10, 2018         NaN                 2022    6'2  183lbs  91+3  91+3
2   Aug 3, 2017         NaN                 2022    5'9  150lbs  84+3  84+3
3   Jul 1, 2011         NaN                 2020    6'4  168lbs   NaN   NaN
4  Aug 30, 2015         NaN                 2023   5'11  154lbs  82+3  82+3
5   Jul 1, 2012         NaN                 2020    5'8  163lbs  83+3  83+3
6   Aug 1, 2012         NaN                 2020    5'8  146lbs  77+3  77+3
7  Jul 11, 2014         NaN                 2021    6'0  190lbs  87+5  87+5
8   Aug 1, 2005         NaN                 2020    6'0  181lbs  73+3  73+3
9  Jul 16, 2014         NaN                 2021    6'2  192lbs   NaN   NaN

     RS    LW    LF    CF    RF    RW   LAM   CAM   RAM    LM   LCM    CM  \
0  88+2  92+2  93+2  93+2  93+2  92+2  93+2  93+2  93+2  91+2  84+2  84+2
1  91+3  89+3  90+3  90+3  90+3  89+3  88+3  88+3  88+3  88+3  81+3  81+3
2  84+3  89+3  89+3  89+3  89+3  89+3  89+3  89+3  89+3  88+3  81+3  81+3
3   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN
4  82+3  87+3  87+3  87+3  87+3  87+3  88+3  88+3  88+3  88+3  87+3  87+3
5  83+3  89+3  88+3  88+3  88+3  89+3  89+3  89+3  89+3  89+3  82+3  82+3
6  77+3  85+3  84+3  84+3  84+3  85+3  87+3  87+3  87+3  86+3  88+3  88+3
7  87+5  86+5  87+5  87+5  87+5  86+5  85+5  85+5  85+5  84+5  79+5  79+5
8  73+3  70+3  71+3  71+3  71+3  70+3  71+3  71+3  71+3  72+3  75+3  75+3
9   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN

    RCM    RM   LWB   LDM   CDM   RDM   RWB    LB   LCB    CB   RCB    RB  \
0  84+2  91+2  64+2  61+2  61+2  61+2  64+2  59+2  47+2  47+2  47+2  59+2
1  81+3  88+3  65+3  61+3  61+3  61+3  65+3  61+3  53+3  53+3  53+3  61+3
2  81+3  88+3  65+3  60+3  60+3  60+3  65+3  60+3  47+3  47+3  47+3  60+3
3   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN
4  87+3  88+3  77+3  77+3  77+3  77+3  77+3  73+3  66+3  66+3  66+3  73+3
5  82+3  89+3  66+3  63+3  63+3  63+3  66+3  60+3  49+3  49+3  49+3  60+3
6  88+3  86+3  82+3  81+3  81+3  81+3  82+3  79+3  71+3  71+3  71+3  79+3
7  79+5  84+5  69+5  68+5  68+5  68+5  69+5  66+5  63+5  63+5  63+5  66+5
8  75+3  72+3  81+3  84+3  84+3  84+3  81+3  84+3  87+3  87+3  87+3  84+3
9   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN

   Crossing  Finishing  HeadingAccuracy  ShortPassing  Volleys  Dribbling  \
0      84.0       95.0             70.0          90.0     86.0       97.0
1      84.0       94.0             89.0          81.0     87.0       88.0
2      79.0       87.0             62.0          84.0     84.0       96.0
3      17.0       13.0             21.0          50.0     13.0       18.0
4      93.0       82.0             55.0          92.0     82.0       86.0
5      81.0       84.0             61.0          89.0     80.0       95.0
6      86.0       72.0             55.0          93.0     76.0       90.0
7      77.0       93.0             77.0          82.0     88.0       87.0
8      66.0       60.0             91.0          78.0     66.0       63.0
9      13.0       11.0             15.0          29.0     13.0       12.0

   Curve  FKAccuracy  LongPassing  BallControl  Acceleration  SprintSpeed  \
0   93.0        94.0         87.0         96.0          91.0         86.0
1   81.0        76.0         77.0         94.0          89.0         91.0
2   88.0        87.0         78.0         95.0          94.0         90.0
3   21.0        19.0         51.0         42.0          57.0         58.0
4   85.0        83.0         91.0         91.0          78.0         76.0
5   83.0        79.0         83.0         94.0          94.0         88.0
6   85.0        78.0         88.0         93.0          80.0         72.0
7   86.0        84.0         64.0         90.0          86.0         75.0
8   74.0        72.0         77.0         84.0          76.0         75.0
9   13.0        14.0         26.0         16.0          43.0         60.0

   Agility  Reactions  Balance  ShotPower  Jumping  Stamina  Strength  \
0     91.0       95.0     95.0       85.0     68.0     72.0      59.0
1     87.0       96.0     70.0       95.0     95.0     88.0      79.0
2     96.0       94.0     84.0       80.0     61.0     81.0      49.0
3     60.0       90.0     43.0       31.0     67.0     43.0      64.0
4     79.0       91.0     77.0       91.0     63.0     90.0      75.0
5     95.0       90.0     94.0       82.0     56.0     83.0      66.0
6     93.0       90.0     94.0       79.0     68.0     89.0      58.0
7     82.0       92.0     83.0       86.0     69.0     90.0      83.0
8     78.0       85.0     66.0       79.0     93.0     84.0      83.0
9     67.0       86.0     49.0       22.0     76.0     41.0      78.0

   LongShots  Aggression  Interceptions  Positioning  Vision  Penalties  \
0       94.0        48.0           22.0         94.0    94.0       75.0
1       93.0        63.0           29.0         95.0    82.0       85.0
2       82.0        56.0           36.0         89.0    87.0       81.0
3       12.0        38.0           30.0         12.0    68.0       40.0
4       91.0        76.0           61.0         87.0    94.0       79.0
5       80.0        54.0           41.0         87.0    89.0       86.0
6       82.0        62.0           83.0         79.0    92.0       82.0
7       85.0        87.0           41.0         92.0    84.0       85.0
8       59.0        88.0           90.0         60.0    63.0       75.0
9       12.0        34.0           19.0         11.0    70.0       11.0

   Composure  Marking  StandingTackle  SlidingTackle  GKDiving  GKHandling  \
0       96.0     33.0            28.0           26.0       6.0        11.0
1       95.0     28.0            31.0           23.0       7.0        11.0
2       94.0     27.0            24.0           33.0       9.0         9.0
3       68.0     15.0            21.0           13.0      90.0        85.0
4       88.0     68.0            58.0           51.0      15.0        13.0
5       91.0     34.0            27.0           22.0      11.0        12.0
6       84.0     60.0            76.0           73.0      13.0         9.0
7       85.0     62.0            45.0           38.0      27.0        25.0
8       82.0     87.0            92.0           91.0      11.0         8.0
9       70.0     27.0            12.0           18.0      86.0        92.0

   GKKicking  GKPositioning  GKReflexes Release Clause
0       15.0           14.0         8.0        €226.5M
1       15.0           14.0        11.0        €127.1M
2       15.0           15.0        11.0        €228.1M
3       87.0           88.0        94.0        €138.6M
4        5.0           10.0        13.0        €196.4M
5        6.0            8.0         8.0        €172.1M
6        7.0           14.0         9.0        €137.4M
7       31.0           33.0        37.0          €164M
8        9.0            7.0        11.0        €104.6M
9       78.0           88.0        89.0        €144.5M """

# functions to get the rounded values from different columns
def defending(data):
    return int(round((data[['Marking', 'StandingTackle', 
                               'SlidingTackle']].mean()).mean()))

def general(data):
    return int(round((data[['HeadingAccuracy', 'Dribbling', 'Curve', 
                               'BallControl']].mean()).mean()))

def mental(data):
    return int(round((data[['Aggression', 'Interceptions', 'Positioning', 
                               'Vision','Composure']].mean()).mean()))

def passing(data):
    return int(round((data[['Crossing', 'ShortPassing', 
                               'LongPassing']].mean()).mean()))

def mobility(data):
    return int(round((data[['Acceleration', 'SprintSpeed', 
                               'Agility','Reactions']].mean()).mean()))
def power(data):
    return int(round((data[['Balance', 'Jumping', 'Stamina', 
                               'Strength']].mean()).mean()))

def rating(data):
    return int(round((data[['Potential', 'Overall']].mean()).mean()))

def shooting(data):
    return int(round((data[['Finishing', 'Volleys', 'FKAccuracy', 
                               'ShotPower','LongShots', 'Penalties']].mean()).mean()))

# renaming a column
df.rename(columns={'Club Logo':'Club_Logo'}, inplace=True)
df.columns
""" Index(['Unnamed: 0', 'ID', 'Name', 'Age', 'Photo', 'Nationality', 'Flag',
       'Overall', 'Potential', 'Club', 'Club_Logo', 'Value', 'Wage', 'Special',
       'Preferred Foot', 'International Reputation', 'Weak Foot',
       'Skill Moves', 'Work Rate', 'Body Type', 'Real Face', 'Position',
       'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until',
       'Height', 'Weight', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Crossing',
       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause',
       'Defending', 'General', 'Mental', 'Passing', 'Mobility', 'Power',
       'Rating', 'Shooting'],
      dtype='object') """

# adding these categories to the data
df['Defending'] = df.apply(defending, axis = 1)
df['General'] = df.apply(general, axis = 1)
df['Mental'] = df.apply(mental, axis = 1)
df['Passing'] = df.apply(passing, axis = 1)
df['Mobility'] = df.apply(mobility, axis = 1)
df['Power'] = df.apply(power, axis = 1)
df['Rating'] = df.apply(rating, axis = 1)
df['Shooting'] = df.apply(shooting, axis = 1)

# dataset after transformation
df.head(10)

# creating the players dataset
players = df[['Name','Defending','General','Mental','Passing',
                'Mobility','Power','Rating','Shooting','Flag','Age',
                'Nationality', 'Photo', 'Club_Logo', 'Club']]

players.head(10)

# different positions acquired by the players 
plt.figure(figsize = (18, 8))
plt.style.use('fivethirtyeight')
ax = sns.countplot('Position', data = df, palette = 'dark')
ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)
plt.show()

# plotting count of players based on their heights
plt.figure(figsize = (13, 8))
ax = sns.countplot(x = 'Height', data = df, palette = 'bone')
ax.set_title(label = 'Count of players on Basis of Height', fontsize = 20)
ax.set_xlabel(xlabel = 'Height in Foot per inch', fontsize = 16)
ax.set_ylabel(ylabel = 'Count', fontsize = 16)
plt.show()

# To show Different Work rate of the players participating in the FIFA 2019
plt.figure(figsize = (15, 7))
plt.style.use('_classic_test')

sns.countplot(x = 'Work Rate', data = df, palette = 'hls')
plt.title('Different work rates of the Players Participating in the FIFA 2019', fontsize = 20)
plt.xlabel('Work rates associated with the players', fontsize = 16)
plt.ylabel('count of Players', fontsize = 16)
plt.show()

# Histogram for the Speciality Scores of the Players
x = df.Special
plt.figure(figsize = (12, 8))
plt.style.use('tableau-colorblind10')

ax = sns.distplot(x, bins = 58, kde = False, color = 'cyan')
ax.set_xlabel(xlabel = 'Special score range', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of the Players',fontsize = 16)
ax.set_title(label = 'Histogram for the Speciality Scores of the Players', fontsize = 20)
plt.show()

# Every Nations' Player and their overall scores
some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia') # defining a tuple consisting of country names
data_countries = df.loc[df['Nationality'].isin(some_countries) & df['Overall']] # extracting the overall data of the countries selected in the line above
data_countries.head()
""" 
    Unnamed: 0      ID          Name  Age  \
3            3  193080        De Gea   27
8            8  155862  Sergio Ramos   32
14          14  215914      N. Kanté   27
15          15  211110     P. Dybala   24
16          16  202126       H. Kane   24

                                             Photo Nationality  \
3   https://cdn.sofifa.org/players/4/19/193080.png       Spain
8   https://cdn.sofifa.org/players/4/19/155862.png       Spain
14  https://cdn.sofifa.org/players/4/19/215914.png      France
15  https://cdn.sofifa.org/players/4/19/211110.png   Argentina
16  https://cdn.sofifa.org/players/4/19/202126.png     England

                                   Flag  Overall  Potential  \
3   https://cdn.sofifa.org/flags/45.png       91         93
8   https://cdn.sofifa.org/flags/45.png       91         91
14  https://cdn.sofifa.org/flags/18.png       89         90
15  https://cdn.sofifa.org/flags/52.png       89         94
16  https://cdn.sofifa.org/flags/14.png       89         91

                 Club                                     Club_Logo   Value  \
3   Manchester United   https://cdn.sofifa.org/teams/2/light/11.png    €72M
8         Real Madrid  https://cdn.sofifa.org/teams/2/light/243.png    €51M
14            Chelsea    https://cdn.sofifa.org/teams/2/light/5.png    €63M
15           Juventus   https://cdn.sofifa.org/teams/2/light/45.png    €89M
16  Tottenham Hotspur   https://cdn.sofifa.org/teams/2/light/18.png  €83.5M

     Wage  Special Preferred Foot  International Reputation  Weak Foot  \
3   €260K     1471          Right                       4.0        3.0
8   €380K     2201          Right                       4.0        3.0
14  €225K     2189          Right                       3.0        3.0
15  €205K     2092           Left                       3.0        3.0
16  €205K     2165          Right                       3.0        4.0

    Skill Moves       Work Rate Body Type Real Face Position  Jersey Number  \
3           1.0  Medium/ Medium      Lean       Yes       GK            1.0
8           3.0    High/ Medium    Normal       Yes      RCB           15.0
14          2.0    Medium/ High      Lean       Yes      LDM           13.0
15          4.0    High/ Medium    Normal       Yes       LF           21.0
16          3.0      High/ High    Normal       Yes       ST            9.0

          Joined Loaned From Contract Valid Until Height  Weight    LS    ST  \
3    Jul 1, 2011         NaN                 2020    6'4  168lbs   NaN   NaN
8    Aug 1, 2005         NaN                 2020    6'0  181lbs  73+3  73+3
14  Jul 16, 2016         NaN                 2023    5'6  159lbs  72+3  72+3
15   Jul 1, 2015         NaN                 2022   5'10  165lbs  83+3  83+3
16   Jul 1, 2010         NaN                 2024    6'2  196lbs  86+3  86+3

      RS    LW    LF    CF    RF    RW   LAM   CAM   RAM    LM   LCM    CM  \
3    NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN
8   73+3  70+3  71+3  71+3  71+3  70+3  71+3  71+3  71+3  72+3  75+3  75+3
14  72+3  77+3  77+3  77+3  77+3  77+3  79+3  79+3  79+3  79+3  82+3  82+3
15  83+3  87+3  86+3  86+3  86+3  87+3  87+3  87+3  87+3  86+3  79+3  79+3
16  86+3  82+3  84+3  84+3  84+3  82+3  82+3  82+3  82+3  81+3  79+3  79+3

     RCM    RM   LWB   LDM   CDM   RDM   RWB    LB   LCB    CB   RCB    RB  \
3    NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN
8   75+3  72+3  81+3  84+3  84+3  84+3  81+3  84+3  87+3  87+3  87+3  84+3
14  82+3  79+3  85+3  87+3  87+3  87+3  85+3  84+3  83+3  83+3  83+3  84+3
15  79+3  86+3  62+3  58+3  58+3  58+3  62+3  56+3  45+3  45+3  45+3  56+3
16  79+3  81+3  65+3  66+3  66+3  66+3  65+3  62+3  60+3  60+3  60+3  62+3

    Crossing  Finishing  HeadingAccuracy  ShortPassing  Volleys  Dribbling  \
3       17.0       13.0             21.0          50.0     13.0       18.0
8       66.0       60.0             91.0          78.0     66.0       63.0
14      68.0       65.0             54.0          86.0     56.0       79.0
15      82.0       84.0             68.0          87.0     88.0       92.0
16      75.0       94.0             85.0          80.0     84.0       80.0

    Curve  FKAccuracy  LongPassing  BallControl  Acceleration  SprintSpeed  \
3    21.0        19.0         51.0         42.0          57.0         58.0
8    74.0        72.0         77.0         84.0          76.0         75.0
14   49.0        49.0         81.0         80.0          82.0         78.0
15   88.0        88.0         75.0         92.0          87.0         83.0
16   78.0        68.0         82.0         84.0          68.0         72.0

    Agility  Reactions  Balance  ShotPower  Jumping  Stamina  Strength  \
3      60.0       90.0     43.0       31.0     67.0     43.0      64.0
8      78.0       85.0     66.0       79.0     93.0     84.0      83.0
14     82.0       93.0     92.0       71.0     77.0     96.0      76.0
15     91.0       86.0     85.0       82.0     75.0     80.0      65.0
16     71.0       91.0     71.0       88.0     78.0     89.0      84.0

    LongShots  Aggression  Interceptions  Positioning  Vision  Penalties  \
3        12.0        38.0           30.0         12.0    68.0       40.0
8        59.0        88.0           90.0         60.0    63.0       75.0
14       69.0        90.0           92.0         71.0    79.0       54.0
15       88.0        48.0           32.0         84.0    87.0       86.0
16       85.0        76.0           35.0         93.0    80.0       90.0

    Composure  Marking  StandingTackle  SlidingTackle  GKDiving  GKHandling  \
3        68.0     15.0            21.0           13.0      90.0        85.0
8        82.0     87.0            92.0           91.0      11.0         8.0
14       85.0     90.0            91.0           85.0      15.0        12.0
15       84.0     23.0            20.0           20.0       5.0         4.0
16       89.0     56.0            36.0           38.0       8.0        10.0

    GKKicking  GKPositioning  GKReflexes Release Clause  Defending  General  \
3        87.0           88.0        94.0        €138.6M         16       26
8         9.0            7.0        11.0        €104.6M         90       78
14       10.0            7.0        10.0        €121.3M         89       66
15        4.0            5.0         8.0        €153.5M         21       85
16       11.0           14.0        11.0        €160.7M         43       82

    Mental  Passing  Mobility  Power  Rating  Shooting
3       43       39        66     54      92        21
8       77       74        78     82      91        68
14      83       78        84     85      90        61
15      67       81        87     76      92        86
16      75       79        76     80      90        85 """

plt.rcParams['figure.figsize'] = (15, 7)
ax = sns.barplot(x = data_countries['Nationality'], y = data_countries['Overall'], palette = 'spring') # creating a bargraph
ax.set_xlabel(xlabel = 'Countries', fontsize = 9)
ax.set_ylabel(ylabel = 'Overall Scores', fontsize = 9)
ax.set_title(label = 'Distribution of overall scores of players from different countries', fontsize = 20)
plt.show()

df['Club'].value_counts().head(10) # finding the number of players in each club
""" No Club                    241
TSG 1899 Hoffenheim         33
Wolverhampton Wanderers     33
CD Leganés                  33
Southampton                 33
Burnley                     33
Rayo Vallecano              33
Manchester United           33
RC Celta                    33
Eintracht Frankfurt         33
Name: Club, dtype: int64 """

data = df.copy() # creating a copy dataset
# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set(style="ticks")
some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',
             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid') # creating a tuple of club names

data_clubs = data.loc[data['Club'].isin(some_clubs) & data['Overall']] # extracting the overall data of the clubs selected in the line above

data_clubs.head()

plt.rcParams['figure.figsize'] = (15, 8)
ax = sns.boxplot(x = data_clubs['Club'], y = data_clubs['Overall'], palette = 'inferno') # creating a boxplot
ax.set_xlabel(xlabel = 'Some Popular Clubs', fontsize = 9)
ax.set_ylabel(ylabel = 'Overall Score', fontsize = 9)
ax.set_title(label = 'Distribution of Overall Score in Different popular Clubs', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()

# finding out the top 10 left footed footballers

left = data[data['Preferred Foot'] == 'Left'][['Name', 'Age', 'Club', 'Nationality']].head(10)
left
""" 
            Name  Age               Club Nationality
0       L. Messi   31       FC Barcelona   Argentina
13   David Silva   32    Manchester City       Spain
15     P. Dybala   24           Juventus   Argentina
17  A. Griezmann   27    Atlético Madrid      France
19   T. Courtois   26        Real Madrid     Belgium
24  G. Chiellini   33           Juventus       Italy
26      M. Salah   26          Liverpool       Egypt
28  J. Rodríguez   26  FC Bayern München    Colombia
35       Marcelo   30        Real Madrid      Brazil
36       G. Bale   28        Real Madrid       Wales """
# finding out the top 10 Right footed footballers

right = data[data['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club', 'Nationality']].head(10)
right
""" 
                 Name  Age                 Club Nationality
1   Cristiano Ronaldo   33             Juventus    Portugal
2           Neymar Jr   26  Paris Saint-Germain      Brazil
3              De Gea   27    Manchester United       Spain
4        K. De Bruyne   27      Manchester City     Belgium
5           E. Hazard   27              Chelsea     Belgium
6           L. Modrić   32          Real Madrid     Croatia
7           L. Suárez   31         FC Barcelona     Uruguay
8        Sergio Ramos   32          Real Madrid       Spain
9            J. Oblak   25      Atlético Madrid    Slovenia
10     R. Lewandowski   29    FC Bayern München      Poland """
# comparing the performance of left-footed and right-footed footballers
# ballcontrol vs dribbing

sns.lmplot(x = 'BallControl', y = 'Dribbling', data = data, col = 'Preferred Foot')
plt.show()

data.groupby(data['Club'])['Nationality'].nunique().sort_values(ascending = False).head(10) # checking the clubs where players from the most number of nations play
""" Club
No Club                   28
Brighton & Hove Albion    21
Fulham                    19
Udinese                   18
Napoli                    18
Empoli                    18
Eintracht Frankfurt       18
West Ham United           18
AS Monaco                 18
Lazio                     18
Name: Nationality, dtype: int64 """

data.groupby(data['Club'])['Nationality'].nunique().sort_values(ascending = True).head(10) # checking the clubs where players from the least number of nations play
""" Club
Santos                       1
Ceará Sporting Club          1
América FC (Minas Gerais)    1
Paraná                       1
Chapecoense                  1
Padova                       1
Cittadella                   1
Sangju Sangmu FC             1
Ranheim Fotball              1
CA Osasuna                   1
Name: Nationality, dtype: int64 """

df.head()

df.drop(['Unnamed: 0'],axis=1,inplace=True) # dropping the unnamed column
df.head() # dataset after dropping column

#Player with maximum Potential and Overall Performance
player = str(df.loc[df['Potential'].idxmax()][1])
print('Maximum Potential : '+str(df.loc[df['Potential'].idxmax()][1]))
print('Maximum Overall Perforamnce : '+str(df.loc[df['Overall'].idxmax()][1]))
""" Maximum Potential : K. Mbappé
Maximum Overall Perforamnce : L. Messi """
# finding the best players for each performance criteria

pr_cols=['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']
i=0
while i < len(pr_cols):
    print('Best {0} : {1}'.format(pr_cols[i],df.loc[df[pr_cols[i]].idxmax()][1]))
    i += 1
""" Best Crossing : K. De Bruyne
Best Finishing : L. Messi
Best HeadingAccuracy : Naldo
Best ShortPassing : L. Modrić
Best Volleys : E. Cavani
Best Dribbling : L. Messi
Best Curve : Quaresma
Best FKAccuracy : L. Messi
Best LongPassing : T. Kroos
Best BallControl : L. Messi
Best Acceleration : Douglas Costa
Best SprintSpeed : K. Mbappé
Best Agility : Neymar Jr
Best Reactions : Cristiano Ronaldo
Best Balance : Bernard
Best ShotPower : Cristiano Ronaldo
Best Jumping : Cristiano Ronaldo
Best Stamina : N. Kanté
Best Strength : A. Akinfenwa
Best LongShots : L. Messi
Best Aggression : B. Pearson
Best Interceptions : N. Kanté
Best Positioning : Cristiano Ronaldo
Best Vision : L. Messi
Best Penalties : M. Balotelli
Best Composure : L. Messi
Best Marking : A. Barzagli
Best StandingTackle : G. Chiellini
Best SlidingTackle : Sergio Ramos
Best GKDiving : De Gea
Best GKHandling : J. Oblak
Best GKKicking : M. Neuer
Best GKPositioning : G. Buffon
Best GKReflexes : De Gea """

# creating a list of best players in each of the pr_cols criteria
i=0
best = []
while i < len(pr_cols):
    best.append(df.loc[df[pr_cols[i]].idxmax()][1])
    i +=1
best
""" ['K. De Bruyne',
 'L. Messi',
 'Naldo',
 'L. Modrić',
 'E. Cavani',
 'L. Messi',
 'Quaresma',
 'L. Messi',
 'T. Kroos',
 'L. Messi',
 'Douglas Costa',
 'K. Mbappé',
 'Neymar Jr',
 'Cristiano Ronaldo',
 'Bernard',
 'Cristiano Ronaldo',
 'Cristiano Ronaldo',
 'N. Kanté',
 'A. Akinfenwa',
 'L. Messi',
 'B. Pearson',
 'N. Kanté',
 'Cristiano Ronaldo',
 'L. Messi',
 'M. Balotelli',
 'L. Messi',
 'A. Barzagli',
 'G. Chiellini',
 'Sergio Ramos',
 'De Gea',
 'J. Oblak',
 'M. Neuer',
 'G. Buffon',
 'De Gea'] """

# Plot to show the preferred foot choice of different players
f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="Preferred Foot", hue="Real Face", data=df)
plt.show()

df.loc[df['Potential'].idxmax()][1] # Finding the player with the maximum potential
'K. Mbappé'
# showing the name of the players which occurs the most number of times from the first 20 names
plt.subplots(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='black',
                          width=1920,
                          height=1080
                         ).generate(" ".join(df.Name[0:20]))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('players.png')
plt.show()

df.columns # all the columns in the dataset
""" Index(['ID', 'Name', 'Age', 'Photo', 'Nationality', 'Flag', 'Overall',
       'Potential', 'Club', 'Club_Logo', 'Value', 'Wage', 'Special',
       'Preferred Foot', 'International Reputation', 'Weak Foot',
       'Skill Moves', 'Work Rate', 'Body Type', 'Real Face', 'Position',
       'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until',
       'Height', 'Weight', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Crossing',
       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause',
       'Defending', 'General', 'Mental', 'Passing', 'Mobility', 'Power',
       'Rating', 'Shooting'],
      dtype='object') """
      
# checking which clubs have been mentioned the most
plt.subplots(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='black',
                          width=1920,
                          height=1080
                         ).generate(" ".join(df.Club))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('players.png')
plt.show()

# showing the name of the players which occurs the most number of times(left join)
plt.subplots(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='black',
                          width=1920,
                          height=1080
                         ).generate(" ".join(left.Name))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('players.png')
plt.show()

#df.columns
# showing the name of the players which occurs the most number of times(right join)
plt.subplots(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='white',
                          width=1920,
                          height=1080
                         ).generate(" ".join(right.Name))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('players.png')
plt.show()

# Checking which player has been mentioned the most in the 'best' list that we have prepared
plt.subplots(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='white',
                          width=1920,
                          height=1080
                         ).generate(" ".join(best))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('players.png')
plt.show()

import requests
import random
from math import pi

import matplotlib.image as mpimg
from matplotlib.offsetbox import (OffsetImage,AnnotationBbox)

# defining a method to show the details of a player
def details(row, title, image, age, nationality, photo, logo, club):
    
    flag_image = "img_flag.jpg"
    player_image = "img_player.jpg"
    logo_image = "img_club_logo.jpg"
     
    # obtaining the player image, flag image and logo image
    img_flag = requests.get(image).content
    with open(flag_image, 'wb') as handler:
        handler.write(img_flag)
    
    player_img = requests.get(photo).content
    with open(player_image, 'wb') as handler:
        handler.write(player_img)
     
    logo_img = requests.get(logo).content
    with open(logo_image, 'wb') as handler:
        handler.write(logo_img)
     
    # Defining the colour schemes
    r = lambda: random.randint(0,255)
    colorRandom = '#%02X%02X%02X' % (r(),r(),r())
    
    if colorRandom == '#ffffff':colorRandom = '#a5d6a7' # if random colour  is white, assign a different colour
    
    basic_color = '#37474f'
    color_annotate = '#01579b'
    
    img = mpimg.imread(flag_image)
    
    plt.figure(figsize=(15,8))
    categories=list(players)[1:]
    coulumnDontUseGraph = ['Flag', 'Age', 'Nationality', 'Photo', 'Logo', 'Club']
    N = len(categories) - len(coulumnDontUseGraph)
    
    # adjusting the angles to show different aspects in the graph
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax = plt.subplot(111, projection='polar') # sepcifying a  polar graph type
    ax.set_theta_offset(pi / 2) # set the offset in radians
    ax.set_theta_direction(-1) #the angle increases in the clockwise direction
    plt.xticks(angles[:-1], categories, color= 'black', size=17)
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75,100], ["25","50","75","100"], color= basic_color, size= 10)
    plt.ylim(0,100)
    
    #creating the list of values which are not in (image, age, nationality, photo, logo, club) to show in the graph
    values = players.loc[row].drop('Name').values.flatten().tolist() 
    valuesDontUseGraph = [image, age, nationality, photo, logo, club]
    values = [e for e in values if e not in (valuesDontUseGraph)]
    values += values[:1]
    
    # customizing the graph attributes
    ax.plot(angles, values, color= basic_color, linewidth=1, linestyle='solid')
    ax.fill(angles, values, color= colorRandom, alpha=0.5)
    axes_coords = [0, 0, 1, 1]
    ax_image = plt.gcf().add_axes(axes_coords,zorder= -1)
    ax_image.imshow(img,alpha=0.5)
    ax_image.axis('off')
    
    # placeholders for showing nationality, age and team name
    ax.annotate('Nationality: ' + nationality.upper(), xy=(10,10), xytext=(103, 138),
                fontsize= 12,
                color = 'white',
                bbox={'facecolor': color_annotate, 'pad': 7})
                      
    ax.annotate('Age: ' + str(age), xy=(10,10), xytext=(43, 180),
                fontsize= 15,
                color = 'white',
                bbox={'facecolor': color_annotate, 'pad': 7})
    
    ax.annotate('Team: ' + club.upper(), xy=(10,10), xytext=(92, 168),
                fontsize= 12,
                color = 'white',
                bbox={'facecolor': color_annotate, 'pad': 7})

    # specifying the location for showing the image of player
    arr_img_player = plt.imread(player_image, format='jpg')
    imagebox_player = OffsetImage(arr_img_player)
    imagebox_player.image.axes = ax
    abPlayer = AnnotationBbox(imagebox_player, (0.5, 0.7),
                        xybox=(313, 223),
                        xycoords='data',
                        boxcoords="offset points"
                        )
    # specifying the location for showing the logo
    arr_img_logo = plt.imread(logo_image, format='jpg')
    
    imagebox_logo = OffsetImage(arr_img_logo)
    imagebox_logo.image.axes = ax
    abLogo = AnnotationBbox(imagebox_logo, (0.5, 0.7),
                        xybox=(-350, -246),
                        xycoords='data',
                        boxcoords="offset points"
                        )

    ax.add_artist(abPlayer)
    ax.add_artist(abLogo)

    plt.title(title, size=50, color= basic_color)
# defining a method to show the leading performers
def graphPolar(id = 0):
    if 0 <= id < len(data.ID):
        details(row = players.index[id], 
                title = players['Name'][id], 
                age = players['Age'][id], 
                photo = players['Photo'][id],
                nationality = players['Nationality'][id],
                image = players['Flag'][id], 
                logo = players['Club_Logo'][id], 
                club = players['Club'][id])
    else:
        print('The base has 17917 players. You can put positive numbers from 0 to 17917')
graphPolar(0)

graphPolar(1)

graphPolar(2)