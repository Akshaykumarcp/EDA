""" 
Overview of program
- import lib
- read dataset
- check null values
- basic lookup
    - view top rows for 10, 200 rows 
    - copy dataset 
    - check unique values of features
- feature engineering
    - create new features year, month and date 
- drop unwanted features
- convert datatype
- viz
    - Box plot of Calories with Jitter bu day of the month 
    - Barplot of calories by the day of the week
    - Scatterplot of calories and intense_activities
    - Scatterplot of calories vs Fairly Active Minutes
    - Un-normalized value of calories and different activities based on activity minutes 
    - Un-normalized value of calories and different activities based on distance"""

import numpy as np
import pandas as pd 
import seaborn as sns
import os
pd.pandas.set_option('display.max_columns',None)

activity = pd.read_csv('case_study7_fitbit/FitBit data.csv') # importing the dataset

# https://stackoverflow.com/questions/52553062/pandas-profiling-doesnt-display-the-output
import pandas_profiling
report = activity.profile_report() # seeing the full report about the dataset
report.to_file('case_study7_fitbit/profile_report.html')

activity.shape 
# (457, 15)

# check the number of missing values in the dataset
activity.isnull().sum() 
""" 
Id                          0
ActivityDate                0
TotalSteps                  0
TotalDistance               0
TrackerDistance             0
LoggedActivitiesDistance    0
VeryActiveDistance          0
ModeratelyActiveDistance    0
LightActiveDistance         0
SedentaryActiveDistance     0
VeryActiveMinutes           0
FairlyActiveMinutes         0
LightlyActiveMinutes        0
SedentaryMinutes            0
Calories                    0
dtype: int64 """

activity.head(10)
""" 
           Id ActivityDate  TotalSteps  TotalDistance  TrackerDistance  \
0  1503960366    3/25/2016       11004           7.11             7.11
1  1503960366    3/26/2016       17609          11.55            11.55
2  1503960366    3/27/2016       12736           8.53             8.53
3  1503960366    3/28/2016       13231           8.93             8.93
4  1503960366    3/29/2016       12041           7.85             7.85
5  1503960366    3/30/2016       10970           7.16             7.16
6  1503960366    3/31/2016       12256           7.86             7.86
7  1503960366     4/1/2016       12262           7.87             7.87
8  1503960366     4/2/2016       11248           7.25             7.25
9  1503960366     4/3/2016       10016           6.37             6.37

   LoggedActivitiesDistance  VeryActiveDistance  ModeratelyActiveDistance  \
0                       0.0                2.57                      0.46
1                       0.0                6.92                      0.73
2                       0.0                4.66                      0.16
3                       0.0                3.19                      0.79
4                       0.0                2.16                      1.09
5                       0.0                2.36                      0.51
6                       0.0                2.29                      0.49
7                       0.0                3.32                      0.83
8                       0.0                3.00                      0.45
9                       0.0                0.91                      1.28

   LightActiveDistance  SedentaryActiveDistance  VeryActiveMinutes  \
0                 4.07                      0.0                 33
1                 3.91                      0.0                 89
2                 3.71                      0.0                 56
3                 4.95                      0.0                 39
4                 4.61                      0.0                 28
5                 4.29                      0.0                 30
6                 5.04                      0.0                 33
7                 3.64                      0.0                 47
8                 3.74                      0.0                 40
9                 4.18                      0.0                 15

   FairlyActiveMinutes  LightlyActiveMinutes  SedentaryMinutes  Calories
0                   12                   205               804      1819
1                   17                   274               588      2154
2                    5                   268               605      1944
3                   20                   224              1080      1932
4                   28                   243               763      1886
5                   13                   223              1174      1820
6                   12                   239               820      1889
7                   21                   200               866      1868
8                   11                   244               636      1843
9                   30                   314               655      1850 """

# copying the datset to activity1
activity1 = activity.copy() 

# checking out the unique activity dates in the dataset
activity1['ActivityDate'].unique() 
""" array(['3/25/2016', '3/26/2016', '3/27/2016', '3/28/2016', '3/29/2016',
       '3/30/2016', '3/31/2016', '4/1/2016', '4/2/2016', '4/3/2016',
       '4/4/2016', '4/5/2016', '4/6/2016', '4/7/2016', '4/8/2016',
       '4/9/2016', '4/10/2016', '4/11/2016', '4/12/2016', '3/12/2016',
       '3/13/2016', '3/14/2016', '3/15/2016', '3/16/2016', '3/17/2016',
       '3/18/2016', '3/19/2016', '3/20/2016', '3/21/2016', '3/22/2016',
       '3/23/2016', '3/24/2016'], dtype=object) """

# cheking out the datset before transformation
activity1['ActivityDate'].head(10)  
""" 
0    3/25/2016
1    3/26/2016
2    3/27/2016
3    3/28/2016
4    3/29/2016
5    3/30/2016
6    3/31/2016
7     4/1/2016
8     4/2/2016
9     4/3/2016
Name: ActivityDate, dtype: object """

# adding the year month and date columns to the dataset
activity1['year'] = pd.DatetimeIndex(activity1['ActivityDate']).year
activity1['month'] = pd.DatetimeIndex(activity1['ActivityDate']).month
activity1['date'] = pd.DatetimeIndex(activity1['ActivityDate']).day

# cheking out the datset after adding year, month and day
activity1.head(10) 
""" 
           Id ActivityDate  TotalSteps  TotalDistance  TrackerDistance  \
0  1503960366    3/25/2016       11004           7.11             7.11
1  1503960366    3/26/2016       17609          11.55            11.55
2  1503960366    3/27/2016       12736           8.53             8.53
3  1503960366    3/28/2016       13231           8.93             8.93
4  1503960366    3/29/2016       12041           7.85             7.85
5  1503960366    3/30/2016       10970           7.16             7.16
6  1503960366    3/31/2016       12256           7.86             7.86
7  1503960366     4/1/2016       12262           7.87             7.87
8  1503960366     4/2/2016       11248           7.25             7.25
9  1503960366     4/3/2016       10016           6.37             6.37

   LoggedActivitiesDistance  VeryActiveDistance  ModeratelyActiveDistance  \
0                       0.0                2.57                      0.46
1                       0.0                6.92                      0.73
2                       0.0                4.66                      0.16
3                       0.0                3.19                      0.79
4                       0.0                2.16                      1.09
5                       0.0                2.36                      0.51
6                       0.0                2.29                      0.49
7                       0.0                3.32                      0.83
8                       0.0                3.00                      0.45
9                       0.0                0.91                      1.28

   LightActiveDistance  SedentaryActiveDistance  VeryActiveMinutes  \
0                 4.07                      0.0                 33
1                 3.91                      0.0                 89
2                 3.71                      0.0                 56
3                 4.95                      0.0                 39
4                 4.61                      0.0                 28
5                 4.29                      0.0                 30
6                 5.04                      0.0                 33
7                 3.64                      0.0                 47
8                 3.74                      0.0                 40
9                 4.18                      0.0                 15

   FairlyActiveMinutes  LightlyActiveMinutes  SedentaryMinutes  Calories  \
0                   12                   205               804      1819
1                   17                   274               588      2154
2                    5                   268               605      1944
3                   20                   224              1080      1932
4                   28                   243               763      1886
5                   13                   223              1174      1820
6                   12                   239               820      1889
7                   21                   200               866      1868
8                   11                   244               636      1843
9                   30                   314               655      1850

   year  month  date
0  2016      3    25
1  2016      3    26
2  2016      3    27
3  2016      3    28
4  2016      3    29
5  2016      3    30
6  2016      3    31
7  2016      4     1
8  2016      4     2
9  2016      4     3 """

# dropping the TrackerDistance column
activity1=activity1.drop(['TrackerDistance'],axis=1)  

# checking out the first 200 rows of the datset after transformation
activity1.head(200) 
""" 
             Id ActivityDate  TotalSteps  TotalDistance  \
0    1503960366    3/25/2016       11004           7.11
1    1503960366    3/26/2016       17609          11.55
2    1503960366    3/27/2016       12736           8.53
3    1503960366    3/28/2016       13231           8.93
4    1503960366    3/29/2016       12041           7.85
..          ...          ...         ...            ...
195  4020332650    4/11/2016        2993           2.15
196  4020332650    4/12/2016           8           0.01
197  4057192912    3/12/2016           0           0.00
198  4057192912    3/13/2016           0           0.00
199  4057192912    3/14/2016        8433           6.23

     LoggedActivitiesDistance  VeryActiveDistance  ModeratelyActiveDistance  \
0                         0.0                2.57                      0.46
1                         0.0                6.92                      0.73
2                         0.0                4.66                      0.16
3                         0.0                3.19                      0.79
4                         0.0                2.16                      1.09
..                        ...                 ...                       ...
195                       0.0                0.00                      0.00
196                       0.0                0.00                      0.00
197                       0.0                0.00                      0.00
198                       0.0                0.00                      0.00
199                       0.0                2.45                      0.33

     LightActiveDistance  SedentaryActiveDistance  VeryActiveMinutes  \
0                   4.07                      0.0                 33
1                   3.91                      0.0                 89
2                   3.71                      0.0                 56
3                   4.95                      0.0                 39
4                   4.61                      0.0                 28
..                   ...                      ...                ...
195                 2.09                      0.0                  0
196                 0.01                      0.0                  0
197                 0.00                      0.0                  0
198                 0.00                      0.0                  0
199                 3.44                      0.0                 30

     FairlyActiveMinutes  LightlyActiveMinutes  SedentaryMinutes  Calories  \
0                     12                   205               804      1819
1                     17                   274               588      2154
2                      5                   268               605      1944
3                     20                   224              1080      1932
4                     28                   243               763      1886
..                   ...                   ...               ...       ...
195                    0                   114               888      2507
196                    0                     1               321       446
197                    0                     0              1440      1777
198                    0                     0              1440      1777
199                    7                   135              1268      2453

     year  month  date
0    2016      3    25
1    2016      3    26
2    2016      3    27
3    2016      3    28
4    2016      3    29
..    ...    ...   ...
195  2016      4    11
196  2016      4    12
197  2016      3    12
198  2016      3    13
199  2016      3    14

[200 rows x 17 columns] """

### Groupby the day of the month and make a boxplot of calories burnt
import matplotlib.pyplot as plt

# figure size
plt.figure(figsize=(15,8))

# Usual boxplot
ax = sns.boxplot(x='date', y='Calories', data=activity1)
 
# Add jitter with the swarmplot function.
ax = sns.swarmplot(x='date', y='Calories', data=activity1, color="grey")

ax.set_title('Box plot of Calories with Jitter bu day of the month')
plt.show()

# converting the datatype to datetime
activity1['Week'] = pd.to_datetime(activity1.ActivityDate).dt.week
activity1['Year'] = pd.to_datetime(activity1.ActivityDate).dt.year

activity1.head()  # cheking out the datset after transformation
""" 
           Id ActivityDate  TotalSteps  TotalDistance  \
0  1503960366    3/25/2016       11004           7.11
1  1503960366    3/26/2016       17609          11.55
2  1503960366    3/27/2016       12736           8.53
3  1503960366    3/28/2016       13231           8.93
4  1503960366    3/29/2016       12041           7.85

   LoggedActivitiesDistance  VeryActiveDistance  ModeratelyActiveDistance  \
0                       0.0                2.57                      0.46
1                       0.0                6.92                      0.73
2                       0.0                4.66                      0.16
3                       0.0                3.19                      0.79
4                       0.0                2.16                      1.09

   LightActiveDistance  SedentaryActiveDistance  VeryActiveMinutes  \
0                 4.07                      0.0                 33
1                 3.91                      0.0                 89
2                 3.71                      0.0                 56
3                 4.95                      0.0                 39
4                 4.61                      0.0                 28

   FairlyActiveMinutes  LightlyActiveMinutes  SedentaryMinutes  Calories  \
0                   12                   205               804      1819
1                   17                   274               588      2154
2                    5                   268               605      1944
3                   20                   224              1080      1932
4                   28                   243               763      1886

   year  month  date  Week  Year
0  2016      3    25    12  2016
1  2016      3    26    12  2016
2  2016      3    27    12  2016
3  2016      3    28    13  2016
4  2016      3    29    13  2016 """

# cheking the datatype of ActivityDate field
activity1.ActivityDate.dtype 
# dtype('O')

# converting it to datetime
activity1['ActivityDate'] = pd.to_datetime(activity1['ActivityDate'])

# converting the day of the week to the name of the day
activity1['day'] = activity1['ActivityDate'].dt.day_name() 

# cheking out the datset after transformation
activity1.head(10) 
""" 
   year  month  date  Week  Year        day
0  2016      3    25    12  2016     Friday
1  2016      3    26    12  2016   Saturday
2  2016      3    27    12  2016     Sunday
3  2016      3    28    13  2016     Monday
4  2016      3    29    13  2016    Tuesday
5  2016      3    30    13  2016  Wednesday
6  2016      3    31    13  2016   Thursday
7  2016      4     1    13  2016     Friday
8  2016      4     2    13  2016   Saturday
9  2016      4     3    13  2016     Sunday """

# figure size
plt.figure(figsize=(15,8))

# simple barplot
ax = sns.barplot(x='day', y='Calories',  data=activity1)

ax.set_title('Barplot of calories by the day of the week')
plt.show()

# figure size
plt.figure(figsize=(15,8))

# Simple scatterplot
ax = sns.scatterplot(x='Calories', y='SedentaryMinutes', data=activity1)

ax.set_title('Scatterplot of calories and intense_activities')
plt.show()

# figure size
plt.figure(figsize=(15,8))

# Simple scatterplot
ax = sns.scatterplot(x='Calories', y='LightlyActiveMinutes', data=activity1)

ax.set_title('Scatterplot of calories and intense_activities')
plt.show()

# figure size
plt.figure(figsize=(15,8))

# Simple scatterplot between calories burnt in the moderately active minutes
ax = sns.scatterplot(x='Calories', y='FairlyActiveMinutes', data=activity1)

ax.set_title('Scatterplot of calories vs Fairly Active Minutes')
plt.show()

# figure size
plt.figure(figsize=(15,8))

# Simple scatterplot between calories burnt in the intensely active minutes
ax = sns.scatterplot(x='Calories', y='VeryActiveMinutes', data=activity1)

ax.set_title('Scatterplot of calories and intense_activities')
plt.show()

activity1.head(10) # cheking out the datset before transformation
""" 
           Id ActivityDate  TotalSteps  TotalDistance  \
0  1503960366   2016-03-25       11004           7.11
1  1503960366   2016-03-26       17609          11.55
2  1503960366   2016-03-27       12736           8.53
3  1503960366   2016-03-28       13231           8.93
4  1503960366   2016-03-29       12041           7.85
5  1503960366   2016-03-30       10970           7.16
6  1503960366   2016-03-31       12256           7.86
7  1503960366   2016-04-01       12262           7.87
8  1503960366   2016-04-02       11248           7.25
9  1503960366   2016-04-03       10016           6.37

   LoggedActivitiesDistance  VeryActiveDistance  ModeratelyActiveDistance  \
0                       0.0                2.57                      0.46
1                       0.0                6.92                      0.73
2                       0.0                4.66                      0.16
3                       0.0                3.19                      0.79
4                       0.0                2.16                      1.09
5                       0.0                2.36                      0.51
6                       0.0                2.29                      0.49
7                       0.0                3.32                      0.83
8                       0.0                3.00                      0.45
9                       0.0                0.91                      1.28

   LightActiveDistance  SedentaryActiveDistance  VeryActiveMinutes  \
0                 4.07                      0.0                 33
1                 3.91                      0.0                 89
2                 3.71                      0.0                 56
3                 4.95                      0.0                 39
4                 4.61                      0.0                 28
5                 4.29                      0.0                 30
6                 5.04                      0.0                 33
7                 3.64                      0.0                 47
8                 3.74                      0.0                 40
9                 4.18                      0.0                 15

   FairlyActiveMinutes  LightlyActiveMinutes  SedentaryMinutes  Calories  \
0                   12                   205               804      1819
1                   17                   274               588      2154
2                    5                   268               605      1944
3                   20                   224              1080      1932
4                   28                   243               763      1886
5                   13                   223              1174      1820
6                   12                   239               820      1889
7                   21                   200               866      1868
8                   11                   244               636      1843
9                   30                   314               655      1850

   year  month  date  Week  Year        day
0  2016      3    25    12  2016     Friday
1  2016      3    26    12  2016   Saturday
2  2016      3    27    12  2016     Sunday
3  2016      3    28    13  2016     Monday
4  2016      3    29    13  2016    Tuesday
5  2016      3    30    13  2016  Wednesday
6  2016      3    31    13  2016   Thursday
7  2016      4     1    13  2016     Friday
8  2016      4     2    13  2016   Saturday
9  2016      4     3    13  2016     Sunday """

# dropping the columns week and year
activity1=activity1.drop(['Week','Year'],axis=1) 

# cheking out the datset after transformation
activity1.head(10) 
""" 
           Id ActivityDate  TotalSteps  TotalDistance  \
0  1503960366   2016-03-25       11004           7.11
1  1503960366   2016-03-26       17609          11.55
2  1503960366   2016-03-27       12736           8.53
3  1503960366   2016-03-28       13231           8.93
4  1503960366   2016-03-29       12041           7.85
5  1503960366   2016-03-30       10970           7.16
6  1503960366   2016-03-31       12256           7.86
7  1503960366   2016-04-01       12262           7.87
8  1503960366   2016-04-02       11248           7.25
9  1503960366   2016-04-03       10016           6.37

   LoggedActivitiesDistance  VeryActiveDistance  ModeratelyActiveDistance  \
0                       0.0                2.57                      0.46
1                       0.0                6.92                      0.73
2                       0.0                4.66                      0.16
3                       0.0                3.19                      0.79
4                       0.0                2.16                      1.09
5                       0.0                2.36                      0.51
6                       0.0                2.29                      0.49
7                       0.0                3.32                      0.83
8                       0.0                3.00                      0.45
9                       0.0                0.91                      1.28

   LightActiveDistance  SedentaryActiveDistance  VeryActiveMinutes  \
0                 4.07                      0.0                 33
1                 3.91                      0.0                 89
2                 3.71                      0.0                 56
3                 4.95                      0.0                 39
4                 4.61                      0.0                 28
5                 4.29                      0.0                 30
6                 5.04                      0.0                 33
7                 3.64                      0.0                 47
8                 3.74                      0.0                 40
9                 4.18                      0.0                 15

   FairlyActiveMinutes  LightlyActiveMinutes  SedentaryMinutes  Calories  \
0                   12                   205               804      1819
1                   17                   274               588      2154
2                    5                   268               605      1944
3                   20                   224              1080      1932
4                   28                   243               763      1886
5                   13                   223              1174      1820
6                   12                   239               820      1889
7                   21                   200               866      1868
8                   11                   244               636      1843
9                   30                   314               655      1850

   year  month  date        day
0  2016      3    25     Friday
1  2016      3    26   Saturday
2  2016      3    27     Sunday
3  2016      3    28     Monday
4  2016      3    29    Tuesday
5  2016      3    30  Wednesday
6  2016      3    31   Thursday
7  2016      4     1     Friday
8  2016      4     2   Saturday
9  2016      4     3     Sunday """

# checking the number of rows and columns in the transformed  dataset
activity1.shape 
# (457, 18)

## plot the raw values 
col_select = ['Calories','VeryActiveMinutes','FairlyActiveMinutes','LightlyActiveMinutes','SedentaryMinutes']
wide_df = activity1[col_select]

# figure size
plt.figure(figsize=(15,8))

# timeseries plot using lineplot
ax = sns.lineplot(data=wide_df)

ax.set_title('Un-normalized value of calories and different activities based on activity minutes')
plt.show()

# figure size
plt.figure(figsize=(15,8))

# Simple scatterplot between  calories burnt and total distance covered
ax = sns.scatterplot(x='Calories', y='TotalDistance', data=activity1)

ax.set_title('Scatterplot of calories and intense_activities')
plt.show()

# figure size
plt.figure(figsize=(15,8))

# Simple scatterplot between calories burnt and the loggged activities distance
ax = sns.scatterplot(x='Calories', y='LoggedActivitiesDistance', data=activity1)

ax.set_title('Scatterplot of calories and intense_activities')
plt.show()

# figure size
plt.figure(figsize=(15,8))

# Simple scatterplot between calories burnt and the distance of intense activies
ax = sns.scatterplot(x='Calories', y='VeryActiveDistance', data=activity1)

ax.set_title('Scatterplot of calories and intense_activities')
plt.show()

# figure size
plt.figure(figsize=(15,8))

# Simple scatterplot between calories burnt and the distance of moderate activies
ax = sns.scatterplot(x='Calories', y='ModeratelyActiveDistance', data=activity1)

ax.set_title('Scatterplot of calories and intense_activities')
plt.show()

# figure size
plt.figure(figsize=(15,8))

# Simple scatterplot
ax = sns.scatterplot(x='Calories', y='LightActiveDistance', data=activity1)

ax.set_title('Scatterplot of calories and intense_activities')
plt.show()

## plot the raw values 

rol_select = ['TotalDistance','LoggedActivitiesDistance','VeryActiveDistance','ModeratelyActiveDistance', 'LightActiveDistance']
wide_df1 = activity1[rol_select]

# figure size
plt.figure(figsize=(15,8))

# timeseries plot using lineplot
ax = sns.lineplot(data=wide_df1)

ax.set_title('Un-normalized value of calories and different activities based on distance')
plt.show()

""" 
- The EDA here gives us the insight about the relation between the active hours, the distance for which the user has
         moderate and intense activity and the calories burnt during that period. """

