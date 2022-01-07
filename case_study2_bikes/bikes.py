""" 
Problem Statement- Bike-sharing system are meant to rent the bicycle and return to the different place 
    for the bike sharing purpose in Washington DC. 
    You are provided with rental data spanning for 2 years. 
    You must predict the total count of bikes rented during each hour covered by the test set, using only information
    available prior to the rental period.
 """

#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Qt5Agg')
train = pd.read_csv('case_study2_bikes/train_bikes.csv', parse_dates=['datetime']) # loading the training data

train.head()
""" 
             datetime  season  holiday  workingday  weather  temp   atemp  humidity  windspeed  casual  registered  count
0 2011-01-01 00:00:00       1        0           0        1  9.84  14.395        81        0.0       3          13     16
1 2011-01-01 01:00:00       1        0           0        1  9.02  13.635        80        0.0       8          32     40
2 2011-01-01 02:00:00       1        0           0        1  9.02  13.635        80        0.0       5          27     32
3 2011-01-01 03:00:00       1        0           0        1  9.84  14.395        75        0.0       3          10     13
4 2011-01-01 04:00:00       1        0           0        1  9.84  14.395        75        0.0       0           1      1 """

train.tail() # looking at the training data from end
""" 
                 datetime  season  holiday  workingday  weather   temp   atemp  humidity  windspeed  casual  registered  count
10881 2012-12-19 19:00:00       4        0           1        1  15.58  19.695        50    26.0027       7         329    336
10882 2012-12-19 20:00:00       4        0           1        1  14.76  17.425        57    15.0013      10         231    241
10883 2012-12-19 21:00:00       4        0           1        1  13.94  15.910        61    15.0013       4         164    168
10884 2012-12-19 22:00:00       4        0           1        1  13.94  17.425        61     6.0032      12         117    129
10885 2012-12-19 23:00:00       4        0           1        1  13.12  16.665        66     8.9981       4          84     88 """

train.season.value_counts()
""" 
4    2734
2    2733
3    2733
1    2686
Name: season, dtype: int64 """

# plotting the counts based on the season
train.plot.scatter(x = 'season', y = 'count')
plt.show()

train.plot.scatter(x = 'holiday', y = 'count') # plotting the counts based on the holidays
plt.show()

train.plot.scatter(x = 'workingday', y = 'count') # plotting the counts based on working day
plt.show()

train.plot.scatter(x = 'weather', y = 'count') # plotting the counts based on the weather
plt.show()

train.plot.scatter(x = 'temp', y = 'count') # plotting the counts based on the temparature
plt.show()

train.plot.scatter(x = 'atemp', y = 'count') # plotting the counts based on atemp
plt.show()

train.plot.scatter(x = 'humidity', y = 'count')# plotting the counts based on humidity
plt.show()

train.plot.scatter(x = 'windspeed', y = 'count') # plotting the counts based on windspeed
plt.show()

train.plot.scatter(x = 'casual', y = 'count')# plotting the counts based casual user
plt.show()

train.info() # observing the data types of the columns
""" 
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10886 entries, 0 to 10885
Data columns (total 12 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   datetime    10886 non-null  datetime64[ns]
 1   season      10886 non-null  int64
 2   holiday     10886 non-null  int64
 3   workingday  10886 non-null  int64
 4   weather     10886 non-null  int64
 5   temp        10886 non-null  float64
 6   atemp       10886 non-null  float64
 7   humidity    10886 non-null  int64
 8   windspeed   10886 non-null  float64
 9   casual      10886 non-null  int64
 10  registered  10886 non-null  int64
 11  count       10886 non-null  int64
dtypes: datetime64[ns](1), float64(3), int64(8)
memory usage: 1020.7 KB """

train.describe() # Generate descriptive statistics that summarize the central tendency,dispersion and shape of a dataset's distribution
""" 
             season       holiday    workingday       weather         temp         atemp      humidity     windspeed        casual    registered         count
count  10886.000000  10886.000000  10886.000000  10886.000000  10886.00000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000
mean       2.506614      0.028569      0.680875      1.418427     20.23086     23.655084     61.886460     12.799395     36.021955    155.552177    191.574132
std        1.116174      0.166599      0.466159      0.633839      7.79159      8.474601     19.245033      8.164537     49.960477    151.039033    181.144454
min        1.000000      0.000000      0.000000      1.000000      0.82000      0.760000      0.000000      0.000000      0.000000      0.000000      1.000000
25%        2.000000      0.000000      0.000000      1.000000     13.94000     16.665000     47.000000      7.001500      4.000000     36.000000     42.000000
50%        3.000000      0.000000      1.000000      1.000000     20.50000     24.240000     62.000000     12.998000     17.000000    118.000000    145.000000
75%        4.000000      0.000000      1.000000      2.000000     26.24000     31.060000     77.000000     16.997900     49.000000    222.000000    284.000000
max        4.000000      1.000000      1.000000      4.000000     41.00000     45.455000    100.000000     56.996900    367.000000    886.000000    977.000000 """

test = pd.read_csv('case_study2_bikes/test_bikes.csv') # loading the test data

test.head()  #looking at the 1st 5 rows of the test data
""" 
              datetime  season  holiday  workingday  weather   temp   atemp  humidity  windspeed
0  2011-01-20 00:00:00       1        0           1        1  10.66  11.365        56    26.0027
1  2011-01-20 01:00:00       1        0           1        1  10.66  13.635        56     0.0000
2  2011-01-20 02:00:00       1        0           1        1  10.66  13.635        56     0.0000
3  2011-01-20 03:00:00       1        0           1        1  10.66  12.880        56    11.0014
4  2011-01-20 04:00:00       1        0           1        1  10.66  12.880        56    11.0014 """

test.tail() # last 5 rows of the test data
""" 
                 datetime  season  holiday  workingday  weather   temp   atemp  humidity  windspeed
6488  2012-12-31 19:00:00       1        0           1        2  10.66  12.880        60    11.0014
6489  2012-12-31 20:00:00       1        0           1        2  10.66  12.880        60    11.0014
6490  2012-12-31 21:00:00       1        0           1        1  10.66  12.880        60    11.0014
6491  2012-12-31 22:00:00       1        0           1        1  10.66  13.635        56     8.9981
6492  2012-12-31 23:00:00       1        0           1        1  10.66  13.635        65     8.9981 """

test.info() # observing the data types of the columns for test data
""" 
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6493 entries, 0 to 6492
Data columns (total 9 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   datetime    6493 non-null   object
 1   season      6493 non-null   int64
 2   holiday     6493 non-null   int64
 3   workingday  6493 non-null   int64
 4   weather     6493 non-null   int64
 5   temp        6493 non-null   float64
 6   atemp       6493 non-null   float64
 7   humidity    6493 non-null   int64
 8   windspeed   6493 non-null   float64
dtypes: float64(3), int64(5), object(1)
memory usage: 456.7+ KB """

test.describe() # Generate descriptive statistics that summarize the central tendency,dispersion and shape of a dataset's distribution for test data
""" 
            season      holiday   workingday      weather         temp        atemp     humidity    windspeed
count  6493.000000  6493.000000  6493.000000  6493.000000  6493.000000  6493.000000  6493.000000  6493.000000
mean      2.493300     0.029108     0.685815     1.436778    20.620607    24.012865    64.125212    12.631157
std       1.091258     0.168123     0.464226     0.648390     8.059583     8.782741    19.293391     8.250151
min       1.000000     0.000000     0.000000     1.000000     0.820000     0.000000    16.000000     0.000000
25%       2.000000     0.000000     0.000000     1.000000    13.940000    16.665000    49.000000     7.001500
50%       3.000000     0.000000     1.000000     1.000000    21.320000    25.000000    65.000000    11.001400
75%       3.000000     0.000000     1.000000     2.000000    27.060000    31.060000    81.000000    16.997900
max       4.000000     1.000000     1.000000     4.000000    40.180000    50.000000   100.000000    55.998600 """

# installing the pandas profiling library. It is used for a deeper understanding than the normal 
# Dataframe.describe() method
# pip install pandas_profiling
import pandas_profiling
train.profile_report()

print("count samples & features: ", train.shape) # printing the number of rows and columns
print("Are there missing values: ", train.isnull().values.any()) # printing if dataset has any NaN value
""" 
count samples & features:  (10886, 12)
Are there missing values:  False """

# method for creating the count plot based on hour for a given year 
def plot_by_hour(data, year=None, agg='sum'):
    dd = data
    if year: dd = dd[ dd.datetime.dt.year == year ]
    dd.loc[:, ('hour')] = dd.datetime.dt.hour # extracting the hour data if the year in the data is equal to the year passed as argument
    
    by_hour = dd.groupby(['hour', 'workingday'])['count'].agg(agg).unstack() # groupby hour and working day
    by_hour.plot(kind='bar', ylim=(0, 80000), figsize=(15,5), width=0.9, title="Year = {0}".format(year))  # returning the figure grouped by hour
    plt.show()

plot_by_hour(train, year=2011)  # plotting the count plot based on hour for 2011 

plot_by_hour(train, year=2012) # plotting the count plot based on hour for 2012

# method for creating the count plot based on year 
def plot_by_year(agg_attr, title):
    # extracting the required fields
    dd = train.copy()
    dd['year'] = train.datetime.dt.year # extratcing the year
    dd['month'] = train.datetime.dt.month # extratcing the month
    dd['hour'] = train.datetime.dt.hour # extratcing the hour
    
    by_year = dd.groupby([agg_attr, 'year'])['count'].agg('sum').unstack() # groupby year
    by_year.plot(kind='bar', figsize=(15,5), width=0.9, title=title) # returning the figure grouped by year
    plt.show()


plot_by_year('month', "Rent bikes per month in 2011 and 2012") # plotting monthly bike rentals based on year
plot_by_year('hour', "Rent bikes per hour in 2011 and 2012") # plotting hourls bike rentals based  on year

# method to plot a graph for count per hour
def plot_hours(data, message = ''):
    dd = data.copy()
    dd['hour'] = data.datetime.dt.hour # extratcing the hour
    
    hours = {}
    for hour in range(24):
        hours[hour] = dd[ dd.hour == hour ]['count'].values

    plt.figure(figsize=(20,10))
    plt.ylabel("Count rent")
    plt.xlabel("Hours")
    plt.title("count vs hours\n" + message)
    plt.boxplot( [hours[hour] for hour in range(24)] )
    
    axis = plt.gca()
    axis.set_ylim([1, 1100])

    plt.show()
 
plot_hours( train[train.datetime.dt.year == 2011], 'year 2011') # box plot for hourly count for the mentioned year
plot_hours( train[train.datetime.dt.year == 2012], 'year 2012') # box plot for hourly count for the mentioned year

dt = pd.to_datetime(train["datetime"]) # converting the column to datetime for train dataset

train["hour"] = dt.map(lambda x: x.hour) # adding the hour column for train dataset

train.head()
""" datetime  season  holiday  workingday  weather  temp   atemp  humidity  windspeed  casual  registered  count  hour
0 2011-01-01 00:00:00       1        0           0        1  9.84  14.395        81        0.0       3          13     16     0
1 2011-01-01 01:00:00       1        0           0        1  9.02  13.635        80        0.0       8          32     40     1
2 2011-01-01 02:00:00       1        0           0        1  9.02  13.635        80        0.0       5          27     32     2
3 2011-01-01 03:00:00       1        0           0        1  9.84  14.395        75        0.0       3          10     13     3
4 2011-01-01 04:00:00       1        0           0        1  9.84  14.395        75        0.0       0           1      1     4 """

dt_test = pd.to_datetime(test["datetime"]) # converting the column to datetime for test dataset
test["hour"] = dt_test.map(lambda x: x.hour) # adding the hour column for test dataset
test.head()
""" 
              datetime  season  holiday  workingday  weather   temp   atemp  humidity  windspeed  hour
0  2011-01-20 00:00:00       1        0           1        1  10.66  11.365        56    26.0027     0
1  2011-01-20 01:00:00       1        0           1        1  10.66  13.635        56     0.0000     1
2  2011-01-20 02:00:00       1        0           1        1  10.66  13.635        56     0.0000     2
3  2011-01-20 03:00:00       1        0           1        1  10.66  12.880        56    11.0014     3
4  2011-01-20 04:00:00       1        0           1        1  10.66  12.880        56    11.0014     4 """

plot_hours( train[train.workingday == 1], 'working day') # plotting hourly count of rented bikes for working days for a given year
plot_hours( train[train.workingday == 0], 'non working day') # plotting hourly count of rented bikes for non-working days for a given year

# method to convert categorical data to numerical data
def categorical_to_numeric(x):
    if 0 <=  x < 6:
        return 0
    elif 6 <= x < 13:
        return 1
    elif 13 <= x < 19:
        return 2
    elif 19 <= x < 24:
        return 3
train['hour'] = train['hour'].apply(categorical_to_numeric)# applying the above conversion logic to training data

train.head()
""" 
             datetime  season  holiday  workingday  weather  temp   atemp  humidity  windspeed  casual  registered  count  hour
0 2011-01-01 00:00:00       1        0           0        1  9.84  14.395        81        0.0       3          13     16     0
1 2011-01-01 01:00:00       1        0           0        1  9.02  13.635        80        0.0       8          32     40     0
2 2011-01-01 02:00:00       1        0           0        1  9.02  13.635        80        0.0       5          27     32     0
3 2011-01-01 03:00:00       1        0           0        1  9.84  14.395        75        0.0       3          10     13     0
4 2011-01-01 04:00:00       1        0           0        1  9.84  14.395        75        0.0       0           1      1     0"""

test['hour'] = test['hour'].apply(categorical_to_numeric) # applying the above conversion logic to test data

test.head()
""" 
              datetime  season  holiday  workingday  weather   temp   atemp  humidity  windspeed  hour
0  2011-01-20 00:00:00       1        0           1        1  10.66  11.365        56    26.0027     0
1  2011-01-20 01:00:00       1        0           1        1  10.66  13.635        56     0.0000     0
2  2011-01-20 02:00:00       1        0           1        1  10.66  13.635        56     0.0000     0
3  2011-01-20 03:00:00       1        0           1        1  10.66  12.880        56    11.0014     0
4  2011-01-20 04:00:00       1        0           1        1  10.66  12.880        56    11.0014     0 """

# drop unnecessary columns
train = train.drop(['datetime'], axis=1)
test = test.drop(['datetime'], axis=1)
train.head()
""" 
   season  holiday  workingday  weather  temp   atemp  humidity  windspeed  casual  registered  count  hour
0       1        0           0        1  9.84  14.395        81        0.0       3          13     16     0
1       1        0           0        1  9.02  13.635        80        0.0       8          32     40     0
2       1        0           0        1  9.02  13.635        80        0.0       5          27     32     0
3       1        0           0        1  9.84  14.395        75        0.0       3          10     13     0
4       1        0           0        1  9.84  14.395        75        0.0       0           1      1     0 """

# an Hour bs Count Graph depicting average bike demand based on the hour 
figure,axes = plt.subplots(figsize = (10, 5))
hours = train.groupby(["hour"]).agg("mean")["count"]  
hours.plot(kind="line", ax=axes) 
plt.title('Hours VS Counts')
axes.set_xlabel('Time in Hours')
axes.set_ylabel('Average of the Bike Demand')
plt.show()

# count of different temp values
a = train.groupby('temp')[['count']].mean()
a
""" count
temp	
0.82	77.714286
1.64	91.500000
2.46	43.000000
3.28	19.272727
4.10	50.272727
4.92	58.416667
5.74	53.233645
6.56	68.109589
7.38	67.754717
8.20	81.995633
9.02	73.616935
9.84	86.442177
10.66	92.560241
11.48	111.066298
12.30	120.002597
13.12	148.547753
13.94	145.053269
14.76	152.957173
15.58	179.682353
16.40	170.217500
17.22	182.609551
18.04	160.878049
18.86	159.692118
19.68	185.058824
20.50	204.672783
21.32	196.480663
22.14	184.717122
22.96	212.392405
23.78	235.650246
24.60	237.182051
25.42	222.062035
26.24	232.403974
27.06	211.025381
27.88	203.433036
28.70	257.679157
29.52	277.691218
30.34	303.193980
31.16	352.801653
31.98	318.683673
32.80	355.623762
33.62	348.323077
34.44	340.225000
35.26	342.934211
36.08	362.869565
36.90	318.717391
37.72	332.176471
38.54	238.857143
39.36	317.833333
41.00	294.000000 """

a.plot()
plt.show()

# count of different atemp values
a = train.groupby('atemp')[['count']].mean()
a
""" count
atemp	
0.760	1.000000
1.515	3.000000
2.275	38.000000
3.030	82.285714
3.790	39.062500
4.545	66.090909
5.305	63.200000
6.060	64.876712
6.820	56.380952
7.575	55.933333
8.335	58.444444
9.090	80.000000
9.850	81.456693
10.605	95.951807
11.365	90.442804
12.120	102.656410
12.880	89.518219
13.635	94.308017
14.395	116.483271
15.150	133.967456
15.910	133.897638
16.665	148.509186
17.425	147.799363
18.180	133.585366
18.940	149.555556
19.695	179.682353
20.455	170.217500
21.210	182.609551
21.970	160.878049
22.725	159.692118
23.485	185.058824
24.240	204.672783
25.000	195.109589
25.760	179.626478
26.515	212.392405
27.275	200.503546
28.030	133.312500
28.790	142.771429
29.545	151.046693
30.305	227.291429
31.060	308.323398
31.820	258.655518
32.575	331.746324
33.335	244.107143
34.090	295.183036
34.850	277.448763
35.605	312.144654
36.365	349.243902
37.120	334.144068
37.880	351.835052
38.635	335.783784
39.395	319.194030
40.150	369.577778
40.910	324.512821
41.665	281.434783
42.425	301.958333
43.180	307.142857
43.940	215.428571
44.695	354.333333
45.455	312.000000 """

a.plot()
plt.show()

# count based on holiday
a = train.groupby('holiday')[['count']].mean()
a.plot()
plt.show()

# method to  select the features. If a feature is not in the blaklist, it gets selected
def select_features(data):
    black_list = ['casual', 'registered', 'count', 'is_test', 'datetime', 'count_log']
    return [feat for feat in data.columns if feat not in black_list]

from sklearn.dummy import DummyRegressor

# a method to show results of various model and their predictions
def _simple_modeling(X_train, X_test, y_train, y_test):
    # sepcifying the model names
    models = [
        ('dummy-mean', DummyRegressor(strategy='mean')),
        ('dummy-median', DummyRegressor(strategy='median')),
        ('random-forest', RandomForestRegressor(random_state=0)),
    ]
    
    results = []

    for name, model in models:
        model.fit(X_train, y_train)# fitting the training data to model
        y_pred = model.predict(X_test) # doing predictions using the model
        
        results.append((name, y_test, y_pred)) # creating the list of predictions from various models
        
    return results

from sklearn.metrics import mean_squared_log_error as rmsle

# a method to return the performance metric of the model used in the above method
def simple_modeling(X_train, X_test, y_train, y_test):
    results = _simple_modeling(X_train, X_test, y_train, y_test) # using the function defined above to caluclate the predictions
    
    return [ (r[0], rmsle(r[1], r[2]) ) for r in results] # returning the performance metrics

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

forest_reg = RandomForestRegressor(n_estimators=100) # instantiating the random Forest Regressor

score = cross_val_score(forest_reg, train, train, cv=4) # calcuating the cross validation score

print (score)
""" [0.99516611 0.99725277 0.99769427 0.9973147 ] """
