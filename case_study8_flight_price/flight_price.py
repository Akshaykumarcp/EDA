""" 
Summary of program:
- import lib's
- read datasets
- EDA about dataset
    - list columns/features
    - No of features 
    - No of rows
    - Datatypes of features
    - Contain null values ? Boolean
    - Null counts by features
- Drop unwanted features
- Feature engineering
    - New features i,e Journey_Day, Journey_Month 
- convert featues to appropriate datatype and formats (hour, minute, etc)
- divide dataset into independent and dependent features
- convert categorical values to numerical values
- feature scaling"""


import pandas as pd
pd.pandas.set_option('display.max_columns',None)

training_set = pd.read_excel("case_study8_flight_price/Data_Train.xlsx")
test_set = pd.read_excel("case_study8_flight_price/Test_set.xlsx")

# seeing how the training data looks
training_set.head() 
""" 
       Airline Date_of_Journey    Source Destination                  Route  \
0       IndiGo      24/03/2019  Banglore   New Delhi              BLR → DEL   
1    Air India       1/05/2019   Kolkata    Banglore  CCU → IXR → BBI → BLR   
2  Jet Airways       9/06/2019     Delhi      Cochin  DEL → LKO → BOM → COK
3       IndiGo      12/05/2019   Kolkata    Banglore        CCU → NAG → BLR
4       IndiGo      01/03/2019  Banglore   New Delhi        BLR → NAG → DEL

  Dep_Time  Arrival_Time Duration Total_Stops Additional_Info  Price
0    22:20  01:10 22 Mar   2h 50m    non-stop         No info   3897
1    05:50         13:15   7h 25m     2 stops         No info   7662
2    09:25  04:25 10 Jun      19h     2 stops         No info  13882
3    18:05         23:30   5h 25m      1 stop         No info   6218
4    16:50         21:35   4h 45m      1 stop         No info  13302 """

# priniting the details about the datasets
print("\nEDA on Training Set\n")
print("#"*30)
print("\nFeatures/Columns : \n", training_set.columns)
print("\n\nNumber of Features/Columns : ", len(training_set.columns))
print("\nNumber of Rows : ",len(training_set))
print("\n\nData Types :\n", training_set.dtypes)
print("\n Contains NaN/Empty cells : ", training_set.isnull().values.any())
print("\n Total empty cells by column :\n", training_set.isnull().sum(), "\n\n")
""" 
EDA on Training Set

##############################
>>> print("\nFeatures/Columns : \n", training_set.columns)

Features/Columns :
 Index(['Airline', 'Date_of_Journey', 'Source', 'Destination', 'Route',
       'Dep_Time', 'Arrival_Time', 'Duration', 'Total_Stops',
       'Additional_Info', 'Price'],
      dtype='object')
>>> print("\n\nNumber of Features/Columns : ", len(training_set.columns))


Number of Features/Columns :  11
>>> print("\nNumber of Rows : ",len(training_set))

Number of Rows :  10683
>>> print("\n\nData Types :\n", training_set.dtypes)


Data Types :
 Airline            object
Date_of_Journey    object
Source             object
Destination        object
Route              object
Dep_Time           object
Arrival_Time       object
Duration           object
Total_Stops        object
Additional_Info    object
Price               int64
dtype: object
>>> print("\n Contains NaN/Empty cells : ", training_set.isnull().values.any())

 Contains NaN/Empty cells :  True
>>> print("\n Total empty cells by column :\n", training_set.isnull().sum(), "\n\n")

 Total empty cells by column :
 Airline            0
Date_of_Journey    0
Source             0
Destination        0
Route              1
Dep_Time           0
Arrival_Time       0
Duration           0
Total_Stops        1
Additional_Info    0
Price              0
dtype: int64 """

# Test Set
print("#"*30)
print("\nEDA on Test Set\n")
print("#"*30)
print("\nFeatures/Columns : \n",test_set.columns)
print("\n\nNumber of Features/Columns : ",len(test_set.columns))
print("\nNumber of Rows : ",len(test_set))
print("\n\nData Types :\n", test_set.dtypes)
print("\n Contains NaN/Empty cells : ", test_set.isnull().values.any())
print("\n Total empty cells by column :\n", test_set.isnull().sum())
print("Original Length of Training Set : ", len(training_set))
""" 
##############################
>>> print("\nFeatures/Columns : \n",test_set.columns)

Features/Columns :
 Index(['Airline', 'Date_of_Journey', 'Source', 'Destination', 'Route',
       'Dep_Time', 'Arrival_Time', 'Duration', 'Total_Stops',
       'Additional_Info'],
      dtype='object')
>>> print("\n\nNumber of Features/Columns : ",len(test_set.columns))


Number of Features/Columns :  10
>>> print("\nNumber of Rows : ",len(test_set))

Number of Rows :  2671
>>> print("\n\nData Types :\n", test_set.dtypes)


Data Types :
 Airline            object
Date_of_Journey    object
Source             object
Destination        object
Route              object
Dep_Time           object
Arrival_Time       object
Duration           object
Total_Stops        object
Additional_Info    object
dtype: object
>>> print("\n Contains NaN/Empty cells : ", test_set.isnull().values.any())

 Contains NaN/Empty cells :  False
>>> print("\n Total empty cells by column :\n", test_set.isnull().sum())

 Total empty cells by column :
 Airline            0
Date_of_Journey    0
Source             0
Destination        0
Route              0
Dep_Time           0
Arrival_Time       0
Duration           0
Total_Stops        0
Additional_Info    0
dtype: int64 """

# dropping the NaN value.  we chose to drop it as there is only one NaN value. 
training_set = training_set.dropna() 

print("Length of Training Set after dropping NaN: ", len(training_set))
# Length of Training Set after dropping NaN:  10682

# This means that there was only one row having missing values which we have removed

# converting the data type to data-time format and adding new columns
training_set['Journey_Day'] = pd.to_datetime(training_set.Date_of_Journey, format='%d/%m/%Y').dt.day

training_set['Journey_Month'] = pd.to_datetime(training_set.Date_of_Journey, format='%d/%m/%Y').dt.month
training_set.head() # cheking the dataset after transformation
""" 
       Airline Date_of_Journey    Source Destination                  Route  \
0       IndiGo      24/03/2019  Banglore   New Delhi              BLR → DEL
1    Air India       1/05/2019   Kolkata    Banglore  CCU → IXR → BBI → BLR
2  Jet Airways       9/06/2019     Delhi      Cochin  DEL → LKO → BOM → COK
3       IndiGo      12/05/2019   Kolkata    Banglore        CCU → NAG → BLR
4       IndiGo      01/03/2019  Banglore   New Delhi        BLR → NAG → DEL

  Dep_Time  Arrival_Time Duration Total_Stops Additional_Info  Price  \
0    22:20  01:10 22 Mar   2h 50m    non-stop         No info   3897
1    05:50         13:15   7h 25m     2 stops         No info   7662
2    09:25  04:25 10 Jun      19h     2 stops         No info  13882
3    18:05         23:30   5h 25m      1 stop         No info   6218
4    16:50         21:35   4h 45m      1 stop         No info  13302

   Journey_Day  Journey_Month
0           24              3
1            1              5
2            9              6
3           12              5
4            1              3 """

# Test Set
test_set['Journey_Day'] = pd.to_datetime(test_set.Date_of_Journey, format='%d/%m/%Y').dt.day
test_set['Journey_Month'] = pd.to_datetime(test_set.Date_of_Journey, format='%d/%m/%Y').dt.month
test_set.head() # cheking the test dataset after transformation
""" 
             Airline Date_of_Journey    Source Destination            Route  \
0        Jet Airways       6/06/2019     Delhi      Cochin  DEL → BOM → COK
1             IndiGo      12/05/2019   Kolkata    Banglore  CCU → MAA → BLR
2        Jet Airways      21/05/2019     Delhi      Cochin  DEL → BOM → COK
3  Multiple carriers      21/05/2019     Delhi      Cochin  DEL → BOM → COK
4           Air Asia      24/06/2019  Banglore       Delhi        BLR → DEL

  Dep_Time  Arrival_Time Duration Total_Stops              Additional_Info  \
0    17:30  04:25 07 Jun  10h 55m      1 stop                      No info
1    06:20         10:20       4h      1 stop                      No info
2    19:15  19:00 22 May  23h 45m      1 stop  In-flight meal not included
3    08:00         21:00      13h      1 stop                      No info
4    23:55  02:45 25 Jun   2h 50m    non-stop                      No info

   Journey_Day  Journey_Month
0            6              6
1           12              5
2           21              5
3           21              5
4           24              6 """

# Compare the dates and delete the original date feature

training_set.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)
print('training set after transformation')

training_set.head()
""" 
training set after transformation

       Airline    Source Destination                  Route Dep_Time  \
0       IndiGo  Banglore   New Delhi              BLR → DEL    22:20
1    Air India   Kolkata    Banglore  CCU → IXR → BBI → BLR    05:50
2  Jet Airways     Delhi      Cochin  DEL → LKO → BOM → COK    09:25
3       IndiGo   Kolkata    Banglore        CCU → NAG → BLR    18:05
4       IndiGo  Banglore   New Delhi        BLR → NAG → DEL    16:50

   Arrival_Time Duration Total_Stops Additional_Info  Price  Journey_Day  \
0  01:10 22 Mar   2h 50m    non-stop         No info   3897           24
1         13:15   7h 25m     2 stops         No info   7662            1
2  04:25 10 Jun      19h     2 stops         No info  13882            9
3         23:30   5h 25m      1 stop         No info   6218           12
4         21:35   4h 45m      1 stop         No info  13302            1

   Journey_Month
0              3
1              5
2              6
3              5
4              3 """
        
test_set.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)

print('test set after transformation')
test_set.head()
""" 
test set after transformation 

             Airline    Source Destination            Route Dep_Time  \
0        Jet Airways     Delhi      Cochin  DEL → BOM → COK    17:30
1             IndiGo   Kolkata    Banglore  CCU → MAA → BLR    06:20
2        Jet Airways     Delhi      Cochin  DEL → BOM → COK    19:15
3  Multiple carriers     Delhi      Cochin  DEL → BOM → COK    08:00
4           Air Asia  Banglore       Delhi        BLR → DEL    23:55

   Arrival_Time Duration Total_Stops              Additional_Info  \
0  04:25 07 Jun  10h 55m      1 stop                      No info
1         10:20       4h      1 stop                      No info
2  19:00 22 May  23h 45m      1 stop  In-flight meal not included
3         21:00      13h      1 stop                      No info
4  02:45 25 Jun   2h 50m    non-stop                      No info

   Journey_Day  Journey_Month
0            6              6
1           12              5
2           21              5
3           21              5
4           24              6"""

# checking the data types of all the columns
training_set.info() 
""" <class 'pandas.core.frame.DataFrame'>
Int64Index: 10682 entries, 0 to 10682
Data columns (total 12 columns):
Airline            10682 non-null object
Source             10682 non-null object
Destination        10682 non-null object
Route              10682 non-null object
Dep_Time           10682 non-null object
Arrival_Time       10682 non-null object
Duration           10682 non-null object
Total_Stops        10682 non-null object
Additional_Info    10682 non-null object
Price              10682 non-null int64
Journey_Day        10682 non-null int64
Journey_Month      10682 non-null int64
dtypes: int64(3), object(9)
memory usage: 1.1+ MB """

# Our duration column had time written in this format 2h 50m . 
# To help machine learning algorithm derive useful insights, we will convert this 
# text into numeric.
duration = list(training_set['Duration'])

for i in range(len(duration)) :
    if len(duration[i].split()) != 2: 
        if 'h' in duration[i] :
            duration[i] = duration[i].strip() + ' 0m'
        elif 'm' in duration[i] :
            duration[i] = '0h {}'.format(duration[i].strip())

dur_hours = []
dur_minutes = []  

for i in range(len(duration)) :
    dur_hours.append(int(duration[i].split()[0][:-1])) #for examole if duration is 49 mintutes 4 sec then it will reflect like 
    dur_minutes.append(int(duration[i].split()[1][:-1]))#0:49:4 and if 2 hours 10 seconds then it will reflect like 2:0:10
    
training_set['Duration_hours'] = dur_hours
training_set['Duration_minutes'] =dur_minutes

training_set.drop(labels = 'Duration', axis = 1, inplace = True) # dropping the original duration column from training set


# Test Set(applyig same code to convert 'Duration' to 'Duration _Hours' and 'Duration_Minutes')
#2h 50m
durationT = list(test_set['Duration'])

for i in range(len(durationT)) :
    if len(durationT[i].split()[0]) != 2:       
        if 'h' in durationT[i] :
            durationT[i] = durationT[i].strip() + ' 0m'
        elif 'm' in durationT[i] :
            durationT[i] = '0h {}'.format(durationT[i].strip())
            
dur_hours = []
dur_minutes = []  

for i in range(len(durationT)) :
    #print(durationT[i].split())
    dur_hours.append(int(durationT[i].split()[0][:-1]))
    if(len(durationT[i].split())>1):
        dur_minutes.append(int(durationT[i].split()[1][:-1]))
    else:
        dur_minutes.append(int(0))
  
    
test_set['Duration_hours'] = dur_hours
test_set['Duration_minutes'] = dur_minutes

#  dropping the original duration column from training set
test_set.drop(labels = 'Duration', axis = 1, inplace = True) 

# # checking the data types of all the columns once again. 'Duration_hours' and 'Duration_minutes' are integer type columns
training_set.info() 
""" <class 'pandas.core.frame.DataFrame'>
Int64Index: 10682 entries, 0 to 10682
Data columns (total 13 columns):
Airline             10682 non-null object
Source              10682 non-null object
Destination         10682 non-null object
Route               10682 non-null object
Dep_Time            10682 non-null object
Arrival_Time        10682 non-null object
Total_Stops         10682 non-null object
Additional_Info     10682 non-null object
Price               10682 non-null int64
Journey_Day         10682 non-null int64
Journey_Month       10682 non-null int64
Duration_hours      10682 non-null int64
Duration_minutes    10682 non-null int64
dtypes: int64(5), object(8)
memory usage: 1.1+ MB """

#Converting 'Dep_Time' to 'Depart_Time_hour' and 'Depart_time_Minutes'
training_set['Depart_Time_Hour'] = pd.to_datetime(training_set.Dep_Time).dt.hour
training_set['Depart_Time_Minutes'] = pd.to_datetime(training_set.Dep_Time).dt.minute

training_set.drop(labels = 'Dep_Time', axis = 1, inplace = True)

# cheking the training set after transformation
training_set.head() 
""" 
       Airline    Source Destination                  Route  Arrival_Time  \
0       IndiGo  Banglore   New Delhi              BLR → DEL  01:10 22 Mar
1    Air India   Kolkata    Banglore  CCU → IXR → BBI → BLR         13:15
2  Jet Airways     Delhi      Cochin  DEL → LKO → BOM → COK  04:25 10 Jun
3       IndiGo   Kolkata    Banglore        CCU → NAG → BLR         23:30
4       IndiGo  Banglore   New Delhi        BLR → NAG → DEL         21:35

  Total_Stops Additional_Info  Price  Journey_Day  Journey_Month  \
0    non-stop         No info   3897           24              3
1     2 stops         No info   7662            1              5
2     2 stops         No info  13882            9              6
3      1 stop         No info   6218           12              5
4      1 stop         No info  13302            1              3

   Duration_hours  Duration_minutes  Depart_Time_Hour  Depart_Time_Minutes
0               2                50                22                   20
1               7                25                 5                   50
2              19                 0                 9                   25
3               5                25                18                    5
4               4                45                16                   50 """

#Converting 'Arr_Time' to 'Arr_Time_hour' and 'Arr_time_Minutes' and dropping the original column
training_set['Arr_Time_Hour'] = pd.to_datetime(training_set.Arrival_Time).dt.hour
training_set['Arr_Time_Minutes'] = pd.to_datetime(training_set.Arrival_Time).dt.minute

training_set.drop(labels = 'Arrival_Time', axis = 1, inplace = True)

# cheking the training set after transformation
training_set.head() 
""" 
       Airline    Source Destination                  Route Total_Stops  \
0       IndiGo  Banglore   New Delhi              BLR → DEL    non-stop
1    Air India   Kolkata    Banglore  CCU → IXR → BBI → BLR     2 stops
2  Jet Airways     Delhi      Cochin  DEL → LKO → BOM → COK     2 stops
3       IndiGo   Kolkata    Banglore        CCU → NAG → BLR      1 stop
4       IndiGo  Banglore   New Delhi        BLR → NAG → DEL      1 stop

  Additional_Info  Price  Journey_Day  Journey_Month  Duration_hours  \
0         No info   3897           24              3               2
1         No info   7662            1              5               7
2         No info  13882            9              6              19
3         No info   6218           12              5               5
4         No info  13302            1              3               4

   Duration_minutes  Depart_Time_Hour  Depart_Time_Minutes  Arr_Time_Hour  \
0                50                22                   20              1
1                25                 5                   50             13
2                 0                 9                   25              4
3                25                18                    5             23
4                45                16                   50             21

   Arr_Time_Minutes
0                10
1                15
2                25
3                30
4                35 """

# applying the same to test set
test_set['Depart_Time_Hour'] = pd.to_datetime(test_set.Dep_Time).dt.hour
test_set['Depart_Time_Minutes'] = pd.to_datetime(test_set.Dep_Time).dt.minute


test_set.drop(labels = 'Dep_Time', axis = 1, inplace = True)

# cheking the test set after transformation
test_set.head() 
""" 
             Airline    Source Destination            Route  Arrival_Time  \
0        Jet Airways     Delhi      Cochin  DEL → BOM → COK  04:25 07 Jun
1             IndiGo   Kolkata    Banglore  CCU → MAA → BLR         10:20
2        Jet Airways     Delhi      Cochin  DEL → BOM → COK  19:00 22 May
3  Multiple carriers     Delhi      Cochin  DEL → BOM → COK         21:00
4           Air Asia  Banglore       Delhi        BLR → DEL  02:45 25 Jun

  Total_Stops              Additional_Info  Journey_Day  Journey_Month  \
0      1 stop                      No info            6              6
1      1 stop                      No info           12              5
2      1 stop  In-flight meal not included           21              5
3      1 stop                      No info           21              5
4    non-stop                      No info           24              6

   Duration_hours  Duration_minutes  Depart_Time_Hour  Depart_Time_Minutes
0              10                55                17                   30
1               4                 0                 6                   20
2              23                45                19                   15
3              13                 0                 8                    0
4               2                50                23                   55 """

test_set['Arr_Time_Hour'] = pd.to_datetime(test_set.Arrival_Time).dt.hour
test_set['Arr_Time_Minutes'] = pd.to_datetime(test_set.Arrival_Time).dt.minute

test_set.drop(labels = 'Arrival_Time', axis = 1, inplace = True)

test_set.head() # cheking the test set after transformation
""" 
             Airline    Source Destination            Route Total_Stops  \
0        Jet Airways     Delhi      Cochin  DEL → BOM → COK      1 stop
1             IndiGo   Kolkata    Banglore  CCU → MAA → BLR      1 stop
2        Jet Airways     Delhi      Cochin  DEL → BOM → COK      1 stop
3  Multiple carriers     Delhi      Cochin  DEL → BOM → COK      1 stop
4           Air Asia  Banglore       Delhi        BLR → DEL    non-stop

               Additional_Info  Journey_Day  Journey_Month  Duration_hours  \
0                      No info            6              6              10
1                      No info           12              5               4
2  In-flight meal not included           21              5              23
3                      No info           21              5              13
4                      No info           24              6               2

   Duration_minutes  Depart_Time_Hour  Depart_Time_Minutes  Arr_Time_Hour  \
0                55                17                   30              4
1                 0                 6                   20             10
2                45                19                   15             19
3                 0                 8                    0             21
4                50                23                   55              2

   Arr_Time_Minutes
0                25
1                20
2                 0
3                 0
4                45 """

# 6 is the index of "Price" in the Training Set , setting it as the label column
Y_train = training_set.iloc[:,6].values  

# Independent Variables
# selects all columns except "Price"
X_train = training_set.iloc[:,training_set.columns != 'Price'].values 

# Independent Variables for Test Set
X_test = test_set.iloc[:,:].values

X_train
""" array([['IndiGo', 'Banglore', 'New Delhi', ..., 20, 1, 10],
       ['Air India', 'Kolkata', 'Banglore', ..., 50, 13, 15],
       ['Jet Airways', 'Delhi', 'Cochin', ..., 25, 4, 25],
       ...,
       ['Jet Airways', 'Banglore', 'Delhi', ..., 20, 11, 20],
       ['Vistara', 'Banglore', 'New Delhi', ..., 30, 14, 10],
       ['Air India', 'Delhi', 'Cochin', ..., 55, 19, 15]], dtype=object) """

# From the info above it could be observed that many colmns are of object type. So, 
# converting those categorical columns to numerical columns

from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
le2 = LabelEncoder()

# Training Set
X_train[:,0] = le1.fit_transform(X_train[:,0])
X_train
""" array([[3, 'Banglore', 'New Delhi', ..., 20, 1, 10],
       [1, 'Kolkata', 'Banglore', ..., 50, 13, 15],
       [4, 'Delhi', 'Cochin', ..., 25, 4, 25],
       ...,
       [4, 'Banglore', 'Delhi', ..., 20, 11, 20],
       [10, 'Banglore', 'New Delhi', ..., 30, 14, 10],
       [1, 'Delhi', 'Cochin', ..., 55, 19, 15]], dtype=object) """

X_train[:,1] = le1.fit_transform(X_train[:,1])
X_train
""" array([[3, 0, 'New Delhi', ..., 20, 1, 10],
       [1, 3, 'Banglore', ..., 50, 13, 15],
       [4, 2, 'Cochin', ..., 25, 4, 25],
       ...,
       [4, 0, 'Delhi', ..., 20, 11, 20],
       [10, 0, 'New Delhi', ..., 30, 14, 10],
       [1, 2, 'Cochin', ..., 55, 19, 15]], dtype=object) """

X_train[:,2] = le1.fit_transform(X_train[:,2])
X_train
""" array([[3, 0, 5, ..., 20, 1, 10],
       [1, 3, 0, ..., 50, 13, 15],
       [4, 2, 1, ..., 25, 4, 25],
       ...,
       [4, 0, 2, ..., 20, 11, 20],
       [10, 0, 5, ..., 30, 14, 10],
       [1, 2, 1, ..., 55, 19, 15]], dtype=object) """

X_train[:,3] = le1.fit_transform(X_train[:,3])
X_train
""" array([[3, 0, 5, ..., 20, 1, 10],
       [1, 3, 0, ..., 50, 13, 15],
       [4, 2, 1, ..., 25, 4, 25],
       ...,
       [4, 0, 2, ..., 20, 11, 20],
       [10, 0, 5, ..., 30, 14, 10],
       [1, 2, 1, ..., 55, 19, 15]], dtype=object) """

X_train[:,4] = le1.fit_transform(X_train[:,4])
X_train
""" array([[3, 0, 5, ..., 20, 1, 10],
       [1, 3, 0, ..., 50, 13, 15],
       [4, 2, 1, ..., 25, 4, 25],
       ...,
       [4, 0, 2, ..., 20, 11, 20],
       [10, 0, 5, ..., 30, 14, 10],
       [1, 2, 1, ..., 55, 19, 15]], dtype=object) """

X_train[:,5] = le1.fit_transform(X_train[:,5])
X_train
""" array([[3, 0, 5, ..., 20, 1, 10],
       [1, 3, 0, ..., 50, 13, 15],
       [4, 2, 1, ..., 25, 4, 25],
       ...,
       [4, 0, 2, ..., 20, 11, 20],
       [10, 0, 5, ..., 30, 14, 10],
       [1, 2, 1, ..., 55, 19, 15]], dtype=object) """

# Applying similar operations on the Test Set
X_test[:,0] = le2.fit_transform(X_test[:,0])

X_test[:,1] = le2.fit_transform(X_test[:,1])

X_test[:,2] = le2.fit_transform(X_test[:,2])

X_test[:,3] = le2.fit_transform(X_test[:,3])

X_test[:,4] = le2.fit_transform(X_test[:,4])

X_test[:,5] = le2.fit_transform(X_test[:,5])

X_test
""" array([[4, 2, 1, ..., 30, 4, 25],
       [3, 3, 0, ..., 20, 10, 20],
       [4, 2, 1, ..., 15, 19, 0],
       ...,
       [4, 2, 1, ..., 50, 4, 25],
       [1, 2, 1, ..., 0, 19, 15],
       [6, 2, 1, ..., 55, 19, 15]], dtype=object) """

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_train
""" array([[-0.41080484, -1.65835945,  2.41653414, ..., -0.2349499 ,
        -1.80043628, -0.8900139 ],
       [-1.26115217,  0.89001433, -0.97381203, ...,  1.36360731,
        -0.05090913, -0.5870944 ],
       [ 0.01436882,  0.04055641, -0.2957428 , ...,  0.0314763 ,
        -1.3630545 ,  0.0187446 ],
       ...,
       [ 0.01436882, -1.65835945,  0.38232644, ..., -0.2349499 ,
        -0.34249699, -0.2841749 ],
       [ 2.56541078, -1.65835945,  2.41653414, ...,  0.2979025 ,
         0.0948848 , -0.8900139 ],
       [-1.26115217,  0.04055641, -0.2957428 , ...,  1.63003351,
         0.82385444, -0.5870944 ]]) """

X_test = sc_X.transform(X_test)
X_test
""" array([[ 0.01436882,  0.04055641, -0.2957428 , ...,  0.2979025 ,
        -1.3630545 ,  0.0187446 ],
       [-0.41080484,  0.89001433, -0.97381203, ..., -0.2349499 ,
        -0.48829092, -0.2841749 ],
       [ 0.01436882,  0.04055641, -0.2957428 , ..., -0.5013761 ,
         0.82385444, -1.4958529 ],
       ...,
       [ 0.01436882,  0.04055641, -0.2957428 , ...,  1.36360731,
        -1.3630545 ,  0.0187446 ],
       [-1.26115217,  0.04055641, -0.2957428 , ..., -1.30065471,
         0.82385444, -0.5870944 ],
       [ 0.86471614,  0.04055641, -0.2957428 , ...,  1.63003351,
         0.82385444, -0.5870944 ]]) """

# applying similar operation on the Y labels
Y_train = Y_train.reshape((len(Y_train), 1)) 

Y_train = sc_X.fit_transform(Y_train)

Y_train = Y_train.ravel()
Y_train
""" array([-1.12553455, -0.30906781,  1.03978296, ..., -0.40296691,
        0.77218138,  0.57809433])

We have our training and test data sets seperated which can be used to build a machine learning model now. """