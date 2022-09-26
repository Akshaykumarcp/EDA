# -*- coding: utf-8 -*-
"""
Created on Fri May  1 06:47:11 2020

@author: Akshay kumar C P
"""

'''
Agenda : LOAN EDA

1. find pattern which indicates the person is likly to default - AIM - so that in future we shall take actions
2. loan approval based on applicant profile
3. application is likely to repay the loan
4. when applicant is nt able to repay the loan he/she will be loss to the company
'''

'''
Going ahead :
    1. data understanding
    2. data cleaning
    3. data analysis
    4. recommendations
'''

# target variable - load status

# import lib's
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

# read datasets
data = pd.read_csv("loan/loan.csv")

#list all columns
dataColumns = data.columns

# info about dataset
data.info()

'''
Data cleaning
'''

# find null values
data.isnull().sum()

# lets look at null values in %
round(data.isnull().sum()/len(data.index),2)* 100

# anything more than 90% drop the columns
missing_columns = data.columns[100 * (data.isnull().sum()/len(data.index)) > 90]

data = data.drop(missing_columns,axis=1)

# lets look at null values in %
round(data.isnull().sum()/len(data.index),2)* 100

# there are 2 columns having 32 and 64 % of missing values

# Let's see them

data.loc[:,['desc','mths_since_last_delinq']].head()

# the desc column contains text while applying load. NLP can be applied in future for +ve and -ve text analytics.

# so let's drop them too

# dropping the 2 columns
data = data.drop(['desc','mths_since_last_delinq'], axis=1)

# lets look at null values in %
round(data.isnull().sum()/len(data.index),2)* 100

# let's ignore less % of null values as it is EDA

# let's see null values in rows, if >=5 let's remove the rows
data.isnull().sum(axis=1)

# check whether some rows have more than 5 missing values

len(data[data.isnull().sum(axis=1) > 5].index)

# let's look whether all column are in right format

data.info()

# int_rate column is in object, we need in integer

data['int_rate'] = data['int_rate'].apply(lambda x: pd.to_numeric(x.split("%")[0]))

data.info()

# lets extract numerical values from emp_length

data['emp_length']

# let's drop missing values

data = data[~data['emp_length'].isnull()]

# using re to extract numerical 
import re
data['emp_length'] = data['emp_length'].apply(lambda x: re.findall('\d+',str(x))[0])

# convert from object type to int
data['emp_length'] = data['emp_length'].apply(lambda x: pd.to_numeric(x))

data.info()

'''
data analysis
'''
'''
3 kind of variable

1. applicants - age, occupation etc
2. loan - amount given , interest rate, purpose etc
3. behaviour variable - after load is approved. we'l get rid'

'''

behaviour_var =  [
  "delinq_2yrs",
  "earliest_cr_line",
  "inq_last_6mths", 
  "open_acc",
  "pub_rec",
  "revol_bal",
  "revol_util",
  "total_acc",
  "out_prncp",
  "out_prncp_inv",
  "total_pymnt",
  "total_pymnt_inv",
  "total_rec_prncp",
  "total_rec_int",
  "total_rec_late_fee",
  "recoveries",
  "collection_recovery_fee",
  "last_pymnt_d",
  "last_pymnt_amnt",
  "last_credit_pull_d",
  "application_type"]

# let's now remove the behaviour variables from analysis
data = data.drop(behaviour_var, axis=1)

# also, we will not be able to use the variables zip code, address, state etc.
# the variable 'title' is derived from the variable 'purpose'
# thus let get rid of all these variables as well

data = data.drop(['title', 'url', 'zip_code', 'addr_state'], axis=1)

# Next, let's have a look at the target variable - loan_status. We need to relabel the values to a binary form - 0 or 1, 1 indicating that the person has defaulted and 0 otherwise.

data['loan_status'] = data['loan_status'].astype('category')
data['loan_status'].value_counts()

# You can see that fully paid comprises most of the loans. The ones marked 'current' are neither fully paid not defaulted, so let's get rid of the current loans. Also, let's tag the other two values as 0 or 1.

# filtering only fully paid or charged-off
data = data[data['loan_status'] != 'Current']
data['loan_status'] = data['loan_status'].apply(lambda x: 0 if x=='Fully Paid' else 1)

# converting loan_status to integer type
data['loan_status'] = data['loan_status'].apply(lambda x: pd.to_numeric(x))

# summarising the values
data['loan_status'].value_counts()