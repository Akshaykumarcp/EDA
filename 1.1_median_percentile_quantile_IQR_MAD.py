import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

iris = pd.read_csv("iris.csv")

iris_setosa = iris.loc[iris["species"] == "setosa"];
iris_virginica = iris.loc[iris["species"] == "virginica"];
iris_versicolor = iris.loc[iris["species"] == "versicolor"];

# mean, variance and std-dev gets corrupted by outliers
# so median comes into picture

#Median (similar to mean but doesn't corrupted from outlier)
np.median(iris_setosa["petal_length"])
# 1.5

#Median with an outlier
np.median(np.append(iris_setosa["petal_length"],50))
# 1.5

np.median(iris_virginica["petal_length"])
# 5.55

np.median(iris_versicolor["petal_length"])
# 4.35

# how to find median ?
# sort in ascending order and pick middle value
# when order is even, take middle two values and compute mean

# why median is not corrupted by outliers ?
# when sorted, outlier lye at end 

# what happens when half of the values are outlier ?
# when more than 50% of the values are outliers, only then median gets corrupted

# Percentiles & Quantiles --> 25th, 50th, 75th, 100th percentile are called quantiles
# 1st quantile is 25th percentile
# 3rd quantile is 75th percentile
np.percentile(iris_setosa["petal_length"],np.arange(0, 100, 25))
# array([1.   , 1.4  , 1.5  , 1.575])

np.percentile(iris_virginica["petal_length"],np.arange(0, 100, 25))
# array([4.5  , 5.1  , 5.55 , 5.875])

np.percentile(iris_versicolor["petal_length"], np.arange(0, 100, 25))
# array([3.  , 4.  , 4.35, 4.6 ])

np.percentile(iris_setosa["petal_length"],90)
# 1.7

np.percentile(iris_virginica["petal_length"],90)
# 6.3100000000000005

np.percentile(iris_versicolor["petal_length"], 90)
# 4.8

# Ex: why 99 percentile is usefull ?
# 99th percentile = 5.6 days i,e 99 % of customer got delivery of package in 5.6 days --> good

# mean absolute deviation --> how far away are the points from central tendency median. equivalent to std-dev
from statsmodels import robust
robust.mad(iris_setosa["petal_length"])
# 0.14826022185056031

robust.mad(iris_virginica["petal_length"])
# 0.6671709983275211

robust.mad(iris_versicolor["petal_length"])
# 0.5189107764769602

# Inter-quartile range
# 75th percentile - 25th percentile i,e 50 % of values lyes



