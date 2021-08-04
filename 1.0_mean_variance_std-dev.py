import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

iris = pd.read_csv("iris.csv")

iris_setosa = iris.loc[iris["species"] == "setosa"];
iris_virginica = iris.loc[iris["species"] == "virginica"];
iris_versicolor = iris.loc[iris["species"] == "versicolor"];

# mean (central tendency) values of petal_length
np.mean(iris_setosa["petal_length"])
np.mean(iris_virginica["petal_length"])
np.mean(iris_versicolor["petal_length"])
""" 
1.464 # setosa have small petal length
5.5520000000000005
4.26 """

# what is the petal length is very high ?
#Mean with an outlier.
np.mean(np.append(iris_setosa["petal_length"],50))
# 2.4156862745098038
# actual mean is 1.4, due to an outlier value,mean now is 2.4
# it is the problem

# spread --> measure of how widely spread distributions
# when we don't have plots, only by looking at numeric value shall say the spread

# spread in simply english can be thought as a variance

# Variance --> squared distance of all points from mean

# Std-deviation --> square root of variance
# says, what is the average deviation of points from mean
# if SD is small, then spread is small
np.std(iris_setosa["petal_length"])
np.std(iris_virginica["petal_length"])
np.std(iris_versicolor["petal_length"])
""" 0.17176728442867112
0.546347874526844
0.4651881339845203 """

# towards left and right side is the std-dev value lye

