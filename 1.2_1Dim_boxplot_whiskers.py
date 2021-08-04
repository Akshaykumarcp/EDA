# from our previous plots, we can't know what percentile of data exists
# using CDF we get the sense of it but not clearly 
# so we use boxplot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

iris = pd.read_csv("iris.csv")

sns.boxplot(x='species',y='petal_length', data=iris)
plt.savefig("1.2_boxplot.png")
plt.show()