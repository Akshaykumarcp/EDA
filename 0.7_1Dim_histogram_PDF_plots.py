import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

iris = pd.read_csv("iris.csv")

iris_setosa = iris.loc[iris["species"] == "setosa"];
iris_virginica = iris.loc[iris["species"] == "virginica"];
iris_versicolor = iris.loc[iris["species"] == "versicolor"];

# 1Dim scatter plot of petal-length
plt.plot(iris_setosa["petal_length"], np.zeros_like(iris_setosa['petal_length']), 'o')
plt.plot(iris_versicolor["petal_length"], np.zeros_like(iris_versicolor['petal_length']), 'o')
plt.plot(iris_virginica["petal_length"], np.zeros_like(iris_virginica['petal_length']), 'o')
plt.savefig('0.7_1Dim_scatter_plot.png')
plt.show()

# in above plot, difficult to make sense as points because of the overlap present

# so lets use another way of plot below:

sns.FacetGrid(iris, hue="species", size=5) \
   .map(sns.distplot, "petal_length") \
   .add_legend();
plt.savefig('0.7_1Dim_histogramOrDensity_plot.png')
plt.show();

# the plot is also known as probability density function (PDF)
# using PDF values and if else condition, we shall create a simple model to classify flower

# disadvantage of PDF is it cannot say what % of versicolor points have a petal_length of less than 10 ?




