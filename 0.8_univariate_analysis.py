import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

iris = pd.read_csv("iris.csv")

""" from the list of features ('sepal_length', 'sepal_width', 'petal_length', 'petal_width') 
lets say we've to find which one of the feature is good for classifying flower!
then in such scenarios we shall use univariant analysis """

# petal length feature
sns.FacetGrid(iris, hue="species", size=5) \
   .map(sns.distplot, "petal_length") \
   .add_legend();
plt.savefig('0.8_petal_length_histogramOrDensity_plot.png')
plt.show();

# petal length feature
sns.FacetGrid(iris, hue="species", size=5) \
   .map(sns.distplot, "petal_width") \
   .add_legend();
plt.savefig('0.8_petal_width_histogramOrDensity_plot.png')
plt.show();

# sepal length feature
sns.FacetGrid(iris, hue="species", size=5) \
   .map(sns.distplot, "sepal_length") \
   .add_legend();
plt.savefig('0.8_sepal_length_histogramOrDensity_plot.png')
plt.show();

# sepal width feature
sns.FacetGrid(iris, hue="species", size=5) \
   .map(sns.distplot, "sepal_width") \
   .add_legend();
plt.savefig('0.8_sepal_width_histogramOrDensity_plot.png')
plt.show();

# lesser the over lap, better for classifying the flower
# in our case,  petal length feature has lesser over lap and hence petal length feature is sufficient for classiication of flower



