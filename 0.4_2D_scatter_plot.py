import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

iris = pd.read_csv("iris.csv")

# 2D scatter plot
# to understand the axis i,e labels and scale

iris.plot(kind='scatter',x='sepal_length',y='sepal_width');
plt.savefig('0.4_2D_scatter_plot.png')
plt.show()
# plot doesn't start from 0,0. Observe the axis scale
# not interactive, so lets add color to distinguish

sns.set_style('whitegrid')
# color based on species
sns.FacetGrid(iris,hue='species', size=4) \
    .map(plt.scatter,"sepal_length","sepal_width") \
        .add_legend();
plt.savefig('0.4_2D_scatter_plot_with_color.png')
plt.show()

"""
Observations:
- using sepal_length and sepal_width, can distinguish between setosa flower and others
- Distinguishing other flowers is quiet tricky because they overlap each other
"""