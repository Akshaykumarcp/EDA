import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

iris = pd.read_csv("iris.csv")

# hack for viz 4-Dim
# pair plot --> pair wise scatter plot

plt.close();
sns.set_style("whitegrid");
sns.pairplot(iris, hue="species", size=3);
plt.savefig('0.6_pair_plot.png')
plt.show()

""" 
Observations

- petal_length and petal_width are the useful features to identify various flower types.
- Setosa can be easily identified (linearly seperable), Virnica and Versicolor have some overlap (almost linearly seperable).
- We can find "lines" and "if-else" conditions to build a simple model to classify the flower types. 

"""

""" Disadvantages:
- used when number of features are high.
- Cannot visualize higher dimensional patterns 3-D and 4-D. 
- Only possible to view 2D patterns. """

# note: the diagnol elements are PDFs for each feature

