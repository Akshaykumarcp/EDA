import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

iris = pd.read_csv("iris.csv")

# previosly we've seen 1Dim PDF/histogram.
# lets see 2Dim density plot or 3Dim or nDim etc

iris_setosa = iris.loc[iris["species"] == "setosa"];

#2D Density plot, contors-plot
sns.jointplot(x="petal_length", y="petal_width", data=iris_setosa, kind="kde");
plt.savefig('1.4_multivariate_jointplot.png')
plt.show();