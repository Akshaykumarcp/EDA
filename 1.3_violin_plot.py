import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

iris = pd.read_csv("iris.csv")

# violine plot combines benefit of PDF and boxplot
# by making densor regions of data fatter and sparser ones thinner in violin plot

sns.violinplot(x="species", y="petal_length", data=iris, size=8)
plt.savefig("1.3_violin_plot.png")
plt.show()

