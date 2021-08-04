import pandas as pd

iris = pd.read_csv("iris.csv")

# 3D scatter plot

# must open: https://plotly.com/python/3d-scatter-plots/

# hard to plot on VS CODE IDE or jupyter notebook
# we live in 3D space so can viz in 3D. 
# Humans Cannot viz 4D, 5D or nD

# iris dataset has 4Dim so we cannot viz
# instead we use maths and plotting hacks to understand how to see 4D
# 
# Next up, we use pair plot for viz 4D 