import pandas as pd

iris = pd.read_csv("iris.csv")

# how many data points and features are present ?
iris.shape
# (150, 5) --> 150 data points and 5 features

# list column names
iris.columns
""" Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
       'species'],
      dtype='object') """

# list the counts of each features present
iris['species'].value_counts()
""" 
setosa        50
versicolor    50
virginica     50 """

