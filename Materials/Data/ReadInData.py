### Summary ###
# This script provides code for reading in the fake/real news dataset, as used in a data science case study
# Since the dataset is large, we do not store it locally, but instead opt to download it directly from kaggle
# This is achieved using kagglehub
# The following chuck of code is not meant to be run by itself, but rather added as a cell to your Jupyter notebook(s), where it saves the data as a dataframe

# The dataset contains 4 columns, 1) An index column, 2) The title of the article, 3) The text of the article, 4) A label for real (0)/fake (1) news.

# In order to read in the data, you may need to install the kagglehub module
# To do this, run the following line in your terminal: 
# pip install kagglehub -q

import kagglehub
import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

dataset_path = kagglehub.dataset_download('saurabhshahane/fake-news-classification')

print("Path to dataset files:", dataset_path)

path_join = os.path.join(dataset_path, 'WELFake_Dataset.csv')
df = pd.read_csv(path_join)

print(df.head())


