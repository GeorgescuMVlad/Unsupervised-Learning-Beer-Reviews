from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

data = pd.read_csv("beer_reviews.csv")

# data.drop(["brewery_id", "brewery_name", "review_time", "review_profilename", "beer_style", "beer_name", "beer_beerid"],
#           axis=1, inplace=True)

data.drop(["brewery_id", "brewery_name", "review_time", "review_profilename", "beer_name", "beer_beerid"],
          axis=1, inplace=True)

d = dict([e[:: -1] for e in enumerate(data["beer_style"].unique())])
#print(d)
data["beer_style"] = data["beer_style"].map(d)

#data["beer_style"] = data["beer_style"].drop_duplicates()



print(data["beer_style"])
print(data.head(n=10))

mean_abv = data['beer_abv'].mean()
data = pd.DataFrame(data).fillna(mean_abv)
print(list(data.columns.values))

# print(data.iloc[1227:1232, :4])
#print(pd.isna(data).sum())


# print(pd.isna(data[:, 0]).sum())
# print(pd.isna(data[:, 1]).sum())
# print(pd.isna(data[:, 2]).sum())
# print(pd.isna(data[:, 3]).sum())
# print(pd.isna(data[:, 4]).sum())
# print(pd.isna(data[:, 5]).sum())
