from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("beer_reviews.csv")

data.drop(["brewery_id", "brewery_name", "review_time", "review_profilename", "beer_name", "beer_beerid"],
          axis=1, inplace=True)

d = dict([e[:: -1] for e in enumerate(data["beer_style"].unique())])
data["beer_style"] = data["beer_style"].map(d)

print(data.head(n=10))

print(pd.isna(data).sum())  # check NaN values

mean_abv = data['beer_abv'].mean()  # calculate mean of the abv column

print(mean_abv)

data = pd.DataFrame(data).fillna(mean_abv)
print(pd.isna(data).sum())  # check NaN values

reviews = data.iloc[:, :]

# Elbow method visualization - finding the ideal number of clusters

wcss = []
for i in range(1, 11):
    print(i)
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(reviews)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
