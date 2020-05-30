import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

#####################################################################
# In acest fisier am facut niste testari, sa ne obisnuim cu python, sa vedem cum merge KMeans,
# dar lasam si partea asta si il punem pe moodle
#####################################################################

data = pd.read_csv("beer_reviews.csv")

data.drop(["brewery_id", "brewery_name", "review_time", "review_profilename", "beer_style", "beer_name", "beer_beerid"],
          axis=1, inplace=True)

print(data.head(n=10))

mean_abv = data['beer_abv'].mean()
print(mean_abv)

data = pd.DataFrame(data).fillna(mean_abv)

reviews = data.iloc[:, :]
reviews_to_predict = data.iloc[10000:10500, :]

kmeans = KMeans(n_clusters=4, random_state=0)

result = kmeans.fit(reviews)

print(list(data.columns.values))
print(kmeans.cluster_centers_)

reviews_plot = reviews_to_predict
y_kmeans = kmeans.predict(reviews_plot)

# print(kmeans.predict([[3, 4, 3.5, 2, 4, 6.3]]))


# Visualization of 5 clusters

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.scatter3D(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 4], kmeans.cluster_centers_[:, 5],
    s=300,
    c='yellow',
    marker='*', edgecolors='black',
    label='centroids'
)

ax.scatter3D(
    reviews_plot.iloc[y_kmeans == 0, 0], reviews_plot.iloc[y_kmeans == 0, 4], reviews_plot.iloc[y_kmeans == 0, 5],
    c='blue', s=100,
    marker='v', edgecolors='black',
    label='cluster1'
)

ax.scatter3D(
    reviews_plot.iloc[y_kmeans == 1, 0], reviews_plot.iloc[y_kmeans == 1, 4], reviews_plot.iloc[y_kmeans == 1, 5],
    c='green', s=100,
    marker='o', edgecolors='black',
    label='cluster2'
)

ax.scatter3D(
    reviews_plot.iloc[y_kmeans == 2, 0], reviews_plot.iloc[y_kmeans == 2, 4], reviews_plot.iloc[y_kmeans == 2, 5],
    c='red', s=100,
    marker='X', edgecolors='black',
    label='cluster3'
)

ax.scatter3D(
    reviews_plot.iloc[y_kmeans == 3, 0], reviews_plot.iloc[y_kmeans == 3, 4], reviews_plot.iloc[y_kmeans == 3, 5],
    c='violet', s=100,
    marker='s', edgecolors='black',
    label='cluster4'
)

# ax.scatter3D(
#     reviews_plot.iloc[y_kmeans == 4, 0], reviews_plot.iloc[y_kmeans == 4, 4], reviews_plot.iloc[y_kmeans == 4, 5],
#     c='orange', s=100,
#     marker='P', edgecolors='black',
#     label='cluster5'
# )

ax.set_xlabel("overall rating")
ax.set_ylabel("taste rating")
ax.set_zlabel("alcohol level")
plt.legend(scatterpoints=1)
plt.grid()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 5],
            s=300,
            c='yellow',
            marker='*', edgecolors='black',
            label='centroids')

ax2.scatter(reviews_plot.iloc[y_kmeans == 0, 0], reviews_plot.iloc[y_kmeans == 0, 5],
            c='blue', s=100,
            marker='v', edgecolors='black',
            label='cluster1')

ax2.scatter(reviews_plot.iloc[y_kmeans == 1, 0], reviews_plot.iloc[y_kmeans == 1, 5],
            c='green', s=100,
            marker='o', edgecolors='black',
            label='cluster2')

ax2.scatter(reviews_plot.iloc[y_kmeans == 2, 0], reviews_plot.iloc[y_kmeans == 2, 5],
            c='red', s=100,
            marker='X', edgecolors='black',
            label='cluster3'
            )

ax2.scatter(reviews_plot.iloc[y_kmeans == 3, 0], reviews_plot.iloc[y_kmeans == 3, 5],
            c='violet', s=100,
            marker='s', edgecolors='black',
            label='cluster4'
            )

ax2.set_xlabel("overall rating")
ax2.set_ylabel("alcohol level")
plt.legend(scatterpoints=1)
plt.grid()

plt.show()
# print(kmeans.labels_)
