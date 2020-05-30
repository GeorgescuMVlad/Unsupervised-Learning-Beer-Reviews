import pickle
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def plot_2d(centers, indexes, data_plot_x, data_plot_y, title):
    # plot data
    plt.scatter(centers[:, indexes[0]], centers[:, indexes[1]],
                s=300,
                c='yellow',
                marker='*', edgecolors='black',
                label='centroids')
    plt.scatter(data_plot_x[:10], data_plot_y[:10],
                c='blue', s=100,
                marker='v', edgecolors='black',
                label='cluster1')
    plt.scatter(data_plot_x[10:20], data_plot_y[10:20],
                c='green', s=100,
                marker='o', edgecolors='black',
                label='cluster2'
                )
    plt.scatter(data_plot_x[20:30], data_plot_y[20:30],
                c='red', s=100,
                marker='X', edgecolors='black',
                label='cluster3'
                )
    plt.scatter(data_plot_x[30:40], data_plot_y[30:40],
                c='violet', s=100,
                marker='s', edgecolors='black',
                label='cluster4'
                )
    plt.xlabel("review_overall")
    plt.ylabel("beer_abv")
    plt.legend(scatterpoints=1, loc='lower left')
    plt.title(title)
    plt.show()


def visualize_model1_2d(kmeans_model, data_in):
    # predict data
    y_kmeans = kmeans_model.predict(data_in)

    print(list(data_in.columns.values))
    print(kmeans_model.cluster_centers_)

    # init data for plotting
    data_plot = []
    for i in range(40):
        data_plot.append(0)

    length = data_in.shape[0]
    clusters = [0, 0, 0, 0]

    # find data sample for each cluster to put on plot
    for i in range(length):
        cluster = y_kmeans[i]
        if clusters[cluster] < 10:
            index = cluster * 10 + clusters[cluster] % 10
            abv = data_in.iloc[i, 0]
            if index == 0 or 13 < abv < 20:
                data_plot[index] = data_in.iloc[i, 0]
                clusters[cluster] += 1
            elif index > 0 and abv != data_plot[index - 1]:
                data_plot[index] = data_in.iloc[i, 0]
                clusters[cluster] += 1

    print(data_plot)
    plot_2d(kmeans_model.cluster_centers_, [0, 0], data_plot, data_plot, "Model1")


def visualize_model2_2d(kmeans_model, data_in):
    # predict data
    y_kmeans = kmeans_model.predict(data_in)

    print(list(data_in.columns.values))
    print(kmeans_model.cluster_centers_)

    # init data for plot
    data_plot_x = [0] * 40
    data_plot_y = [0] * 40

    length = data_in.shape[0]
    clusters = [0, 0, 0, 0]

    # find data sample for each cluster to put on plot
    for i in range(length):
        cluster = y_kmeans[i]
        if clusters[cluster] < 10:
            index = cluster * 10 + clusters[cluster] % 10
            abv = data_in.iloc[i, 1]
            rating = data_in.iloc[i, 0]
            if index == 0 or 13 < abv < 20:
                data_plot_y[index] = abv
                data_plot_x[index] = rating
                clusters[cluster] += 1
            elif index > 0 and abv != data_plot_y[index - 1]:
                data_plot_y[index] = abv
                data_plot_x[index] = rating
                clusters[cluster] += 1

    plot_2d(kmeans_model.cluster_centers_, [0, 1], data_plot_x, data_plot_y, "Model 2")


def visualize_model3_4_3d(kmeans_model, data_in, features):
    # predict data
    y_kmeans_all = kmeans_model.predict(data_in)

    print(list(data_in.columns.values))
    print(kmeans_model.cluster_centers_)

    # init data for plot
    data_plot_x = [0] * 60
    data_plot_y = [0] * 60
    data_plot_z = [0] * 60

    length = data_in.shape[0]
    clusters = [0, 0, 0, 0]

    # find data sample for each cluster to put on plot
    for i in range(length):
        cluster = y_kmeans_all[i]
        if clusters[cluster] < 10:
            index = cluster * 10 + clusters[cluster] % 10
            abv = data_in.iloc[i, features[2]]
            rating_1 = data_in.iloc[i, features[1]]
            rating_overall = data_in.iloc[i, features[0]]
            if index == 0 or 13 < abv < 20:
                data_plot_x[index] = rating_overall
                data_plot_y[index] = rating_1
                data_plot_z[index] = abv
                clusters[cluster] += 1
            elif index > 0 and abv != data_plot_z[index - 1]:
                data_plot_x[index] = rating_overall
                data_plot_y[index] = rating_1
                data_plot_z[index] = abv
                clusters[cluster] += 1
    print(data_plot_z)
    plot_2d(kmeans_model.cluster_centers_, [features[0], features[2]], data_plot_x, data_plot_z, "Model 3")

    # plot 3D
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.scatter3D(
        kmeans_model.cluster_centers_[:, features[0]], kmeans_model.cluster_centers_[:, features[1]],
        kmeans_model.cluster_centers_[:, features[2]],
        s=300,
        c='yellow',
        marker='*', edgecolors='black',
        label='centroids'
    )

    ax.scatter3D(
        data_plot_x[:10], data_plot_y[:10], data_plot_z[:10],
        c='blue', s=100,
        marker='v', edgecolors='black',
        label='cluster1'
    )

    ax.scatter3D(
        data_plot_x[10:20], data_plot_y[10:20], data_plot_z[10:20],
        c='green', s=100,
        marker='o', edgecolors='black',
        label='cluster2'
    )

    ax.scatter3D(
        data_plot_x[20:30], data_plot_y[20:30], data_plot_z[20:30],
        c='red', s=100,
        marker='X', edgecolors='black',
        label='cluster3'
    )

    ax.scatter3D(
        data_plot_x[30:40], data_plot_y[30:40], data_plot_z[30:40],
        c='violet', s=100,
        marker='s', edgecolors='black',
        label='cluster4'
    )
    ax.set_xlabel("overall rating")
    ax.set_ylabel("aroma rating")
    ax.set_zlabel("alcohol level")
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()


def visualize_model_5_final(kmeans_model, data_in, features):
    # predict data
    reviews_plot = data_in.iloc[10000:10500, :]
    y_kmeans = kmeans_model.predict(reviews_plot)

    print(list(data_in.columns.values))
    print(kmeans_model.cluster_centers_)

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.scatter3D(
        kmeans_model.cluster_centers_[:, features[0]], kmeans_model.cluster_centers_[:, features[1]],
        kmeans_model.cluster_centers_[:, features[2]],
        s=300,
        c='yellow',
        marker='*', edgecolors='black',
        label='centroids'
    )

    ax.scatter3D(
        reviews_plot.iloc[y_kmeans == 0, features[0]], reviews_plot.iloc[y_kmeans == 0, features[1]],
        reviews_plot.iloc[y_kmeans == 0, features[2]],
        c='blue', s=100,
        marker='v', edgecolors='black',
        label='cluster1'
    )

    ax.scatter3D(
        reviews_plot.iloc[y_kmeans == 1, features[0]], reviews_plot.iloc[y_kmeans == 1, features[1]],
        reviews_plot.iloc[y_kmeans == 1, features[2]],
        c='green', s=100,
        marker='o', edgecolors='black',
        label='cluster2'
    )

    ax.scatter3D(
        reviews_plot.iloc[y_kmeans == 2, features[0]], reviews_plot.iloc[y_kmeans == 2, features[1]],
        reviews_plot.iloc[y_kmeans == 2, features[2]],
        c='red', s=100,
        marker='X', edgecolors='black',
        label='cluster3'
    )

    ax.scatter3D(
        reviews_plot.iloc[y_kmeans == 3, features[0]], reviews_plot.iloc[y_kmeans == 3, features[1]],
        reviews_plot.iloc[y_kmeans == 3, features[2]],
        c='violet', s=100,
        marker='s', edgecolors='black',
        label='cluster4'
    )

    ax.set_xlabel("overall rating")
    ax.set_ylabel("taste rating")
    ax.set_zlabel("alcohol level")
    plt.legend(scatterpoints=1)
    plt.grid()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    ax2.scatter(kmeans_model.cluster_centers_[:, features[0]], kmeans_model.cluster_centers_[:, features[2]],
                s=300,
                c='yellow',
                marker='*', edgecolors='black',
                label='centroids')

    ax2.scatter(reviews_plot.iloc[y_kmeans == 0, features[0]], reviews_plot.iloc[y_kmeans == 0, features[2]],
                c='blue', s=100,
                marker='v', edgecolors='black',
                label='cluster1')

    ax2.scatter(reviews_plot.iloc[y_kmeans == 1, features[0]], reviews_plot.iloc[y_kmeans == 1, features[2]],
                c='green', s=100,
                marker='o', edgecolors='black',
                label='cluster2')

    ax2.scatter(reviews_plot.iloc[y_kmeans == 2, features[0]], reviews_plot.iloc[y_kmeans == 2, features[2]],
                c='red', s=100,
                marker='X', edgecolors='black',
                label='cluster3'
                )

    ax2.scatter(reviews_plot.iloc[y_kmeans == 3, features[0]], reviews_plot.iloc[y_kmeans == 3, features[2]],
                c='violet', s=100,
                marker='s', edgecolors='black',
                label='cluster4'
                )

    ax2.set_xlabel("overall rating")
    ax2.set_ylabel("alcohol level")
    plt.legend(scatterpoints=1)
    plt.grid()

    plt.show()


def visualize_all_centers():
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(
        kmeans_3.cluster_centers_[:, 0], kmeans_3.cluster_centers_[:, 1], kmeans_3.cluster_centers_[:, 2],
        s=300,
        c='yellow',
        marker='*', edgecolors='black',
        label='model 3'
    )
    ax.scatter3D(
        kmeans_4.cluster_centers_[:, 0], kmeans_4.cluster_centers_[:, 1], kmeans_4.cluster_centers_[:, 3],
        s=300,
        c='purple',
        marker='*', edgecolors='black',
        label='model 4'
    )
    ax.scatter3D(
        kmeans_5.cluster_centers_[:, 0], kmeans_5.cluster_centers_[:, 1], kmeans_5.cluster_centers_[:, 4],
        s=300,
        c='green',
        marker='*', edgecolors='black',
        label='model 5'
    )
    ax.scatter3D(
        kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1], kmeans_final.cluster_centers_[:, 5],
        s=300,
        c='blue',
        marker='*', edgecolors='black',
        label='model final'
    )

    ax.set_xlabel("overall rating")
    ax.set_ylabel("aroma rating")
    ax.set_zlabel("alcohol level")
    plt.legend(scatterpoints=1)
    plt.grid()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    # ax2.scatter(kmeans_1.cluster_centers_[:, 0], kmeans_1.cluster_centers_[:, 0],
    #             s=300,
    #             c='red',
    #             marker='*', edgecolors='black',
    #             label='model 1')
    ax2.scatter(kmeans_2.cluster_centers_[:, 0], kmeans_2.cluster_centers_[:, 1],
                s=300,
                c='orange',
                marker='*', edgecolors='black',
                label='model 2')
    ax2.scatter(kmeans_3.cluster_centers_[:, 0], kmeans_3.cluster_centers_[:, 2],
                s=300,
                c='yellow',
                marker='*', edgecolors='black',
                label='model 3')
    ax2.scatter(kmeans_4.cluster_centers_[:, 0], kmeans_4.cluster_centers_[:, 3],
                s=300,
                c='purple',
                marker='*', edgecolors='black',
                label='model 4')
    ax2.scatter(kmeans_5.cluster_centers_[:, 0], kmeans_5.cluster_centers_[:, 4],
                s=300,
                c='green',
                marker='*', edgecolors='black',
                label='model 5')
    ax2.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 5],
                s=300,
                c='blue',
                marker='*', edgecolors='black',
                label='model final')

    ax2.set_xlabel("overall rating")
    ax2.set_ylabel("alcohol level")
    plt.legend(scatterpoints=1)
    plt.grid()

    plt.show()


def print_centers():
    # print cluster centers
    print("\nPrinting cluster centers:")

    print("\nCluster centers for model 1")
    print(list(data_model_1.columns.values))
    print(kmeans_1.cluster_centers_)

    print("\nCluster centers for model 2")
    print(list(data_model_2.columns.values))
    print(kmeans_2.cluster_centers_)

    print("\nCluster centers for model 3")
    print(list(data_model_3.columns.values))
    print(kmeans_3.cluster_centers_)

    print("\nCluster centers for model 4")
    print(list(data_model_4.columns.values))
    print(kmeans_4.cluster_centers_)

    print("\nCluster centers for model 5")
    print(list(data_model_5.columns.values))
    print(kmeans_5.cluster_centers_)

    print("\nCluster centers for model final")
    print(list(data.columns.values))
    print(kmeans_final.cluster_centers_)


# INIT DATA
# load models
print("Load models...")
kmeans_1 = pickle.load(open("kmeans_1.pkl", "rb"))
kmeans_2 = pickle.load(open("kmeans_2.pkl", "rb"))
kmeans_3 = pickle.load(open("kmeans_3.pkl", "rb"))
kmeans_4 = pickle.load(open("kmeans_4.pkl", "rb"))
kmeans_5 = pickle.load(open("kmeans_5.pkl", "rb"))
kmeans_final = pickle.load(open("kmeans_final.pkl", "rb"))

print("Loading data set...")
beer_reviews = pd.read_csv("beer_reviews.csv")
data = beer_reviews.copy()  # create copy of original data set

# drop unnecessary columns
data.drop(["brewery_id", "brewery_name", "review_time", "review_profilename", "beer_style", "beer_name", "beer_beerid"],
          axis=1, inplace=True)

# fill empty cells with mean value
mean_abv = data['beer_abv'].mean()
data = pd.DataFrame(data).fillna(mean_abv)

# data contains data about beer_reviews including all features that are necessary and empty cells are filled

# make copies of data for every different model
data_model_1 = data.copy()  # data with only beer_abv
data_model_2 = data.copy()  # data with beer_abv and review_overall
data_model_3 = data.copy()  # data with beer_abv, review_overall and review_aroma
data_model_4 = data.copy()  # data with beer_abv, review_overall, review_aroma and review_appearance
data_model_5 = data.copy()  # data with beer_abv, review_overall, review_aroma, review_appearance and review_taste

# prepare data for each model by dropping columns that we don't need
print("Dropping unnecessary columns...")
data_model_1.drop(["review_overall", "review_aroma", "review_appearance", "review_palate", "review_taste"], axis=1,
                  inplace=True)
data_model_2.drop(["review_aroma", "review_appearance", "review_palate", "review_taste"], axis=1, inplace=True)
data_model_3.drop(["review_appearance", "review_palate", "review_taste"], axis=1, inplace=True)
data_model_4.drop(["review_palate", "review_taste"], axis=1, inplace=True)
data_model_5.drop(["review_palate"], axis=1, inplace=True)

# visualize_model1_2d(kmeans_1, data_model_1)


op = 1
while op != 0:
    print("1 - Print cluster centers for each model")
    print("2 - Visualize model 1, only abv")
    print("3 - Visualize model 2, overall rating, abv")
    print("4 - Visualize model 3, overall rating, aroma, abv")
    print("5 - Visualize model 4, overall rating, aroma, appearance, abv")
    print("6 - Visualize model 5, overall, aroma, appearance, taste, abv")
    print("7 - Visualize model final, overall, aroma, appearance, palate, taste, abv")
    print("8 - Visualize cluster centers for all")
    print("0 - Exit.")
    op = int(input("Enter your value: "))
    print(op)
    if op == 0:
        break
    if op == 1:
        print_centers()
        continue
    if op == 2:
        visualize_model1_2d(kmeans_1, data_model_1)
        continue
    if op == 3:
        visualize_model2_2d(kmeans_2, data_model_2)
        continue
    if op == 4:
        visualize_model3_4_3d(kmeans_3, data_model_3, [0, 1, 2])
        continue
    if op == 5:
        visualize_model3_4_3d(kmeans_4, data_model_4, [0, 1, 3])
        continue
    if op == 6:
        visualize_model_5_final(kmeans_5, data_model_5, [0, 3, 4])
        continue
    if op == 7:
        visualize_model_5_final(kmeans_final, data, [0, 4, 5])
        continue
    if op == 8:
        visualize_all_centers()
        continue
