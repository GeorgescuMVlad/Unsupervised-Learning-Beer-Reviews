import matplotlib.pyplot as plt
import pandas as pd
import pickle
from random import seed
from random import randint
from sklearn.cluster import KMeans


def isInList(lista, value, pos_id):
    if not lista:
        return False
    for k in range(len(lista)):
        if lista[k][pos_id] == value:
            return True
    return False


print("Loading data set...")
beer_reviews = pd.read_csv("beer_reviews.csv")

data = beer_reviews.copy()  # create copy of original data set
data.drop(["brewery_id", "brewery_name", "review_time", "review_profilename", "beer_style", "beer_name", "beer_beerid"],
          axis=1, inplace=True)

beer_reviews.drop(["brewery_id", "review_time", "review_overall",
                   "review_aroma", "review_appearance", "review_profilename", "review_palate", "review_taste"], axis=1,
                  inplace=True)

# fill empty cells with mean value
mean_abv = data['beer_abv'].mean()
data = pd.DataFrame(data).fillna(mean_abv)
beer_reviews = pd.DataFrame(beer_reviews).fillna(mean_abv)

# load model
print("Loading model...")
kmeans_final = pickle.load(open("kmeans_final.pkl", "rb"))
y_kmeans = kmeans_final.predict(data)

# load user data
data_user = pd.read_csv("input_ex.csv")
print(data_user.head(n=5))

# predict user data
y_kmeans_user = kmeans_final.predict(data_user)
print(y_kmeans_user)

# get the best review and its index
reviews_list = data_user['review_overall'].tolist()
max_rating = max(reviews_list)
max_rating_index = reviews_list.index(max(reviews_list))
max_rating_cluster = y_kmeans_user[max_rating_index]

# count data elements in each cluster
clusters = [0, 0, 0, 0]

for i in range(len(y_kmeans_user)):
    clusters[y_kmeans_user[i]] += 1

clusters_descending = sorted(range(len(clusters)), reverse=True, key=lambda k: clusters[k])

# collect recommendations
recommendations = []
best_match_counter = 4  # 4 beers that are in the same cluster as the best rated beer
most_cluster_counter = 3  # 3 beers from the cluster in which are the most beers
second_cluster_counter = 2  # 2 beers from the cluster in which are the second most beers
third_cluster_counter = 1  # 1 beer from the cluster in which are the third most beers
finished = False
length = data.shape[0]

index = randint(0, 1500000)  # start from random position in the data set
print(index)
print("cluster descending ", clusters_descending[1])

# find the recommended beers 
while not finished:
    if index == length - 1:  # if end of data set reached
        index = -1

    index += 1

    if best_match_counter == 0 and most_cluster_counter == 0 and second_cluster_counter == 0 and third_cluster_counter == 0:
        finished = True
        continue

    current_cluster = y_kmeans[index]
    row = beer_reviews.iloc[index]

    if current_cluster == clusters_descending[1]:
        if second_cluster_counter > 0:
            if not isInList(recommendations, row[4], 4):
                recommendations.append(row)
                second_cluster_counter -= 1

    if current_cluster == max_rating_cluster:  # best rating
        if best_match_counter > 0:
            if not isInList(recommendations, row[4], 4):  # check if id with corresponding beer is already chosen
                recommendations.append(row)
                best_match_counter -= 1

    if current_cluster == clusters_descending[0]:  # most elements in cluster
        if most_cluster_counter > 0:
            if not isInList(recommendations, row[4], 4):
                recommendations.append(row)
                most_cluster_counter -= 1

    if current_cluster == clusters_descending[2]:  # third most elements
        if third_cluster_counter > 0:
            if not isInList(recommendations, row[4], 4):
                recommendations.append(row)
                third_cluster_counter -= 1

print("You may like one of the following beers: ")
for i in range(len(recommendations)):
    print(recommendations[i][0], ': ', recommendations[i][2], " - ", recommendations[i][1], "  ",
          "%.2f" % recommendations[i][3], "% abv.")
