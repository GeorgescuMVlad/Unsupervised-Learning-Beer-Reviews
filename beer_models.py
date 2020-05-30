import pickle

import pandas as pd
from sklearn.cluster import KMeans

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

# print sample from each data set
print("Data model 1 (beer_abv): ")
print(data_model_1.head(5))

print("\nData model 2 (review_overall, beer_abv): ")
print(data_model_2.head(5))

print("\nData model 3 (review_overall, review_aroma, beer_abv): ")
print(data_model_3.head(5))

print("\nData model 4 (review_overall, review_aroma, review_appearance, beer_abv): ")
print(data_model_4.head(5))

print("\nData model 5 (review_overall, review_aroma, review_appearance, review_taste, beer_abv): ")
print(data_model_5.head(5))

print("\nData model  (review_overall, review_aroma, review_appearance, review_palate, review_taste, beer_abv): ")
print(data.head(5))

# train the models with 4 clusters
print("Training model 1...")
kmeans_1 = KMeans(n_clusters=4, random_state=0)
kmeans_1.fit(data_model_1)
pickle.dump(kmeans_1, open("kmeans_1.pkl", "wb"))  # save model 1

print("Training model 2...")
kmeans_2 = KMeans(n_clusters=4, random_state=0)
kmeans_2.fit(data_model_2)
pickle.dump(kmeans_2, open("kmeans_2.pkl", "wb"))  # save model 2

print("Training model 3...")
kmeans_3 = KMeans(n_clusters=4, random_state=0)
kmeans_3.fit(data_model_3)
pickle.dump(kmeans_3, open("kmeans_3.pkl", "wb"))  # save model 3

print("Training model 4...")
kmeans_4 = KMeans(n_clusters=4, random_state=0)
kmeans_4.fit(data_model_4)
pickle.dump(kmeans_4, open("kmeans_4.pkl", "wb"))  # save model 4

print("Training model 5...")
kmeans_5 = KMeans(n_clusters=4, random_state=0)
kmeans_5.fit(data_model_5)
pickle.dump(kmeans_5, open("kmeans_5.pkl", "wb"))  # save model 5

print("Training model final...")
kmeans_final = KMeans(n_clusters=4, random_state=0)
kmeans_final.fit(data)
pickle.dump(kmeans_final, open("kmeans_final.pkl", "wb"))  # save model final

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
