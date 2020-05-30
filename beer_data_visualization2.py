import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

data = pd.read_csv("beer_reviews.csv")

data.drop(["brewery_id", "brewery_name", "review_time", "review_profilename", "beer_style", "beer_name", "beer_beerid"],
          axis=1, inplace=True)
#au ramas: review_overall, review_aroma, review_appearance, review_palate, review_taste, beer_abv


abv_data = data["beer_abv"]
review_overall_data = data["review_overall"]
review_aroma_data = data["review_aroma"]
review_appearance_data = data["review_appearance"]
review_palate_data = data["review_palate"]
review_taste_data = data["review_taste"]

fig, axs = plt.subplots(3)
plt.subplots_adjust(hspace=1.3)
axs[0].hist(x=review_overall_data, color='green', alpha=0.7, rwidth=0.85)
axs[0].set_xlabel('Value')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Review Overall')
axs[0].xaxis.set_ticks(np.arange(0, 5.1, 0.5))

axs[1].hist(x=review_aroma_data, color='#0504aa', alpha=0.7, rwidth=0.85)
axs[1].set_xlabel('Value')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Review Aroma')

axs[2].hist(x=review_appearance_data, color='#79ABD8', alpha=0.7, rwidth=0.85)
axs[2].set_xlabel('Value')
axs[2].set_ylabel('Frequency')
axs[2].set_title('Review Appearance')
axs[2].xaxis.set_ticks(np.arange(0, 5.1, 0.5))

plt.figure()
plt.hist(x=abv_data, bins=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], color='orange', alpha=0.7, rwidth=0.85)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Abv')

fig, axs = plt.subplots(2)
plt.subplots_adjust(hspace=.7)
axs[0].hist(x=review_palate_data, color='#DB4328', alpha=0.7, rwidth=0.85)
axs[0].set_xlabel('Value')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Review Palate')

axs[1].hist(x=review_taste_data, color='#05E1A4', alpha=0.7, rwidth=0.85)
axs[1].set_xlabel('Value')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Review Taste')
plt.show()

# n, bins, patches = plt.hist(x=abv_data, bins=[2,3,4,5,6,7,8,9,10,11,12,13,14,15], color='#0504aa',
#                             alpha=0.7, rwidth=0.85)
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Abv')
# plt.show()
