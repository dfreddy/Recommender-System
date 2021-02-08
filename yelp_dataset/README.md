# YELP Datasets

This sub-repository is responsible for preparing the original datasets for the recommender system.

## Steps

The 3 datasets are first trimmed, from 2M user entries to 10k. Each user's reviews are grabbed from the original reviews dataset into a new json. Finally, the businesses related to those reviews are selected from the original businesses dataset.

Lastly, only the main features of each dataset entry are kept for future uses (eg. businesses' longitude is excluded, but categories are kept)


## Links

Original public datasets available @ yelp.com/dataset