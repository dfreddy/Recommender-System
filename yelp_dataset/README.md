# YELP Datasets

This sub-repository is responsible for preparing the original datasets for the recommender system.

## Steps

### Content-Based Sampling

The 3 datasets are first trimmed, from 2M user entries to 10k. Each user's reviews are grabbed from the original reviews dataset into a new json. Finally, the businesses related to those reviews are selected from the original businesses dataset.

### Collaborative Filtering Sampling

The businesses (read: items) dataset is first trimmed. Only the 3518 businesses from the city of Mississauga remain.

Secondly, the reviews (read: ratings) dataset is trimmed, by removing reviews that aren't of the selected businesses. The reviews are now only 67929, instead of the initial 8M.

Lastly, the users dataset is trimmed down to 25916, by removing the users who didn't write the selected reviews.

The final datasets are:

- ratings ~ 68k
- items ~ 3.5k
- users ~ 26k

### Dimentionality Reduction

In order to better manage space and iteration speed, only the essential attributes of the datasets are kept.
(eg. in the businesses dataset, longitude is removed while categories are kept).

## Links

Original public datasets available @ [yelp.com/dataset](https://www.yelp.com/dataset)
