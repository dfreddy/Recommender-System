import Utils, json, time, pprint, csv
import pandas as pd
import numpy as np

CITY = 'Toronto'


def AMSD_user_similarity(u1, u2):
    '''
        Calculates and returns the similarity between users using the AMSD equation
    '''

    # retrieve relevant data
    reviews_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/users_all_reviews.csv')
    
    u1_item_ratings = Utils.getAllUserRatings(u1, reviews_df=reviews_df)
    u2_item_ratings = Utils.getAllUserRatings(u2, reviews_df=reviews_df)
    u1_items = u1_item_ratings.keys()
    u2_items = u2_item_ratings.keys()
    mutual_rated_items = set(item_a_vector).intersection(set(item_b_vector))
    len_mutual_rated_items = len(mutual_rated_items)

    # calculate amsd
    similarity = 0.000001
    if len_mutual_rated_items > 0:
        msd, l = 0, pow(4,2)

        for item_id in mutual_rated_items:
            msd += pow(u1_item_ratings[item_id] - u2_item_ratings[item_id], 2)
        msd = (l - (msd / len_mutual_rated_items)) / l

        if msd > 0.0:
            similarity msd * (len_mutual_rated_items/len(u1_items)) * ((2*len_mutual_rated_items)/(len(u1_item_ratings)+len(u2_item_ratings)))
    
    return similarity


def get_user_based_recommendation(user_id, user_list, similarities, item_id):
    '''
        Modifies the item based recommendation engine to be based on user similarity
        Returns the predicted R value for a single user-item pair
    '''