import Utils, json, time, pprint, csv
import pandas as pd
import numpy as np

CITY = 'Toronto'


def AMSD_user_similarity(u1, u2, u1_item_ratings=None):
    '''
        Calculates and returns the similarity between users using the AMSD equation
    '''

    # retrieve relevant data
    reviews_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/reviews.csv')
    
    if u1_item_ratings is None:
        u1_item_ratings = Utils.getAllUserRatings(u1, reviews_df)
    u1_items = u1_item_ratings.keys()
    u2_item_ratings = Utils.getAllUserRatings(u2, reviews_df)
    u2_items = u2_item_ratings.keys()
    mutual_rated_items = set(u1_items).intersection(set(u2_items))
    len_mutual_rated_items = len(mutual_rated_items)

    # calculate amsd
    similarity = None
    if len_mutual_rated_items > 0:
        msd, l = 0, pow(4,2)

        for item_id in mutual_rated_items:
            msd += pow(u1_item_ratings[item_id] - u2_item_ratings[item_id], 2)
        msd = (l - (msd / len_mutual_rated_items)) / l

        if msd > 0.0:
            similarity = msd * (len_mutual_rated_items/len(u1_items)) * ((2*len_mutual_rated_items)/(len(u1_item_ratings)+len(u2_item_ratings)))
    
    return similarity


def get_user_based_influence(user_id, similarities, item_id, original_score):
    '''
        Modifies the item based recommendation engine to be based on user similarity
        Returns the predicted R value for a single user-item pair
    '''

    reviews_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/reviews.csv')
    users_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/users.csv')
    ra = Utils.getUserData(user_id, users_df)['average']

    weighted_sum, weighted_bottom = 0, 0
    for u in similarities:
        user_data = Utils.getUserData(u, users_df)
        # friends may have no reviews in the city, thus not existing in the datset
        if user_data is not None:
            r_ui = Utils.getUserItemRating(u, item_id, reviews_df)
            if r_ui is not None:
                sim_ua = similarities[u]
                ru = user_data['average']
                weighted_sum += sim_ua * (r_ui - ru)
                weighted_bottom += sim_ua
    
    if weighted_sum == 0:
        return None

    predicted_rating = ra + (weighted_sum / weighted_bottom)

    return 100*(original_score - predicted_score) / original_score