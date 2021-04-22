import Utils, SVD_Inference, Recommender, Matrix, User_Similarity, operator, time, math
import numpy as np
import pandas as pd


CITY = 'Toronto'
MODEL = '20210420185504'

def get_explanation(model_id, user_id, item_id, original_score):
    '''
        Returns the biggest influence in the recommendation (item, category, user's friends or elite users)
    '''

    # load similarity model
    model = Recommender.load_model(model_id)
    # load user data
    user_ratings = Utils.getUserRatingsForCity(user_id)
    user_rated_items_ids = user_ratings.keys()
    nr_user_rated_items = len(user_rated_items_ids)
    users_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/users.csv')
    ra = Utils.getUserData(user_id, users_df)['average']

    # ITEM INFLUENCE
    items_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/businesses.csv')
    user_rated_categories = {} # {'category_1': [item_id_1, item_id_2, ...]}
    item_influence = {}
    for j in user_rated_items_ids:
        # get item categories for later
        item_categories = Utils.getItemData(j, items_df)['categories'].replace(', ', ',').split(',')
        
        # make list of items belonging to this repeated category
        for cat in item_categories:
            if cat in user_rated_categories.keys():
                user_rated_categories[cat] += [j]
            else:
                user_rated_categories[cat] = [j]

        # calculate item influence
        items_list_except_j = Utils.list_except(user_rated_items_ids, j)
        item_influence[j] = get_influence(user_ratings, items_list_except_j, item_id, model, ra, items_df, original_score)

    # CATEGORY INFLUENCE
    cat_influence = {}
    for cat in user_rated_categories.keys():
        if len(user_rated_categories[cat]) > 1:
            items_list_except_cat = Utils.list_except(user_rated_items_ids, user_rated_categories[cat])
            cat_influence[cat] = get_influence(user_ratings, items_list_except_cat, item_id, model, ra, items_df, original_score)

    # FRIENDS INFLUENCE
    # TODO
    # retrieve all user k friends
    # calculate similarity between them
    # calculate Ru' with only the friends
    user_friends = Utils.getUserFriends(user_id)
    k = len(user_friends)
    friends_similarity = {}
    for friend in user_friends:
        friends_similarity[friend] = User_Similarity.AMSD_user_similarity(user_id, friend)
    friends_influence = User_Similarity.get_user_based_recommendation(user_id, friends_similarity, item_id)

    # ELITE INFLUENCE
    # TODO
    # retrieve top k elite users
    # calculate similarities
    # calculate Ru' with only the elite users
    # compare values
    elite_users = Utils.getTopKEliteUsers(k)
    elites_similarity = {}
    for elite in elite_users:
        elites_similarity[elite] = User_Similarity.AMSD_user_similarity(user_id, elite)
    elites_influence = User_Similarity.get_user_based_recommendation(user_id, elites_similarity, item_id)


    # SORT item influence dict
    sorted_items_influence = sorted(item_influence.items(), key=operator.itemgetter(1), reverse=True)
    i, k = 0, 5
    print(Utils.getItemData(item_id))
    print(f'top {k} items influencers')
    while i < k:
        print(f'{sorted_items_influence[i]} {user_ratings[sorted_items_influence[i][0]]} {model.get(sorted_items_influence[i][0],item_id)}')
        print(Utils.getItemData(sorted_items_influence[i][0]))
        i += 1
        
    # SORT cat influence dict
    sorted_cat_influence = sorted(cat_influence.items(), key=operator.itemgetter(1), reverse=True)
    i, k = 0, 10
    print(Utils.getItemData(item_id))
    print(f'top {k} cat influencers')
    while i < k:
        print(f'{sorted_cat_influence[i]}')
        i += 1


def get_influence(user_ratings, items_list, item_id, model, ra, items_df, original_score):
    '''
        Minimalist user recommender prediction algorithm for just one item
        Returns the % influence in the recommendation

        user_ratings = dict of user rated items and their given rating
        items_list   = list of user rated items' ids
        item_id      = item originally recommended
    '''
    
    predicted_score = 0
    weighted_sum, weighted_bottom = 0, 0
    for j in items_list:
        sim_ji = model.get(j,item_id)
        if sim_ji <= 0:
            sim_ji = 0.000001

        r_aj = user_ratings[j]
        rj = Utils.getItemData(j, items_df)['rating']
        weighted_sum += sim_ji * (r_aj - rj)
        weighted_bottom += sim_ji

    predicted_score = ra + (weighted_sum/weighted_bottom)

    return 100*(original_score - predicted_score) / original_score


# For Testing Purposes
if __name__ == '__main__':
    get_explanation(MODEL, 'GlxJs5r01_yqIgb4CYtiog', '102', 3.428578365286457)