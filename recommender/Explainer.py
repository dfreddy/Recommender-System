import Utils, SVD_Inference, Recommender, Matrix, operator, time, math
import numpy as np
import pandas as pd


CITY = 'Toronto'
MODEL = '20210406155019'

def get_explanation(model_id, user_id, item_id, original_score):
    '''
        Returns the biggest influence in the recommendation (item, category, user's friends or elite users)
    '''

    # load similarity model
    model = Recommender.load_model(model_id)
    # load user data
    user_ratings = Utils.getUserRatingsForCity(user_id, CITY)
    user_rated_items_ids = user_ratings.keys()
    nr_user_rated_items = len(user_rated_items_ids)
    users_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/users.csv')
    ra = Utils.getUserData(user_id, users_df)['average']

    # get rated item influences
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

    '''
    # get category influences
    cat_influence = {}
    for cat in user_rated_categories.keys():
        if len(user_rated_categories[cat]) > 1:
            items_list_except_cat = Utils.list_except(user_rated_items_ids, user_rated_categories[cat])
            cat_influence[cat] = get_influence(user_ratings, items_list_except_cat, item_id, model, ra, items_df, original_score)
    '''


    # sort dict
    sorted_items_influence = sorted(item_influence.items(), key=operator.itemgetter(1), reverse=True)
    i, k = 0, 5
    print(Utils.getItemData(item_id))
    print(f'top {k} items influencers')
    while i < k:
        print(f'{sorted_items_influence[i]} {user_ratings[sorted_items_influence[i][0]]}')
        print(Utils.getItemData(sorted_items_influence[i][0]))
        i += 1


def get_influence(user_ratings, items_list, item_id, model, ra, items_df, original_score):
    '''
        Minimalist user recommender prediction for just one item

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

        # TODO
        # give more relevance to the similarity values
        # maybe sim = sim*10 ?
        # since the dataset is so sparse, the similarity values tend to be very low
        # maybe restructure the similarity calc to better respond to this sparsity
        r_aj = user_ratings[j]
        rj = Utils.getItemData(j, items_df)['rating']
        weighted_sum += sim_ji * (r_aj - rj)
        weighted_bottom += sim_ji

    predicted_score = ra + (weighted_sum/weighted_bottom)

    return original_score - predicted_score


# For Testing Purposes
if __name__ == '__main__':
    get_explanation(MODEL, 'GlxJs5r01_yqIgb4CYtiog', '1410', 3.3910595250478908)