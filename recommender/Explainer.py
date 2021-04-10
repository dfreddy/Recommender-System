import Utils, SVD_Inference, Recommender, Matrix, operator, time, math
import numpy as np
import pandas as pd


CITY = 'Toronto'
MODEL = '20210406155019'

def get_explanation(user_id, item_id, model_id, original_score):
    '''
        Returns the biggest influence in the recommendation (item, category, user's friends or elite users)
    '''

    model = Recommender.load_model(model_id)
    user_rated_items_ids = Utils.getUserRatingsForCity(user_id, CITY).keys()
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
        item_influence[j] = get_influence(items_list_except_j, item_id, model, ra, original_score)

    # get category influences
    cat_influence = {}
    for cat in user_rated_categories.keys():
        if len(user_rated_categories[cat]) > 1:
            items_list_except_cat = Utils.list_except(user_rated_items_ids, user_rated_categories[cat])
            cat_influence[cat] = get_influence(items_list_except_cat, item_id, model, ra, original_score)


def get_influence(items_list, item_id, model, ra, original_score):
    '''
        Minimalist user recommender prediction for just one item
    '''
    
    return


# For Testing Purposes
if __name__ == '__main__':
    get_explanation('GlxJs5r01_yqIgb4CYtiog', '304', MODEL, 4)