import Utils, SVD_Inference, Recommender, Matrix, User_Similarity, operator, time, math, random, json
import numpy as np
import pandas as pd


config = json.load(open('./config.json', 'r'))
CITY = config['city']
MODEL = config['model']

def select_random_user_liked_item(user_id):
    '''
        Finds a random item which the user has rated above its average
    '''

    user_ratings = Utils.getUserRatingsForCity(user_id)
    user_avg = Utils.getUserData(user_id)['average']
    items_list, weights = [], []

    for item in user_ratings:
        if user_ratings[item] >= user_avg:
            items_list.append(item)
            weights.append(user_ratings[item])
    
    return random.choices(items_list, weights)[0]


def get_full_explanation(model_id, user_id, item_id, original_score):
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
    reviews_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/reviews.csv')

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
    user_item_ratings = Utils.getAllUserRatings(user_id, reviews_df)
    user_friends = Utils.getUserFriends(user_id)
    k = len(user_friends)
    friends_similarity = {}
    for friend in user_friends:
        friend_sim = User_Similarity.AMSD_user_similarity(user_id, friend, user_item_ratings)
        if friend_sim is not None:
            friends_similarity[friend] = friend_sim

    friends_influence = User_Similarity.get_user_based_influence(user_id, friends_similarity, item_id, original_score)
    print(f'friends influence: {friends_influence}')

    # ELITE INFLUENCE
    elite_users = Utils.getTopKEliteUsers(k)
    elites_similarity = {}
    for elite in elite_users:
        elite_sim = User_Similarity.AMSD_user_similarity(user_id, elite, user_item_ratings)
        if elite_sim is not None:
            elites_similarity[elite] = elite_sim

    elites_influence = User_Similarity.get_user_based_influence(user_id, elites_similarity, item_id, original_score)
    print(f'elites influence: {elite_influence}')

    
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
    print(f'top {k} category influencers')
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


def get_recommendation_from_item(model, user_id, item_id):
    '''
        Returns the list of items most recommended, based on a user's preference for an item
    '''

    R_ai = Recommender.get_recommendation(user_id, model)
    
    user_ratings = Utils.getUserRatingsForCity(user_id)
    user_rated_items_ids = user_ratings.keys()
    items_list_except_j = Utils.list_except(user_rated_items_ids, item_id)
    R_j_ai = Recommender.get_recommendation(user_id, model, items_list_except_j)
    
    scores = {}
    for item in R_ai:
        scores[item] = R_ai[item] * (R_ai[item] - R_j_ai[item])

    # prints results
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    i, k = 0, 5
    item_datat = Utils.getItemData(item_id)
    print(f'since you liked {item_data.name}({item_data.categories}), we recommend:')
    while i < k:
        print(sorted_final_ratings[i])
        print(Utils.getItemData(sorted_scores[i][0]))
        i += 1
    
    return sorted_scores


# For Testing Purposes
if __name__ == '__main__':
    user_id = 'GlxJs5r01_yqIgb4CYtiog'

    #get_full_explanation(MODEL, user_id, '102', 3.428578365286457)
    
    item_id = select_random_user_liked_item(user_id)
    get_recommendation_from_item(MODEL, user_id, 'item_id')