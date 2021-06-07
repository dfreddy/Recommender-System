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
    
    selection = random.choices(items_list, weights)[0]

    print(user_avg)
    print(user_ratings[selection])
    print(Utils.getItemData(selection))

    return selection


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
        item_influence[j] = get_influence(user_ratings, items_list_except_j, item_id, model, original_score)
    
    # CATEGORY INFLUENCE
    cat_influence = {}
    for cat in user_rated_categories.keys():
        if len(user_rated_categories[cat]) > 1:
            items_list_except_cat = Utils.list_except(user_rated_items_ids, user_rated_categories[cat])
            cat_influence[cat] = get_influence(user_ratings, items_list_except_cat, item_id, model, original_score)


    # FRIENDS INFLUENCE
    # TODO
    # go to user_ratings_by_item and find all k of user's friends
    # otherwise, "none of your friends have reviewed this item"
    '''
    user_item_ratings = Utils.getAllUserRatings(user_id, reviews_df)
    user_friends = Utils.getUserFriends(user_id)
    k = len(user_friends)
    friends_similarity = {}
    for friend in user_friends:
        friend_sim = User_Similarity.AMSD_user_similarity(user_id, friend, user_item_ratings)
        if friend_sim is not None:
            friends_similarity[friend] = friend_sim

    friends_influence = User_Similarity.get_user_based_influence(user_id, friends_similarity, item_id, original_score)
    print(f'friends influence: {friends_influence}%')
    '''


    # ELITE INFLUENCE
    # TODO
    # go to user_ratings_by_item and find the top-k most elite users for item_id
    # instead of finding a random k (5?) nr of elite users
    elite_users = Utils.getTopKEliteUsers(k)
    elites_similarity = {}
    for elite in elite_users:
        elite_sim = User_Similarity.AMSD_user_similarity(user_id, elite, user_item_ratings)
        if elite_sim is not None:
            elites_similarity[elite] = elite_sim

    elites_influence = User_Similarity.get_user_based_influence(user_id, elites_similarity, item_id, original_score)
    print(f'elites influence: {elite_influence}%')
    
    # SORT item influence dict
    sorted_items_influence = sorted(item_influence.items(), key=operator.itemgetter(1), reverse=True)
    i, k = 0, 5
    print(Utils.getItemData(item_id))
    print(f'top {k} items %influencers')
    while i < k:
        print(f'{sorted_items_influence[i]} {user_ratings[sorted_items_influence[i][0]]} {model.get(sorted_items_influence[i][0],item_id)}')
        print(Utils.getItemData(sorted_items_influence[i][0]))
        i += 1
    
    # SORT cat influence dict
    sorted_cat_influence = sorted(cat_influence.items(), key=operator.itemgetter(1), reverse=True)
    i, k = 0, 10
    print(Utils.getItemData(item_id))
    print(f'top {k} category %influencers')
    while i < k:
        print(f'{sorted_cat_influence[i]}')
        i += 1


def get_influence(user_ratings, items_list, item_id, model, original_score):
    '''
        Minimalist recommender prediction algorithm for just one item
        Returns the % influence in the recommendation

        user_ratings = dict of user rated items and their given rating
        items_list   = list of user rated items' ids
        item_id      = item originally recommended
    '''
    
    predicted_score = 0
    weighted_sum_top, weighted_sum_bottom = 0, 0
    for j in items_list:
        sim_ji = model.get(j,item_id)
        # skip negative similarities
        if sim_ji <= 0:
            continue

        weighted_sum_top += sim_ji * user_ratings[j]
        weighted_sum_bottom += sim_ji

    if weighted_sum_bottom == 0:
                predicted_score = 1
    else:
        predicted_score = weighted_sum_top / weighted_sum_bottom

    return 100*(original_score - predicted_score) / original_score


def get_most_similar_items(model, item_id):
    '''
        Returns the sorted dict of items most useful and similar to the input item
    '''

    sim_model = Recommender.load_model(model)
    items = Utils.getAllItems()
    items_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/businesses.csv')
    
    scores = {}
    for item in items:
        scores[item['id']] = item['rating'] * sim_model.get(item['id'], item_id)

    # prints results
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    i, k = 0, 5
    item_data = Utils.getItemData(item_id, items_df)
    print(item_data)
    print(f'since you liked {item_data["name"]}, we recommend:\n')
    while i < k:
        print(sorted_scores[i])
        print(Utils.getItemData(sorted_scores[i][0], items_df))
        i += 1
    
    return sorted_scores


def getFriendsBasedRecommendation(model, user_id):
    '''
        Step 1: fetch user's friends
        Step 2: fetch items liked by all friends (mean friends ratings for items > 4)
        Step 3: run recommender with a filter on items available for recommendation
        Step 4: recommend top items, "Because your friends like these items, ..."
    '''

    # step 1
    user_friends = Utils.getUserFriends(user_id)

    # step 2
    filename = '../yelp_dataset/resources/'+CITY+'/user_ratings_by_item.json'
    try:
        file = open(filename, encoding='utf8', mode='r')
    except IOError:
        Utils.saveAllRatingsForAllItems()
        file = open(filename, encoding='utf8', mode='r')
    finally:
        json_data = json.load(file)

    items_ids = Utils.getAllItemsIDs()
    friends_items = []
    
    for item in items_ids:
        users = Utils.getUsersThatRatedItem(item, user_filter=user_friends, json_data=json_data)
        if len(users) > 1: # have at least more than 1 friend rate an item to be considered
            avg_rating = Utils.getFilteredAverageItemRating(item, users, json_data=json_data)
            if avg_rating >= 3.5: # anything rounded up to 4 is considered a good rating
                friends_items.append(item)

    # step 3
    recommendation_dict = Recommender.get_recommendation(user_id, MODEL)
    filtered_recommendation_dict = dict(filter(lambda elem: elem[0] in friends_items, recommendation_dict.items()))
    
    # step 4
    sorted_final_ratings = sorted(filtered_recommendation_dict.items(), key=operator.itemgetter(1), reverse=True)
    i, k = 0, 5
    print(f'top {k} items')
    while i < k:
        if Utils.getItemData(sorted_final_ratings[i][0])['rating'] > 3.0:
            print(sorted_final_ratings[i])
            print(Utils.getItemData(sorted_final_ratings[i][0]))
            i += 1
        else:
            i += 1
            k += 1

    return None


# For Testing Purposes
if __name__ == '__main__':
    user_id = 'V4TPbscN8JsFbEFiwOVBKw'
    user_id = 'GlxJs5r01_yqIgb4CYtiog'
    
    #get_full_explanation(MODEL, user_id, '102', 3.428578365286457)
    
    #item_id = select_random_user_liked_item(user_id)
    #get_most_similar_items(MODEL, item_id)
    
    item_id = '2030'
    #get_explanation_from_item(MODEL, user_id, item_id)

    getFriendsBasedRecommendation(MODEL, user_id)