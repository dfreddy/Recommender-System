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
    users_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/users.csv')
    ra = Utils.getUserData(user_id, users_df)['average']
    user_ratings = Utils.getUserRatingsForCity(user_id)
    user_liked_items = dict(filter(lambda elem: elem[1] >= ra, user_ratings.items()))
    user_rated_items_ids = user_ratings.keys()
    reviews_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/reviews.csv')

    # ITEM INFLUENCE
    items_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/businesses.csv')
    user_rated_categories = {} # {'category_1': [item_id_1, item_id_2, ...]}
    item_influence = {}
    for j in user_liked_items.keys(): # only calculating the influence of items the user liked
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
    
    '''
    # CATEGORY INFLUENCE
    cat_influence = {}
    for cat in user_rated_categories.keys():
        if len(user_rated_categories[cat]) > 1:
            items_list_except_cat = Utils.list_except(user_rated_items_ids, user_rated_categories[cat])
            cat_influence[cat] = get_influence(user_ratings, items_list_except_cat, item_id, model, original_score)
    '''

    '''
    # FRIENDS INFLUENCE
    user_item_ratings = Utils.getAllUserRatings(user_id, reviews_df)
    user_friends = Utils.getUserFriends(user_id)
    item_reviewers = Utils.getUsersThatRatedItem(item_id)
    user_friends_and_reviewers = list(set(user_friends) & set(item_reviewers)) # intersection between user's friends and users that reviewed the item
    k = len(user_friends_and_reviewers)
    if k < 2:
        print('Not enough friends have reviewed this item')
    else:
        friends_similarity = {}
        for friend in user_friends_and_reviewers:
            friend_sim = User_Similarity.AMSD_user_similarity(user_id, friend, user_item_ratings)
            if friend_sim is not None:
                friends_similarity[friend] = friend_sim

        friends_influence = User_Similarity.get_user_based_influence(user_id, friends_similarity, item_id, original_score)
        print(f'friends influence: {friends_influence}%')


    # ELITE INFLUENCE
    elite_users = Utils.getKMostEliteReviewers(item_id, k)
    elites_similarity = {}
    for elite in elite_users:
        elite_sim = User_Similarity.AMSD_user_similarity(user_id, elite, user_item_ratings)
        if elite_sim is not None:
            elites_similarity[elite] = elite_sim

    elites_influence = User_Similarity.get_user_based_influence(user_id, elites_similarity, item_id, original_score)
    print(f'elites influence: {elite_influence}%')
    '''

    # SORT item influence dict
    sorted_items_influence = sorted(item_influence.items(), key=operator.itemgetter(1), reverse=True)
    i, k = 0, 5
    print(Utils.getItemData(item_id))
    print(f'top {k} items %influencers')
    '''
    while i < k:
        print(f'{sorted_items_influence[i]} {user_ratings[sorted_items_influence[i][0]]} {model.get(sorted_items_influence[i][0],item_id)}')
        print(Utils.getItemData(sorted_items_influence[i][0]))
        i += 1
    '''
    print(sorted_items_influence)

    '''
    # SORT cat influence dict
    sorted_cat_influence = sorted(cat_influence.items(), key=operator.itemgetter(1), reverse=True)
    i, k = 0, 10
    print(Utils.getItemData(item_id))
    print(f'top {k} category %influencers')
    while i < k:
        print(f'{sorted_cat_influence[i]}')
        i += 1
    '''


def get_influence(user_ratings, items_list, item_id, model, original_score):
    '''
        Minimalist recommender prediction algorithm for just one item
        Returns the % influence in the recommendation

        user_ratings = dict of user rated items and their given rating
        items_list   = list of user rated items' ids (useful for filtering out a number of items)
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

        item_id -> city-wide numerical id
    '''

    sim_model = Recommender.load_model(model)
    items = Utils.getAllItems()
    items_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/businesses.csv')
    
    scores = {}
    for item in items:
        scores[item['id']] = sim_model.get(item['id'], item_id)

    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    '''
    # prints results
    i, k = 0, 5
    item_data = Utils.getItemData(item_id, items_df)
    while i < k:
        print(sorted_scores[i])
        print(Utils.getItemData(sorted_scores[i][0], items_df))
        i += 1
    '''

    return sorted_scores


def getItemBasedRecommendation(model, user_id, item_id=None):
    '''
        Returns the user's recommendation based on the items most similar to the chosen item

        Step 1: determine chosen item
        Step 2: fetch items most similar to it
        Step 3: run recommender with a filter on items available for recommendation
        Step 4: recommend top items, "Items similar to this item: ..."
    '''

    # step 1
    if item_id is None:
        # get user liked items
        users_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/users.csv')
        ra = Utils.getUserData(user_id, users_df)['average']
        user_ratings = Utils.getUserRatingsForCity(user_id)
        user_liked_items = dict(filter(lambda elem: elem[1] >= ra, user_ratings.items()))
        user_rated_items_ids = user_ratings.keys()
        index = random.randrange(len(user_rated_items_ids)-1)
        item_id = user_rated_items_ids[index]

    # step 2
    items_list = get_most_similar_items(model, item_id)
    items_list = items_list[0:len(items_list)/4]
    items_list = [i[0] for i in items_list] # selecting the top 25% most similar items

    # step 3
    recommendation_dict = Recommender.get_recommendation(user_id, MODEL)
    filtered_recommendation_dict = dict(filter(lambda elem: elem[0] in items_list, recommendation_dict.items()))
    
    # step 4
    sorted_final_ratings = sorted(filtered_recommendation_dict.items(), key=operator.itemgetter(1), reverse=True)
    i, k = 0, 5
    print(f'top {k} items')
    while i < k:
        if Utils.getItemData(sorted_final_ratings[i][0])['rating'] >= 3.0:
            print(sorted_final_ratings[i])
            print(Utils.getItemData(sorted_final_ratings[i][0]))
            i += 1
        else:
            i += 1
            k += 1

    return sorted_final_ratings


def getFriendsBasedRecommendation(model, user_id):
    '''
        Returns the user's recommendation based on their friends' most liked items

        Step 1: fetch user's friends
        Step 2: fetch items liked by majority of friends
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
    users_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/users.csv')
    
    for item in items_ids:
        users = Utils.getUsersThatRatedItem(item, user_filter=user_friends, json_data=json_data)
        if len(users) > 2: # have at least more than 2 friend rate an item to be considered
            # check which friends liked the item
            friends_liked = 0
            for u in users: 
                ra = Utils.getUserData(u, users_df)['average']
                if ra <= json_data[item][u]:
                    friends_liked += 1
            # add item if majority of friends like the item
            if friends_liked >= len(users)/2:
                friends_items.append(item)

    # step 3
    recommendation_dict = Recommender.get_recommendation(user_id, MODEL)
    filtered_recommendation_dict = dict(filter(lambda elem: elem[0] in friends_items, recommendation_dict.items()))
    
    # step 4
    sorted_final_ratings = sorted(filtered_recommendation_dict.items(), key=operator.itemgetter(1), reverse=True)
    i, k = 0, 5
    print(f'top {k} items')
    while i < k:
        if Utils.getItemData(sorted_final_ratings[i][0])['rating'] >= 3.0:
            print(sorted_final_ratings[i])
            print(Utils.getItemData(sorted_final_ratings[i][0]))
            i += 1
        else:
            i += 1
            k += 1

    return sorted_final_ratings


def getCategoryBasedRecommendation(model, user_id):
    '''
        Returns the user's recommendation based on one of their liked categories

        Step 1: determine chosen category
        Step 2: fetch items from category
        Step 3: run recommender with a filter on items available for recommendation
        Step 4: recommend top items, "Because you like this category: ..."
    '''

    # step 1
    # get user liked items
    users_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/users.csv')
    ra = Utils.getUserData(user_id, users_df)['average']
    user_ratings = Utils.getUserRatingsForCity(user_id)
    user_liked_items = dict(filter(lambda elem: elem[1] >= ra, user_ratings.items()))
    user_rated_items_ids = user_ratings.keys()
    # get list of categories of liked items
    items_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/businesses.csv')
    categories_list = []
    for j in user_rated_items_ids:
        categories_list.append(Utils.getItemData(j, items_df)['categories'].replace(', ', ',').split(','))
    # choose category
    index = random.randrange(len(categories_list)-1)
    category = categories_list[index]

    # step 2
    items_list = Utils.getAllItemsFromCategory(category)

    # step 3
    recommendation_dict = Recommender.get_recommendation(user_id, MODEL)
    filtered_recommendation_dict = dict(filter(lambda elem: elem[0] in items_list, recommendation_dict.items()))
    
    # step 4
    sorted_final_ratings = sorted(filtered_recommendation_dict.items(), key=operator.itemgetter(1), reverse=True)
    i, k = 0, 5
    print(f'top {k} items')
    while i < k:
        if Utils.getItemData(sorted_final_ratings[i][0])['rating'] >= 3.0:
            print(sorted_final_ratings[i])
            print(Utils.getItemData(sorted_final_ratings[i][0]))
            i += 1
        else:
            i += 1
            k += 1

    return sorted_final_ratings


# For Testing Purposes
if __name__ == '__main__':
    user_id = 'GlxJs5r01_yqIgb4CYtiog'
    item_id = '0'
    
    #item_id = select_random_user_liked_item(user_id)
    #get_most_similar_items(MODEL, item_id)
    
    #get_full_explanation(MODEL, user_id, item_id, 4)
    #get_explanation_from_item(MODEL, user_id, item_id)

    #getFriendsBasedRecommendation(MODEL, user_id)