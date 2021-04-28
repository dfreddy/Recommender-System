import Utils, SVD_Inference, Matrix, operator, time, math, json
import numpy as np
import pandas as pd
from collections import Counter


config = json.load(open('config.json', 'r'))
CITY = config['city']
MODEL = config['model']

def train_model():
    item_similarity_matrix = SVD_Inference.get_similarity_matrix()
    item_similarity_matrix.save()


def load_model(model_id):
    model = Matrix.PredictionSVD()
    model.load(model_id)

    return model


def print_test_loading(new_percentage, start, errors, deviations, sort_devs):
    end = time.perf_counter()
    t = (end-start)/60
    rmse = format(np.sqrt(np.mean(errors)), '.3f')
    md = format(np.mean(deviations), '.1f')
    std = format(np.std(deviations), '.1f')
    print("{}\t\t{}\t\t{}\t{}\t{}".format(str(new_percentage) + '%', str(format(t, '.1f')) + 'm', rmse, md, np.mean(sort_devs)))


def get_sorted_dict_indexes(unsorted_dict):
    sorted_dict = sorted(unsorted_dict.items(), key=operator.itemgetter(1), reverse=False)
    sorted_with_indexes = {}
    index, c = 1, 0
    while c < len(sorted_dict):
        if c == 0:
            sorted_with_indexes[sorted_dict[c][0]] = index
            index += 1
            c += 1
            continue

        if sorted_dict[c][1] > sorted_dict[c-1][1]:
            sorted_with_indexes[sorted_dict[c][0]] = index
        else:
            sorted_with_indexes[sorted_dict[c][0]] = sorted_with_indexes[sorted_dict[c-1][0]]
        
        index += 1
        c += 1
    
    return sorted_with_indexes


def get_recommendation(user_id, model_id, user_rated_items_ids=None):
    '''
        Input:  user's id; user_rated_items_ids allows for filtering certain items from the user's rated pool

        Output: dict of item ids and their recommendation score; sorted list of (item, R)

        Formula:
                R^(a,i) = ra + (E_j sim(j,i) * (r(a,j) - rj))
            where,
            ra       = average ratings of user a
            rj       = average rating of item j
            r(a,j)   = rating user a gave to item j
            sim(j,i) = similarity between items j and i, for every j where r(a,j) exists
    '''

    # final ratings are stored as { "item_id": final_rating, ... }
    final_ratings_dict = {}

    user_rated_items = Utils.getUserRatingsForCity(user_id)
    if user_rated_items_ids is None:
        user_rated_items_ids = user_rated_items.keys()
    nr_user_rated_items = len(user_rated_items_ids)
    ra = Utils.getUserData(user_id)['average']
    sim_model = load_model(model_id)
    
    # load city items
    city_items_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/businesses.csv')
    total_items = len(city_items_df)
    rj = {}
    for j in user_rated_items_ids:
        rj[j] = Utils.getItemData(j, city_items_df)['rating']
    
    # find the predicted user rating for every item
    counter, percentage, start = 0, 0, time.perf_counter()
    for index, row in city_items_df.iterrows():
        i = row['id']
        # normalize rating from 1..5 to 0..1
        #ri = (Utils.getItemData(i, city_items_df)['rating']-1)/4
        #ri = Utils.getItemData(i, city_items_df)['rating'] / nr_user_rated_items

        # skip item if already rated by user
        if i in user_rated_items_ids:
            counter += 1
            continue

        weighted_sum, weighted_bottom = 0, 0
        for j in user_rated_items_ids:
            # clip the few negative similarity values that might appear 
            sim_ji = sim_model.get(j,i)
            if sim_ji <= 0:
                sim_ji = 0.000001

            r_aj = user_rated_items[j]

            #weighted_sum += sim_ji * ((r_aj - rj[j] + 4) / 2) * ri
            weighted_sum += sim_ji * (r_aj - rj[j])
            weighted_bottom += sim_ji

            #if i == 2446:
            #    print(f'sim_ji for {j}: {sim_ji}')

        final_ratings_dict[i] = ra + (weighted_sum/weighted_bottom)

        #if i == 2446:
        #    print(weighted_sum)

        counter += 1
        new_percentage = int(counter/total_items*100)
        if new_percentage > percentage:
            if new_percentage % 100 == 0:
                end = time.perf_counter()
                percentage = new_percentage
                t = end-start
                print(str(new_percentage) + '% -> ' + str(format(t, '.2f')) + 's')

    # get top-k items for user
    '''
    sorted_final_ratings = sorted(final_ratings_dict.items(), key=operator.itemgetter(1), reverse=True)
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
    '''

    return final_ratings_dict


def test_model(model_id):
    '''
        Input: user to find recommendation for & model
        Output: recommended item's id

        R^(a,i) = ra + (E_j sim(j,i) * (r(a,j) - rj))
        where,
            ra       = average ratings of user a
            rj       = average rating of item j
            r(a,j)   = rating user a gave to item j
            sim(j,i) = similarity between items j and i, for every j where r(a,j) exists
    '''

    # errors are stored as np.power(prediction - actual, 2)
    # apply np.sqrt(np.mean(errors)) to get RMSE
    # deviations are the percentage of deviation from the predicted value (not absolute values)
    errors = []
    deviations = []
    sort_devs = []
    sim_model = load_model(model_id)
    
    # load city items and reviews
    city_items_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/businesses.csv')
    reviews_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/reviews.csv')

    # load a random % of users
    users_df_non_shuffled = pd.read_csv('../yelp_dataset/resources/'+CITY+'/users.csv')
    users_df = users_df_non_shuffled.sample(frac=1).reset_index(drop=True)
    rows = len(users_df)
    users_df = users_df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.01)
    users_df = users_df[0:split_index]
    total_users = len(users_df)

    # iterate users
    start = time.perf_counter()
    counter, percentage = 0, 0
    print("{}\t{}\t{}\t{}\t{}".format("progress", "total time", "RMSE", "MD", "SORT DEV"))
    for index, row in users_df.iterrows():
        # get user data
        user_id = row['user']
        user_rated_items = Utils.getUserRatingsForCity(user_id, reviews_df)
        user_rated_items_ids = user_rated_items.keys()
        nr_user_rated_items = len(user_rated_items_ids)
        if nr_user_rated_items < 10:
            counter += 1
            new_percentage = int(counter/total_users*100)
            if new_percentage > percentage:
                if new_percentage % 5 == 0:
                    percentage = new_percentage
                    print_test_loading(new_percentage, start, errors, deviations, sort_devs)
            continue

        ra = Utils.getUserData(user_id, users_df)['average']
        
        # sort ratings
        sorted_ratings = get_sorted_dict_indexes(user_rated_items)
        # make equal ratings be arrays
        index_counter = Counter(sorted_ratings.values())
        for item in sorted_ratings:
            if index_counter[sorted_ratings[item]] > 1:
                sorted_ratings[item] = [sorted_ratings[item], sorted_ratings[item] + index_counter[sorted_ratings[item]] - 1]

        # find the predicted user rating for every item they've already rated
        sorted_predicted_values = {}
        for i in user_rated_items_ids:

            weighted_sum, weighted_bottom = 0, 0
            for j in user_rated_items_ids:
                # clip the few negative similarity values that might appear 
                sim_ji = sim_model.get(j,i)
                if sim_ji <= 0:
                    sim_ji = 0.000001

                r_aj = user_rated_items[j]
                rj = Utils.getItemData(j, city_items_df)['rating']

                # former calc
                #weighted_sum += sim_ji * ((r_aj - rj[j] + 4) / 2) * ri
                
                weighted_sum += sim_ji * (r_aj - rj)
                weighted_bottom += sim_ji

                # testing
                #if i == 2446:
                #    print(f'sim_ji for {j}: {sim_ji}')

            predicted_value = ra + (weighted_sum/weighted_bottom)
            errors.append(np.power(predicted_value - user_rated_items[i], 2))
            deviations.append(100 * (predicted_value - user_rated_items[i]) / user_rated_items[i])
            sorted_predicted_values[i] = predicted_value

        # find sorting deviation %
        sorted_predicted_ratings = get_sorted_dict_indexes(sorted_predicted_values)
        sort_dev = 0
        for i in sorted_predicted_ratings:
            if isinstance(sorted_ratings[i], int):
                sort_dev += 100 * abs(sorted_ratings[i] - sorted_predicted_ratings[i]) / (nr_user_rated_items - 1)
                continue
            if sorted_predicted_ratings[i] < sorted_ratings[i][0] or sorted_predicted_ratings[i] > sorted_ratings[i][1]:
                sort_dev += 100 * abs(sorted_predicted_ratings[i] - np.mean(sorted_ratings[i])) / (nr_user_rated_items - 1)
                continue
        sort_dev = sort_dev / nr_user_rated_items
        sort_devs.append(sort_dev)

        counter += 1
        new_percentage = int(counter/total_users*100)
        if new_percentage > percentage:
            if new_percentage % 5 == 0:
                percentage = new_percentage
                print_test_loading(new_percentage, start, errors, deviations, sort_devs)
                
    rmse = np.sqrt(np.mean(errors))
    md = np.mean(deviations)
    std = np.std(deviations)

    return rmse, md, std


# For Testing Purposes
if __name__ == '__main__':
    #train_model()

    get_recommendation('GlxJs5r01_yqIgb4CYtiog', MODEL)
    
    '''
    rmse, md, std = test_model(MODEL)
    print(f'rmse = {rmse}')
    print(f'mean of deviation % = {md}')
    print(f'deviation of deviation % = {std}')
    '''
    '''
    model = load_model('20210402181019')

    similarities = {}
    sim_list = []
    i, k, item= 0, 2846, 304
    print(model.get(item, item))
    while i <= k-1:
        similarities[i] = model.get(item,i)
        sim_list.append(similarities[i])
        i += 1

    sim_list = np.array(sim_list)
    print(f'Percentage of negative similarities for {item}: {(np.sum(sim_list < 0))/3518*100}')

    sorted_similarities = sorted(similarities.items(), key=operator.itemgetter(1), reverse=True)
    i, k = 0, 5
    print(f'top {k} items for item {item}')
    while i < k:
        print(sorted_similarities[i])
        print(Utils.getItemData(sorted_similarities[i][0]))
        i += 1
    '''
        
    '''
    model = load_model('20210331033705')
    print(model.get(304,3049))
    print(model.get(744,3049))
    print(model.get(2122,3049))
    '''    

    pass