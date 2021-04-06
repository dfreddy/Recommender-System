import Utils, SVD_Inference, Matrix, operator, time, math
import numpy as np
import pandas as pd


CITY = 'Toronto'
MODEL = '20210406155019'

def train_model():
    item_similarity_matrix = SVD_Inference.get_similarity_matrix()
    item_similarity_matrix.save()


def load_model(model_id):
    model = Matrix.PredictionSVD()
    model.load(model_id)

    return model


def get_recommendation(user_id, model_id):
    '''
        Input: user's id
        Output: recommended item's id

        R^(a,i) = ra + (E_j sim(j,i) * (r(a,j) - rj))
        where,
            ra       = average ratings of user a
            rj       = average rating of item j
            r(a,j)   = rating user a gave to item j
            sim(j,i) = similarity between items j and i, for every j where r(a,j) exists
    '''

    # final ratings are stored as { "item_id": final_rating, ... }
    final_ratings_dict = {}

    user_rated_items = Utils.getUserRatingsForCity(user_id, CITY)
    user_rated_items_ids = user_rated_items.keys()
    nr_user_rated_items = len(user_rated_items_ids)
    ra = Utils.getUserData(user_id)['average']
    sim_model = load_model(model_id)
    
    # load city items
    city_items_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/businesses.csv')
    total_items = city_items_df.size
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
            if new_percentage % 10 == 0:
                end = time.perf_counter()
                percentage = new_percentage
                t = end-start
                print(str(new_percentage) + '% -> ' + str(format(t, '.2f')) + 's')

    # get top-k items for user
    sorted_final_ratings = sorted(final_ratings_dict.items(), key=operator.itemgetter(1), reverse=True)
    i, k = 0, 5
    print(f'top {k} items')
    while i < k:
        print(sorted_final_ratings[i])
        print(Utils.getItemData(sorted_final_ratings[i][0]))
        i += 1


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
    sim_model = load_model(model_id)
    
    # load city items
    city_items_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/businesses.csv')

    # load a random % of users
    users_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/users.csv')
    users_df = users_df.sample(frac=1).reset_index(drop=True)
    rows = len(users_df)
    users_df = users_df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.02)
    users_df = users_df[0:split_index]
    total_users = len(users_df)

    # iterate users
    start = time.perf_counter()
    counter, percentage = 0, 0
    print("{}\t{}\t{}\t{}".format("progress", "total time", "RMSE", "MD"))
    for index, row in users_df.iterrows():
        # get user data
        user_id = row['user']
        user_rated_items = Utils.getUserRatingsForCity(user_id, CITY)
        user_rated_items_ids = user_rated_items.keys()
        nr_user_rated_items = len(user_rated_items_ids)
        if nr_user_rated_items < 5:
            counter += 1
            continue

        ra = Utils.getUserData(user_id, users_df)['average']
        
        # find the predicted user rating for every item they've already rated
        for i in user_rated_items_ids:

            weighted_sum, weighted_bottom = 0, 0
            for j in user_rated_items_ids:
                # clip the few negative similarity values that might appear 
                sim_ji = sim_model.get(j,i)
                if sim_ji <= 0:
                    sim_ji = 0.000001

                r_aj = user_rated_items[j]
                rj = Utils.getItemData(j, city_items_df)['rating']

                #weighted_sum += sim_ji * ((r_aj - rj[j] + 4) / 2) * ri
                weighted_sum += sim_ji * (r_aj - rj)
                weighted_bottom += sim_ji

                #if i == 2446:
                #    print(f'sim_ji for {j}: {sim_ji}')

            if weighted_bottom != 0:
                predicted_value = ra + (weighted_sum/weighted_bottom)
                errors.append(np.power(predicted_value - user_rated_items[i], 2))
                deviations.append(100 * (predicted_value - user_rated_items[i]) / user_rated_items[i])

        counter += 1
        new_percentage = int(counter/total_users*100)
        if new_percentage > percentage:
            if new_percentage % 5 == 0:
                end = time.perf_counter()
                percentage = new_percentage
                t = (end-start)/60
                rmse = format(np.sqrt(np.mean(errors)), '.3f')
                md = format(np.mean(deviations), '.1f')
                print("{}\t\t{}\t\t{}\t{}".format(str(new_percentage) + '%', str(format(t, '.1f')) + 'm', rmse, md))

    rmse = np.sqrt(np.mean(errors))
    md = np.mean(deviations)

    return rmse, md


# For Testing Purposes
if __name__ == '__main__':
    #train_model()

    #get_recommendation('no2KpuffhnfD9PIDdlRM9g', '20210331011104')
    #get_recommendation('iX1IIVWt5__u7ykkczLsRA', '20210331033705')
    #get_recommendation('GlxJs5r01_yqIgb4CYtiog', '20210331033705')
    rmse, md = test_model(MODEL)
    print(f'rmse = {rmse}')
    print(f'mean deviation % = {md}')
    
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

'''
    TODO

    learn similarity between 304 and 2446
    both are bars
    the similarity is very low

    learn similarity between 304 and 1033
    one is a bar while the other is a sports shop
    the similarity is very high
'''