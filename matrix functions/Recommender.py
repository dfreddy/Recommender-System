import Utils, SVD_Inference, Matrix, operator, time
import numpy as np
import pandas as pd

''' TODO
    In order to evaluate the RMSE of the Recommender System
    run a version of get_recommendation where the only items calculated are the ones already rated by the user
'''

city = 'Mississauga'

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

    user_rated_items = Utils.getUserRatingsForCity(user_id, city)
    user_rated_items_ids = user_rated_items.keys()
    nr_user_rated_items = len(user_rated_items_ids)
    ra = Utils.getUserData(user_id)['average']
    sim_model = load_model(model_id)
    
    # load city items
    city_items_df = pd.read_csv('../yelp_dataset/resources/'+city+'/businesses.csv')
    rj = {}
    for j in user_rated_items_ids:
        rj[j] = Utils.getItemData(j, city_items_df)['rating']
    
    # find the predicted user rating for every item
    counter, percentage, total_items, start = 0, 0, 3518, time.perf_counter()
    for index, row in city_items_df.iterrows():
        i = row['id']
        # normalize rating from 1..5 to 0..1
        ri = (Utils.getItemData(i, city_items_df)['rating'] - 1 ) / 4
        #ri = Utils.getItemData(i, city_items_df)['rating'] / nr_user_rated_items

        # skip item if already rated by user
        if i in user_rated_items_ids:
            continue

        weighted_sum, weighted_bottom = 0, 0
        for j in user_rated_items_ids:
            sim_ji = sim_model.get(j,i)
            r_aj = user_rated_items[j]

            weighted_sum += sim_ji * (r_aj - rj[j] + 4) / 2
            weighted_bottom += abs(sim_ji)

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
                t = (end-start)/60
                print(str(new_percentage) + '% -> ' + str(format(t, '.2f')) + 'm')

    # get top-k items for user
    sorted_final_ratings = sorted(final_ratings_dict.items(), key=operator.itemgetter(1), reverse=True)
    i, k = 0, 5
    print(f'top {k} items')
    while i < k:
        print(sorted_final_ratings[i])
        print(Utils.getItemData(sorted_final_ratings[i][0]))
        i += 1


# For Testing Purposes
if __name__ == '__main__':
    #train_model()

    #get_recommendation('no2KpuffhnfD9PIDdlRM9g', '20210331011104')
    #get_recommendation('iX1IIVWt5__u7ykkczLsRA', '20210331033705')
    get_recommendation('GlxJs5r01_yqIgb4CYtiog', '20210331033705')
    
    '''
    model = load_model('20210331033705')

    similarities = {}
    i, k = 0, 3518
    while i <= k-1:
        similarities[i] = model.get(304,i)
        i += 1

    sorted_similarities = sorted(similarities.items(), key=operator.itemgetter(1), reverse=False)
    i, k = 0, 5
    print(f'top {k} items for item 304')
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