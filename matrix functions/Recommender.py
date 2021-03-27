import Utils, svd_inference
import numpy as np
import pandas as pd


def train_model():
    item_similarity_matrix = svd_inference.get_similarity_matrix()
    item_similarity_matrix.save()


def get_recommendation(user_id):
    '''
        Input: user's id
        Output: recommended item's id

        R^(a,i) = ra + ( sim(i,j) * (r(a,j) - rj) ) / ( |sim(j,i)| )
        where,
            ra       = average ratings of user a
            rj       = average ratigns of item j
            r(a,j)   = rating user a gave to item j
            sim(i,j) = similarity between items i and j, for every j where r(a,j) exists
    '''

    r = Utils.getAllUserRatings(user_id)
    ra = np.average(np.array(list(r.values())))

    ''' TODO
        Maybe train the model once and save the resulting U S V arrays in their respective files        
    '''
    # item_similarity = svd_inference.get_similarity_matrix()
