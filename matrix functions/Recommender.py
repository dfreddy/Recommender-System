import Utils
import numpy as np
import pandas as pd


def getRecommendation(user_id):
    '''
        Input: user's id
        Output: recommended item's id
    '''

    r = Utils.getAllUserRatings(user_id)
    ra = np.average(np.array(list(r.values())))

    