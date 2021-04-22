import json, pprint, time
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# FUNCTIONS OVER THE CSV DATASETS
city = 'Toronto'

def getAllRatingsForItem(item, reviews_df=None):
  '''
      Returns all ratings for a certain item, in the format:
        { "user": rating, ... }
  '''

  ratings = {}

  if reviews_df is None:
    reviews_df = pd.read_csv('../yelp_dataset/resources/'+city+'/reviews.csv')

  ratings_df = reviews_df[reviews_df['business'].str.contains(item)]
  for index, row in ratings_df.iterrows():
    ratings[row['user']] = row['rating']

  return ratings


def getAllRatingsForAllItems(city):
  '''
      Returns all ratings for all items, in the format:
        { "item": { "user": rating, ... }, ... }
  '''

  filename = '../yelp_dataset/resources/'+city+'/user_ratings_by_item.json'

  try:
    file = open(filename, encoding='utf8', mode='r')
  except IOError:
    saveAllRatingsForAllItems(city)
    file = open(filename, encoding='utf8', mode='r')
  finally:
    return json.load(file)

  return None


def saveAllRatingsForAllItems(city):
  '''
      Saves all ratings for all items into a json, in the format:
        { "item": { "user": rating, ... }, ... }
  '''

  all_ratings = {}
  counter = 0

  items_df = pd.read_csv('../yelp_dataset/resources/'+city+'/businesses.csv')
  reviews_df = pd.read_csv('../yelp_dataset/resources/'+city+'/reviews.csv')
  
  start = time.perf_counter()

  for index, row in items_df.iterrows():
    ratings = getAllRatingsForItem(row['business'], reviews_df)
    all_ratings[row['id']] = ratings

    counter += 1
    if counter % 500 == 0:
      print(counter)

  end = time.perf_counter()
  print(str(format((end-start)/60, '.3f')) + 'm')
  print(str(format((end-start)/60/3518, '.6f')) + 'm per')

  file = open('../yelp_dataset/resources/'+city+'/user_ratings_by_item.json', encoding='utf8', mode='w')
  json.dump(all_ratings, file, indent=3)
  file.close()

  print('finished')


def getUserRatingsForCity(user_id, reviews_df=None):
  '''
      Returns all the items rated by the user in the format:
        { "item_id": rating, ... }
  '''

  if reviews_df is None:
    reviews_df = pd.read_csv('../yelp_dataset/resources/'+city+'/reviews.csv')
  
  df = reviews_df[reviews_df.user.isin([user_id])]
  ret = {}

  for index, row in df.iterrows():
    item_id = getItemIdByBusiness(row['business'], city_name=city)
    ret[item_id] = row['rating']

  return ret


def getAllUserRatings(user_id, reviews_df=None):
  '''
      Returns all the items rated by the user in the format:
        { "item_id": rating, ... }
  '''

  if reviews_df is None:
    reviews_df = pd.read_csv('../yelp_dataset/resources/'+city+'/users_all_reviews.csv')
    
  df = reviews_df[reviews_df.user.isin([user_id])]
  ret = {}

  for index, row in df.iterrows():
    item_id = getItemIdByBusiness(row['business'])
    ret[item_id] = row['rating']

  return ret


def getUserData(user_id, df=None):
  '''
      Returns the user's data as a dict
  '''

  if df is None:
    df = pd.read_csv('../yelp_dataset/resources/'+city+'/users.csv')
  
  df = df[df.user.isin([user_id])]

  ret = df.to_dict(orient='records')
  if len(ret) == 1:
    return ret[0]

  return None


def getUserFriends(user_id):
  '''
      Returns the user's friends as a list
  '''

  filename = '../yelp_dataset/resources/'+city+'/users_friends.json'
  file = open(filename, encoding='utf8', mode='r')
  users = json.load(file)
  friends = []

  if user_id in users:
    friends = users[user_id].replace(', ', ',').split(',')
  
  return friends


def getTopKEliteUsers(k, df=None):
  '''
      Returns a list with the top-k users with most elite years
  '''

  if df is None:
    df = pd.read_csv('../yelp_dataset/resources/'+city+'/users.csv')
  
  elite_users = df.sort_values(by='elite', ascending=False).head(k)['user'].tolist()

  return elite_users


def getItemData(biz_id, df=None):
  '''
      Returns the business's data as a dict
      Input is the numeric city id, not the alphanumeric general id; either in string or int format
  '''

  if df is None:
    df = pd.read_csv('../yelp_dataset/resources/'+city+'/businesses.csv')
  
  df = df[df.id.isin([int(biz_id)])]

  ret = df.to_dict(orient='records')
  if len(ret) == 1:
    return ret[0]

  return None


def getItemIdByBusiness(biz, business_df=None, city_name=None):
  '''
      Returns the numeric id based on the item's yelp alphanumeric id
  '''

  if city_name is None:
    city_name = city

  if business_df is None:
    business_df = pd.read_csv('../yelp_dataset/resources/'+city_name+'/businesses.csv')

  index = business_df.index[business_df['business'] == biz]

  if len(index) == 0:
    print(biz)
    print(business_df['business'])

  return business_df["id"].values[index[0]]


# GENERAL UTILITY FUNCTIONS

def clip(x):
    return np.clip(x, 0.0, 1.0)


def combineItemsToKey(a, b):
  return str(a) + ',' + str(b)


def getItemsFromKey(key):
  items = key.split(',')

  return items[0], items[1] 


def notRepeating(item_a, item_b, dictionary):
  return combineItemsToKey(item_a, item_b) not in dictionary


def clip_value(x):
  return np.clip(x, 1.0, 5.0)


def list_except(l, elem):

  if type(list()) != type(elem):
    return [x for i,x in enumerate(l) if x!=elem]

  return [x for i,x in enumerate(l) if x not in elem]


# For Testing Purposes
if __name__ == '__main__':
  #print(getAllUserRatings('JnPIjvC0cmooNDfsa9BmXg'))
  #print(getItemIdByBusiness('CV6edrz2Lv_kwyAGdswS2A'))
  #print(getUserData('dIIKEfOgo0KqUfGQvGikPg'))
  #print(getItemData('0'))
  
  #items = getUserRatingsForCity('no2KpuffhnfD9PIDdlRM9g')
  #for k in items:
  #  print(getItemData(k))
  #  print(f'received a rating: {items[k]}')

  
  #df = pd.read_csv('./resources/AMSD_similarity(L=16).csv')
  #index = df.index[(df['item_a'] == 0) & (df['item_b'] == 863)]
  #print(len(index))
  #print(df["similarity"].values[index[0]])
  
  #saveAllRatingsForAllItems('Toronto')

  #getUserFriends('I_6wY8_RsewziNnKhGZg4g')
  getTopKEliteUsers(10)

  pass
