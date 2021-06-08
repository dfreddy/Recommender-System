import json, pprint, time
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# FUNCTIONS OVER THE CSV DATASETS
config = json.load(open('./config.json', 'r'))
CITY = config['city']

def getAllRatingsForItem(item, reviews_df=None):
  '''
      Returns all ratings for a certain item, in the format:
        { "user": rating, ... }
  '''

  ratings = {}

  if reviews_df is None:
    reviews_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/reviews.csv')

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


def getUsersThatRatedItem(item_id, user_filter=None, json_data=None):
  '''
      Returns all users that rated the input item in the format:
        [ user_id1, user_id2, ... ]
      
      item_id: input item
      user_filter: list of user ids to filter
      json_data: can preload the json data
  '''

  item_id = str(item_id)

  if json_data is None:
    filename = '../yelp_dataset/resources/'+CITY+'/user_ratings_by_item.json'
    try:
      file = open(filename, encoding='utf8', mode='r')
    except IOError:
      saveAllRatingsForAllItems()
      file = open(filename, encoding='utf8', mode='r')
    finally:
      json_data = json.load(file)
      users_rated = list(json_data[item_id].keys())

      if user_filter is None:
        return users_rated

      return list( set(users_rated) & set(user_filter) )
  
  else:
    users_rated = list(json_data[item_id].keys())

    if user_filter is None:
      return users_rated

    return list( set(users_rated) & set(user_filter) )

  return None


def getKMostEliteReviewers(item_id, k, json_data=None):
  '''
      Returns the IDs of the top-k most elite users that rated the input item
  '''

  # if K is too low, default to defining a top elite user as one with at least 8 years of being elite
  if k < 2:
    elite_threshold = 8

  item_id = str(item_id)

  if json_data is None:
    filename = '../yelp_dataset/resources/'+CITY+'/user_ratings_by_item.json'
    try:
      file = open(filename, encoding='utf8', mode='r')
    except IOError:
      saveAllRatingsForAllItems()
      file = open(filename, encoding='utf8', mode='r')
    finally:
      json_data = json.load(file)
      users_rated = list(json_data[item_id].keys())

      # retrive the user's elite values
      users_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/users.csv')
      users_elite_dict = {}
      for user in users_rated:
        user_elite_value = getUserData(user, user_df)['elite']
        users_elite_dict[user] = user_elite_value

      # sort elite values
      sorted_elite_users = sorted(users_elite_dict.items(), key=operator.itemgetter(1), reverse=True)
      top_k_elite_users = []

      if k > 1:
        i = 0
        while i<k:
          top_k_elite_users.append(sorted_elite_users[i][0])
          i += 1
      else:
        for elite_user in sorted_elite_users:
          if elite_user[1] > elite_threshold:
            top_k_elite_users.append(elite_user[0])
          else:
            break

      return top_k_elite_users

  else:
    users_rated = list(json_data[item_id].keys())

    # retrive the user's elite values
    users_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/users.csv')
    users_elite_dict = {}
    for user in users_rated:
      user_elite_value = getUserData(user, user_df)['elite']
      users_elite_dict[user] = user_elite_value

    # sort elite values
    sorted_elite_users = sorted(users_elite_dict.items(), key=operator.itemgetter(1), reverse=True)
    top_k_elite_users = []
    
    if k > 1:
      i = 0
      while i<k:
        top_k_elite_users.append(sorted_elite_users[i][0])
        i += 1
    else:
      for elite_user in sorted_elite_users:
        if elite_user[1] > elite_threshold:
          top_k_elite_users.append(elite_user[0])
        else:
          break

    return top_k_elite_users
  
  return None


def getFilteredAverageItemRating(item_id, user_filter, json_data=None):
  '''
      Returns the average rating of the input item, based on the ratings given by the user_filter list of users
  '''

  item_id = str(item_id)
  
  if json_data is None:
    filename = '../yelp_dataset/resources/'+CITY+'/user_ratings_by_item.json'
    try:
      file = open(filename, encoding='utf8', mode='r')
    except IOError:
      saveAllRatingsForAllItems()
      file = open(filename, encoding='utf8', mode='r')
    finally:
      json_data = json.load(file)
      total = 0
      k = len(user_filter)
      ratings = json_data[item_id]

      for user in user_filter:
        total += ratings[user]
  
      return total / k

  else:
    total = 0
    k = len(user_filter)
    ratings = json_data[item_id]

    for user in user_filter:
      total += ratings[user]

    return total / k
  
  return None


def saveAllRatingsForAllItems(city=None):
  '''
      Saves all ratings for all items into a json, in the format:
        { "item": { "user": rating, ... }, ... }
  '''

  if city is None:
    city = CITY
  
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


def getUserItemRating(user_id, item_id, reviews_df=None):
  '''
      Returns the value for the user-item pair
  '''

  if reviews_df is None:
    reviews_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/reviews.csv')

  df = reviews_df[(reviews_df.user==user_id) & (reviews_df.business==item_id)]

  if len(df) == 0:
    return None

  return df.rating


def getUserRatingsForCity(user_id, reviews_df=None):
  '''
      Returns all the items rated by the user in the format:
        { "item_id": rating, ... }
  '''

  if reviews_df is None:
    reviews_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/reviews.csv')
  
  df = reviews_df[reviews_df.user.isin([user_id])]
  ret = {}

  for index, row in df.iterrows():
    item_id = getItemIdByBusiness(row['business'], city_name=CITY)
    ret[item_id] = row['rating']

  return ret


def getAllUserRatings(user_id, reviews_df=None):
  '''
      Returns all the items rated by the user in the format:
        { "item_id": rating, ... }
  '''

  if reviews_df is None:
    reviews_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/users_all_reviews.csv')
    
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
    df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/users.csv')
  
  df = df[df.user.isin([user_id])]

  ret = df.to_dict(orient='records')
  if len(ret) == 1:
    return ret[0]

  return None


def getUserFriends(user_id):
  '''
      Returns the user's friends as a list
  '''

  filename = '../yelp_dataset/resources/'+CITY+'/users_friends.json'
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
    df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/users.csv')
  
  # randomizes the indexes first
  rows = len(df)
  df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)

  return df.sort_values(by='elite', ascending=False).head(k)['user'].tolist()


def getItemData(biz_id, df=None):
  '''
      Returns the business's data as a dict
      
      Input is the numeric city id, not the alphanumeric general id; either in string or int format
  '''

  if df is None:
    df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/businesses.csv')
  
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
    city_name = CITY

  if business_df is None:
    business_df = pd.read_csv('../yelp_dataset/resources/'+city_name+'/businesses.csv')

  index = business_df.index[business_df['business'] == biz]

  if len(index) == 0:
    print(biz)
    print(business_df['business'])

  return business_df["id"].values[index[0]]


def getAllItems():
  '''
      Returns a dict of all items' data
  '''

  items_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/businesses.csv')
  
  return items_df.to_dict(orient='records')


def getAllItemsIDs():
  '''
      Returns a list of all items ids
  '''

  items_df = pd.read_csv('../yelp_dataset/resources/'+CITY+'/businesses.csv')
  
  return items_df['id'].tolist()


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
  #getTopKEliteUsers(10)

  #getUserItemRating('TZQSUDDcA4ek5gBd6BzcjA','qUWqjjjfpB2-4P3He5rsKw')
  
  #print(getAllItems())

  pass
