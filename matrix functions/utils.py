import json, pprint, time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

city = 'Mississauga'


def getAllRatingsForItem(item, reviews_df=None):
  '''
      Returns all ratings for a certain item, in the format:
        { "user": rating, ... }
  '''

  # start = time.perf_counter()

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
    all_ratings[row['business']] = ratings

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


def combineItemsToKey(a, b):
  
  return a + ',' + b


def getItemsFromKey(key):

  items = key.split(',')

  return items[0], items[1] 


def notRepeating(item_a, item_b, dictionary):

  return combineItemsToKey(item_a, item_b) not in dictionary


# getAllRatingsForItem('YJ2Y_asLIlbo-uijVugLow')
# saveAllRatingsForAllItems('Mississauga')
# getAllRatingsForAllItems('Mississauga')
