import json, pprint

def getAllRatingsForItem(item, reviews=None):
  '''
      Returns all ratings for a certain item, in the format:
        { "user": rating, ... }
  '''

  ratings = {}

  if reviews == None:
    file = open('../yelp_dataset/resources/reviews.collaborative.json', encoding='utf8', mode='r')
    reviews = json.load(file)

  for r in reviews:
    
    if r['business_id'] == item:
      ratings[r['user_id']] = r['stars']
    
  return ratings

def getAllRatingsForAllItems(reviews=None):
  '''
      Returns all ratings for all items, in the format:
        { "item": { "user": rating, ... }, ... }
  '''

  ratings = {}
  counter = 0

  if reviews == None:
    file = open('../yelp_dataset/resources/reviews.collaborative.json', encoding='utf8', mode='r')
    reviews = json.load(file)

  for r in reviews:
    
    ratings[r['business_id']] = getAllRatingsForItem(r['business_id'], reviews)

    counter += 1
    print(counter)

  return ratings


getAllRatingsForAllItems()