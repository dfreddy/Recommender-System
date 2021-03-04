import utils

def cosine_similarity():
  '''
      Create item-item similarity matrix as a list:
        [ [item_a, item_b, similarity], ... ]
      Save to .csv for posterity
  '''

  file = open('../yelp_dataset/resources/reviews.collaborative.json', encoding='utf8', mode='r')
  reviews = json.load(file)
