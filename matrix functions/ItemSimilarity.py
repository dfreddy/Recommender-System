import utils

city = 'Mississauga'

def cosine_similarity():
  '''
      Create item-item similarity matrix as a list:
        [ [item_a, item_b, similarity], ... ]
      Save to .csv for posterity
  '''

  file = open('../yelp_dataset/resources/'+city+'/user_ratings_by_item.json', encoding='utf8', mode='r')
  ratings_by_item = json.load(file)
  file.close()

  items_df = pd.read_csv('../yelp_dataset/resources/'+city+'/items.csv')

  ''' TODO
      for every item, calculate the cosine similarity between it and every other item
  '''