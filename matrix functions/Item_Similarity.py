import Utils, json, time, pprint, csv
import pandas as pd

city = 'Mississauga'


def save_to_csv(calc_type, values):
  '''
      saves similarity matrix to .csv
  '''
  fields = ['item_a', 'item_b', 'similarity']
  rows = []

  for key in values:
    a, b = Utils.getItemsFromKey(key)

    rows.append([
      a,
      b,
      values[key]
    ])

  with open('./resources/'+calc_type+'.csv', 'w', newline='', encoding='utf-8') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(rows)

  print('Saved')


def cosine_similarity():
  '''
      Create item-item similarity matrix as a list:
        { item_a,item_b: similarity, ... }
      Save to .csv for posterity

      Note:
        It becomes necessary to save entries for item_a,item_b and for item_b,item_a
        in order to later calculate the latent values
  '''

  file = open('../yelp_dataset/resources/'+city+'/user_ratings_by_item.json', encoding='utf8', mode='r')
  ratings_by_item = json.load(file)
  file.close()

  items_df = pd.read_csv('../yelp_dataset/resources/'+city+'/businesses.csv')

  similarity_matrix = {}
  counter, percentage = 0, 0
  total_items = len(items_df.index)

  start = time.perf_counter()

  for index_a, row_a in items_df.iterrows():
    # iterate every item
    item_a = str(row_a['id'])

    for index_b, row_b in items_df.iterrows():
      # calculate its similarity with every other item
      item_b = str(row_b['id'])
    
      # skip if it's the same item OR if their similarity has already been calculated
      if item_a != item_b and Utils.notRepeating(item_a, item_b, similarity_matrix):
        # their ratings by user
        item_a_vector = ratings_by_item[item_a]
        item_b_vector = ratings_by_item[item_b]
        common_users = set(item_a_vector.keys()).intersection(set(item_b_vector.keys()))

        # calculate similarity if they have common users
        if (len(common_users) > 0):
          dot_product = 0

          for user in common_users:
            dot_product += item_a_vector[user] * item_b_vector[user]

          similarity = dot_product / (sum(item_a_vector.values()) * sum(item_b_vector.values()))

          similarity_matrix[Utils.combineItemsToKey(item_a, item_b)] = similarity
          similarity_matrix[Utils.combineItemsToKey(item_b, item_a)] = similarity

          #print('Cosine similarity(a,b;b,a) = ' + str(format(similarity, '.4f')))

    counter += 1
    new_percentage = int(counter/total_items*100)
    if new_percentage > percentage:
      end = time.perf_counter()
      percentage = new_percentage
      t = (end-start)/60
      print(str(new_percentage) + '% -> ' + str(format(t, '.2f')) + 'm ... ' + str(format(100*t/new_percentage, '.2f')) + 'm')

  save_to_csv('cosine_similarity', similarity_matrix)


def AMSD_similarity():
  '''
      Create item-item similarity matrix as a list:
        { item_a,item_b: similarity, ... }
      Save to .csv for posterity
  '''

  file = open('../yelp_dataset/resources/'+city+'/user_ratings_by_item.json', encoding='utf8', mode='r')
  ratings_by_item = json.load(file)
  file.close()

  items_df = pd.read_csv('../yelp_dataset/resources/'+city+'/businesses.csv')

  similarity_matrix = {}
  counter, percentage = 0, 0
  total_items = len(items_df.index)

  start = time.perf_counter()

  for index_a, row_a in items_df.iterrows():
    # iterate every item
    item_a = str(row_a['id'])

    for index_b, row_b in items_df.iterrows():
      # calculate its similarity with every other item
      item_b = str(row_b['id'])
    
      # skip if it's the same item OR if their similarity has already been calculated
      if item_a != item_b and Utils.notRepeating(item_a, item_b, similarity_matrix):
        # their ratings by user
        item_a_vector = ratings_by_item[item_a]
        item_b_vector = ratings_by_item[item_b]
        common_users = set(item_a_vector.keys()).intersection(set(item_b_vector.keys()))

        # calculate similarity if they have common users
        if (len(common_users) > 0):
          msd, l = 0, pow(4,2)

          for user in common_users:
            msd += pow(item_a_vector[user] - item_b_vector[user], 2)
          msd = msd / len(common_users)
          msd = (l-msd)/l

          if msd > 0.0: # items where the mean difference exceeds the L Threshold are discarded
            similarity_ab = msd * (len(common_users)/len(item_a_vector)) * ((2*len(common_users)) / (len(item_a_vector)+len(item_b_vector)))
            similarity_ba = msd * (len(common_users)/len(item_b_vector)) * ((2*len(common_users)) / (len(item_a_vector)+len(item_b_vector)))
          
            similarity_matrix[Utils.combineItemsToKey(item_a, item_b)] = similarity_ab
            similarity_matrix[Utils.combineItemsToKey(item_b, item_a)] = similarity_ba

          #print('AMSD(a,b) = ' + str(format(similarity_ab, '.4f')))
          #print('AMSD(b,a) = ' + str(format(similarity_ba, '.4f')))

    counter += 1
    new_percentage = int(counter/total_items*100)
    if new_percentage > percentage:
      if new_percentage % 5 == 0:
        end = time.perf_counter()
        percentage = new_percentage
        t = (end-start)/60
        print(str(new_percentage) + '% -> ' + str(format(t, '.2f')) + 'm ... ' + str(format(100*t/new_percentage, '.2f')) + 'm')

  save_to_csv('AMSD_similarity', similarity_matrix)


cosine_similarity()
# AMSD_similarity()