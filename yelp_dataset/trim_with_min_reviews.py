import json, time, math, csv
import pandas as pd

'''
    selecting all businesses from Toronto
'''

city = 'Toronto'
biz_filename = 'resources/yelp_academic_dataset_business.json'
reviews_filename = 'resources/yelp_academic_dataset_review.json'
users_filename = 'resources/yelp_academic_dataset_user.json'

# BUSINESSES
# used to select all businesses from Toronto
def selectFromCity(file):
  lines = []
  total_biz_counter = 0
  total_biz_selected = 0
  old_loading = 0

  print(str(old_loading) + '%')
  start = time.perf_counter()

  while(True):
    
    line = None
    line = file.readline()
    if not line:
      break
    
    json_line = json.loads(line)
    if json_line['city'] == city:
      if json_line['review_count'] >= 50:
        total_biz_selected += 1
        lines.append(line.strip())
    
    total_biz_counter += 1

    new_loading = math.floor( 100 * (total_biz_counter / 209393) )
    if new_loading > old_loading:
      if new_loading % 10 == 0:
        old_loading = new_loading
        end = time.perf_counter()
        print(str(old_loading) + '% -> ' + str(format(end-start, '.2f')) + 's')
    
  print('selected ' + str(total_biz_selected) + ' businesses from ' + city)
  print('trimmed off ' + str(format(100 - ((total_biz_selected/total_biz_counter)*100), '.1f')) + '% from the total entries')

  # save lines to new file
  newfile_name = 'resources/trim by review count/'+city+'/businesses_before_trimming.json'
  newfile = open(newfile_name, encoding='utf8', mode='w')
  json.dump(lines, newfile, indent=2)
  newfile.close()

  return newfile_name


def trim_biz_features(file):
  trimmed_businesses = []
  businesses = json.load(file)

  for b in businesses:
    b = json.loads(b)

    trimmed_businesses.append({
      'name': b['name'],
      'business_id': b['business_id'],
      'stars': b['stars'],
      'categories': b['categories']
    })
    
  # save businesses to new file
  newfile_name = 'resources/trim by review count/'+city+'/businesses_trimmed.json'
  newfile = open(newfile_name, encoding='utf8', mode='w')
  json.dump(trimmed_businesses, newfile, indent=2)
  newfile.close()
  
  return newfile_name


def save_biz_to_csv(file):
  '''
      saves businesses file from .json to a simplified .csv
  '''
  fields = ['id', 'business', 'name', 'rating', 'categories']
  rows = []
  businesses = json.load(file)
  id_counter = -1

  for b in businesses:
    id_counter += 1
    rows.append([
      str(id_counter),
      b['business_id'],
      b['name'],
      str(b['stars']),
      b['categories']
    ])

  outname = file.name[:len(file.name)-5]+'.csv'
  with open(outname, 'w', newline='', encoding='utf-8') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(rows)

  return outname


# REVIEWS
def getBusinesses(filename):
  biz_df = pd.read_csv(filename)
  biz_list = []

  for index, row in biz_df.iterrows():
    biz_list.append(row['business'])

  return biz_list


def trimByBusinesses(file, biz_list):
  lines = []
  total_reviews_counter = 0
  total_reviews_selected = 0
  old_loading = 0

  print(str(old_loading) + '%')
  start = time.perf_counter()

  while(True):
    
    line = None

    line = file.readline()
    if not line:
      break
    
    if json.loads(line)['business_id'] in biz_list:
      total_reviews_selected += 1
      lines.append(line.strip())
    
    total_reviews_counter += 1

    new_loading = math.floor( 100 * (total_reviews_counter / 8021122) )
    if new_loading > old_loading:
      if new_loading % 10 == 0:
        old_loading = new_loading
        end = time.perf_counter()
        print(str(old_loading) + '% -> ' + str(format((end-start)/60, '.2f')) + 'm')
    
  print('selected ' + str(total_reviews_selected) + ' reviews')
  print('trimmed off ' + str(format(100 - ((total_reviews_selected/total_reviews_counter)*100), '.1f')) + '% from the total entries')

  # save lines to new file
  newfile_name = 'resources/trim by review count/'+city+'/reviews_before_trimming.json'
  newfile = open(newfile_name, encoding='utf8', mode='w')
  json.dump(lines, newfile, indent=2)
  newfile.close()
  print(newfile_name)

  return newfile_name


def trim_reviews_features(file, biz_csv):
  reviews = json.load(file)
  trimmed_reviews = []
  biz_df = pd.read_csv(biz_csv)

  for r in reviews:
    r = json.loads(r)
    
    trimmed_reviews.append({
      'user_id': r['user_id'],
      'business_id': str(getItemIdByBusiness(r['business_id'], biz_df)),
      'stars': str(r['stars'])
    })
    
  # save reviews to new file
  newfile_name = 'resources/trim by review count/'+city+'/reviews_trimmed.json'
  newfile = open(newfile_name, encoding='utf8', mode='w')
  json.dump(trimmed_reviews, newfile, indent=2)
  newfile.close()

  return newfile_name


def getItemIdByBusiness(biz, business_df=None):
  '''
      Returns the numeric id based on the item's yelp alphanumeric id
  '''

  index = business_df.index[business_df['business'] == biz]

  return business_df["id"].values[index[0]]


def save_reviews_to_csv(file):
  '''
      saves reviews file from .json to a simplified .csv
  '''
  fields = ['user', 'business', 'rating']
  rows = []
  reviews = json.load(file)

  for r in reviews:
    rows.append([
      r['user_id'],
      r['business_id'],
      str(r['stars'])
    ])

  outname = file.name[:len(file.name)-5]+'.csv'
  with open(outname, 'w', newline='', encoding='utf-8') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(rows)

  return outname


# USERS
def getReviewsUserIDs(filename):
  reviews_df = pd.read_csv(filename)
  users_dict = {}

  for index, row in reviews_df.iterrows():
    users_dict[row['user']] = True

  return users_dict


def trimByReviews(file, users_dict):
  lines = []
  total_users_counter = 0
  total_users_selected = 0
  old_loading = 0

  print(str(old_loading) + '%')
  start = time.perf_counter()

  while(True):
    
    line = None

    line = file.readline()
    if not line:
      break
    
    json_line = json.loads(line)
    if users_dict.get(json_line['user_id'], False):
      total_users_selected += 1
      lines.append(line.strip())
    
    total_users_counter += 1

    new_loading = math.floor( 100 * (total_users_counter / 2000000) )
    if new_loading > old_loading:
      if new_loading % 5 == 0:
        old_loading = new_loading
        end = time.perf_counter()
        print(str(old_loading) + '% -> ' + str(format((end-start)/60, '.2f')) + 'm')
    
  print('selected ' + str(total_users_selected) + ' users')
  print('trimmed off ' + str(format(100 - ((total_users_selected/total_users_counter)*100), '.1f')) + '% from the total entries')

  # save lines to new file
  newfile_name = 'resources/trim by review count/'+city+'/users_before_trimming.json'
  newfile = open(newfile_name, encoding='utf8', mode='w')
  json.dump(lines, newfile, indent=2)
  newfile.close()
  print(newfile_name)

  return newfile_name


def trim_users_features(file):
  trimmed_users = []
  users_friends = {}
  users = json.load(file)

  for u in users:
    u = json.loads(u)

    # count the nr of times the user has been elite
    elite = u['elite'].count(',')
    if len(u['elite']) != 0: elite += 1

    trimmed_users.append({
      'user_id': u['user_id'],
      'name': u['name'],
      'elite': elite,
      'average_stars': u['average_stars']
    })

    users_friends[u['user_id']] = u['friends']
    
  # save users to file
  outfile_name = 'resources/trim by review count/'+city+'/users_trimmed.json'
  outfile = open(outfile_name, encoding='utf8', mode='w')
  json.dump(trimmed_users, outfile, indent=2)
  outfile.close()
  print(outfile_name)
  
  # save friendslists to file
  # note: when opening, use .split() on the items to turn to list
  #       lists ocupy more space than simple strings
  friendsfile_name = 'resources/trim by review count/'+city+'/users_friends.json'
  friendsfile = open(friendsfile_name, encoding='utf8', mode='w')
  json.dump(users_friends, friendsfile, indent=2)
  friendsfile.close()
  print(friendsfile_name)

  return outfile_name


def save_users_to_csv(file):
  '''
      saves users file from .json to a simplified .csv
  '''
  fields = ['user', 'name', 'elite', 'average']
  rows = []
  users = json.load(file)

  for u in users:
    rows.append([
      u['user_id'],
      u['name'],
      str(u['elite']),
      str(u['average_stars'])
    ])

  outfile = file.name[:len(file.name)-5]+'.csv'
  with open(outfile, 'w', newline='', encoding='utf-8') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(rows)

  return outfile


if __name__ == '__main__':
  '''
  # SELECT BUSINESSES
  # select all businesses from Mississauga
  file = open(biz_filename, encoding='utf8', mode='r')
  city_filename = selectFromCity(file)
  file.close()
  # trim the useless features
  file = open(city_filename, encoding='utf8', mode='r')
  filename = trim_biz_features(file)
  file.close()
  print('saved to json')
  # save to csv
  file = open(filename, encoding='utf8', mode='r')
  biz_csv_filename = save_biz_to_csv(file)
  file.close()
  print('saved to csv')

  # SELECT REVIEWS
  # get list of businesses
  biz_list = getBusinesses(biz_csv_filename)
  # select reviews from file
  file = open(reviews_filename, encoding='utf8', mode='r')
  trimmed_filename = trimByBusinesses(file, set(biz_list))
  file.close()
  # trim off useless features
  file = open(trimmed_filename, encoding='utf8', mode='r')
  filename = trim_reviews_features(file, biz_csv_filename)
  file.close()
  print('trimmed features...')
  # save to csv
  file = open(filename, encoding='utf8', mode='r')
  reviews_csv_filename = save_reviews_to_csv(file)
  file.close()
  print('saved to csv')
  '''
  # SELECT USERS
  # get reviews ids
  reviews_csv_filename = 'resources/trim by review count/Toronto/reviews_trimmed.csv'
  users_dict = getReviewsUserIDs(reviews_csv_filename)
  print('found ' + str(len(users_dict)) + ' unique users')
  # get users by reviews
  file = open(users_filename, encoding='utf8', mode='r')
  selected_users_filename = trimByReviews(file, users_dict)
  file.close()
  # trim features
  file = open(selected_users_filename, encoding='utf8', mode='r')
  trimmed_users_filename = trim_users_features(file)
  file.close()
  print('trimmed features...')
  # save to csv
  file = open(trimmed_users_filename, encoding='utf8', mode='r')
  users_csv_filename = save_users_to_csv(file)
  file.close()
  print('saved to csv')
