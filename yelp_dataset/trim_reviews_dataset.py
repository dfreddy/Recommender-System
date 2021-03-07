import json, time, math, csv

'''
    selecting all reviews for the businesses from Toronto
'''

extention = '.json'
reviews_filename = 'resources/yelp_academic_dataset_review'
users_filename = 'resources/yelp_academic_dataset_user_10k'
trimmed_filename = 'resources/yelp_academic_dataset_review_trimmed_423822'
biz_filename = 'resources/yelp_academic_dataset_business_Toronto'


def trimByUsers(file, users_list):

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
    
    if isListed(json.loads(line)['user_id'], users_list):
      total_reviews_selected += 1
      lines.append(line.strip())
    
    total_reviews_counter += 1

    new_loading = math.floor( 100 * (total_reviews_counter / 8021122) )
    if new_loading > old_loading:
      old_loading = new_loading
      end = time.perf_counter()
      print(str(old_loading) + '% -> ' + str(format((end-start)/60, '.2f')) + 'm')
    
  print('selected ' + str(total_reviews_selected) + ' reviews')
  print('trimmed off ' + str(format(100 - ((total_reviews_selected/total_reviews_counter)*100), '.1f')) + '%')

  # save lines to new file
  newfile_name = reviews_filename + '_UserTrimmed' + extention
  newfile = open(newfile_name, encoding='utf8', mode='w')
  json.dump(lines, newfile, indent=2)
  newfile.close()
  print(newfile_name)

  return newfile_name


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
    
    if isListed(json.loads(line)['business_id'], biz_list):
      total_reviews_selected += 1
      lines.append(line.strip())
    
    total_reviews_counter += 1

    new_loading = math.floor( 100 * (total_reviews_counter / 8021122) )
    if new_loading > old_loading:
      old_loading = new_loading
      end = time.perf_counter()
      print(str(old_loading) + '% -> ' + str(format((end-start)/60, '.2f')) + 'm')
    
  print('selected ' + str(total_reviews_selected) + ' reviews')
  print('trimmed off ' + str(format(100 - ((total_reviews_selected/total_reviews_counter)*100), '.1f')) + '% from the total entries')

  # save lines to new file
  newfile_name = reviews_filename + '_BizTrimmed' + extention
  newfile = open(newfile_name, encoding='utf8', mode='w')
  json.dump(lines, newfile, indent=2)
  newfile.close()
  print(newfile_name)

  return newfile_name


def getUsers(filename):
  
  users_file = open(filename, encoding='utf8', mode='r')
  data = json.load(users_file)
  users = []
  for i in data:
    users.append(json.loads(i)['user_id'])

  return users


def getBusinesses(filename):
  
  biz_file = open(filename, encoding='utf8', mode='r')
  data = json.load(biz_file)
  biz_list = []
  for i in data:
    biz_list.append(json.loads(i)['business_id'])

  return biz_list


def isListed(item_id, item_list):
  
  found = False
  if item_id in item_list: found = True
  
  return found


def trim_features(file):

  trimmed_reviews = []
  reviews = json.load(file)

  for r in reviews:
    r = json.loads(r)

    trimmed_reviews.append({
      'user_id': r['user_id'],
      'review_id': r['review_id'],
      'business_id': r['business_id'],
      'stars': r['stars']
    })
    
  # save reviews to new file
  newfile_name = 'resources/Toronto/reviews' + extention
  newfile = open(newfile_name, encoding='utf8', mode='w')
  json.dump(trimmed_reviews, newfile, indent=2)
  newfile.close()

  return newfile_name


def save_to_csv(file):
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

  with open('./resources/Toronto/reviews.csv', 'w', newline='', encoding='utf-8') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(rows)


def main():
  
  '''
  # get list of users
  start = time.perf_counter()
  users_list = getUsers(users_filename + extention)
  end = time.perf_counter()
  print('time to get users list: ' + str(end-start) + 's')

  # select reviews from file
  file = open(reviews_filename + extention, encoding='utf8', mode='r')
  trimByUsers(file, users_list)
  file.close()
  '''
  
  # get list of businesses
  biz_list = getBusinesses(biz_filename + extention)
  print('found ' + str(len(biz_list)) + ' businesses')

  # select reviews from file
  file = open(reviews_filename + extention, encoding='utf8', mode='r')
  trimmed_filename = trimByBusinesses(file, biz_list)
  file.close()
  
  # trim off useless features
  print('trimming features...')
  file = open(trimmed_filename, encoding='utf8', mode='r')
  filename = trim_features(file)
  file.close()

  # save to csv
  file = open(filename, encoding='utf8', mode='r')
  save_to_csv(file)
  file.close()


main()