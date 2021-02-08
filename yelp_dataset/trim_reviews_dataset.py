import json, time, math

'''
    selecting all reviews by the users selected
'''

extention = '.json'
reviews_filename = 'resources/yelp_academic_dataset_review'
users_filename = 'resources/yelp_academic_dataset_user_10k'
trimmed_filename = 'resources/yelp_academic_dataset_review_trimmed_423822'


def trim(file, users_list):

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
    
    if isUserListed(json.loads(line)['user_id'], users_list):
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
  newfile_name = reviews_filename + '_trimmed' + extention
  newfile = open(newfile_name, encoding='utf8', mode='w')
  newfile.write(json.dumps(lines, indent=2))
  newfile.close()
  print(newfile_name)

  return


def getUsers(filename):
  
  users_file = open(filename, encoding='utf8', mode='r')
  data = json.load(users_file)
  users = []
  for i in data:
    users.append(json.loads(i)['user_id'])

  return users


def isUserListed(user_id, users_list):
  
  found = False
  if user_id in users_list: found = True
  
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
      'useful': r['useful'],
      'funny': r['funny'],
      'cool': r['cool'],
      'stars': r['stars']
    })
    
  # save reviews to new file
  newfile_name = 'resources/reviews' + extention
  newfile = open(newfile_name, encoding='utf8', mode='w')
  newfile.write(json.dumps(trimmed_reviews, indent=2))
  newfile.close()
  print(newfile_name)
  return


def main():
  
  '''
  # get list of users
  start = time.perf_counter()
  users_list = getUsers(users_filename + extention)
  end = time.perf_counter()
  print('time to get users list: ' + str(end-start) + 's')

  # select reviews from file
  file = open(reviews_filename + extention, encoding='utf8', mode='r')
  trim(file, users_list)
  file.close()
  '''
  
  file = open(trimmed_filename + extention, encoding='utf8', mode='r')
  trim_features(file)
  file.close()


main()