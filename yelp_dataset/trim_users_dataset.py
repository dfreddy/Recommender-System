import json, time, math

'''
    selecting 1/20th of the full users dataset
'''

extention = '.json'
filename = 'resources/yelp_academic_dataset_user'
trimmed_filename = 'resources/yelp_academic_dataset_user_trimmed_10k'


def trim(file):

  while(True):
    
    lines = []
    line = None

    for line_counter in range(1, 9844):
      print(line_counter)
      line = file.readline()
      if not line:
        break
      lines.append(line.strip())
    
  # save lines to new file
  newfile_name = filename + '_10k' + extention
  newfile = open(newfile_name, encoding='utf8', mode='w')
  newfile.write(json.dumps(lines, indent=2))
  newfile.close()
  print(newfile_name)

  return newfile_name


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
    
    if isUserListed(json.loads(line)['user_id'], users_dict):
      total_users_selected += 1
      lines.append(line.strip())
    
    total_users_counter += 1

    new_loading = math.floor( 100 * (total_users_counter / 2000000) )
    if new_loading > old_loading:
      old_loading = new_loading
      end = time.perf_counter()
      print(str(old_loading) + '% -> ' + str(format((end-start)/60, '.2f')) + 'm')
    
  print('selected ' + str(total_users_selected) + ' users')
  print('trimmed off ' + str(format(100 - ((total_users_selected/total_users_counter)*100), '.1f')) + '% from the total entries')

  # save lines to new file
  newfile_name = filename + '_ReviewsTrimmed' + extention
  newfile = open(newfile_name, encoding='utf8', mode='w')
  newfile.write(json.dumps(lines, indent=2))
  newfile.close()
  print(newfile_name)

  return newfile_name


def trim_features(file):

  print('starting trimming...')
  trimmed_users = []
  users = json.load(file)

  for u in users:
    u = json.loads(u)

    # count the nr of times the user has been elite
    elite = u['elite'].count(',')
    if len(u['elite']) != 0: elite += 1

    trimmed_users.append({
      'user_id': u['user_id'],
      'name': u['name'],
      'review_count': u['review_count'],
      'useful': u['useful'],
      'funny': u['funny'],
      'cool': u['cool'],
      'elite': elite,
      'average_stars': u['average_stars']
    })
    
  # save users to new file
  newfile_name = 'resources/users' + extention
  newfile = open(newfile_name, encoding='utf8', mode='w')
  newfile.write(json.dumps(trimmed_users, indent=2))
  newfile.close()
  print(newfile_name)
  
  return newfile_name


def getReviewsUserIDs(filename):
  
  reviews_file = open(filename, encoding='utf8', mode='r')
  data = json.load(reviews_file)
  users_dict = {}
  for i in data:
    user_id = i['user_id']
    users_dict[user_id] = True

  return users_dict


def isUserListed(user_id, users_dict):
    
  return users_dict.get(user_id, False)


def main():
  
  # file = open(filename + extention, encoding='utf8', mode='r')
  # trim(file)
  # file.close()

  # file = open(trimmed_filename + extention, encoding='utf8', mode='r')
  # trim_features(file)
  # file.close()

  # get reviews ids
  users_dict = getReviewsUserIDs('resources/reviews.json')
  print('found ' + str(len(users_dict)) + ' unique users')

  # get users by reviews
  file = open(filename + extention, encoding='utf8', mode='r')
  selected_filename = trimByReviews(file, users_dict)
  file.close()

  # trim features
  file = open(selected_filename, encoding='utf8', mode='r')
  trim_features(file)
  file.close()


main()