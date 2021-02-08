import json

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
    return
    
    # if eof
    if not line:
      return


def trim_features(file):

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
  return


def main():
  
  # file = open(filename + extention, encoding='utf8', mode='r')
  # trim(file)
  # file.close()

  # file = open(trimmed_filename + extention, encoding='utf8', mode='r')
  # trim_features(file)
  # file.close()


main()