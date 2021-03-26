import json, time, math, csv

'''
    selecting all businesses from Las Vegas
'''

extention = '.json'
biz_filename = 'resources/yelp_academic_dataset_business'
reviews_filename = 'resources/yelp_academic_dataset_review_trimmed_423822'
trimmed_filename = 'resources/yelp_academic_dataset_business_trimmed_84015'


def trim(file, biz_dict):
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
    
    if isBizListed(json.loads(line)['business_id'], biz_dict):
      total_biz_selected += 1
      lines.append(line.strip())
    
    total_biz_counter += 1

    new_loading = math.floor( 100 * (total_biz_selected / len(biz_dict)) )
    if new_loading > old_loading:
      old_loading = new_loading
      end = time.perf_counter()
      print(str(old_loading) + '% -> ' + str(format(end-start, '.2f')) + 's')
    
  print('selected ' + str(total_biz_selected) + ' reviews')
  print('trimmed off ' + str(format(100 - ((total_biz_selected/total_biz_counter)*100), '.1f')) + '%')

  # save lines to new file
  newfile_name = biz_filename + '_trimmed' + extention
  newfile = open(newfile_name, encoding='utf8', mode='w')
  json.dump(lines, newfile, indent=2)
  newfile.close()
  print(newfile_name)

  return


# used to select all businesses from Las Vegas
def selectFromCity(file, city):
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
    
    if json.loads(line)['city'] == city:
      total_biz_selected += 1
      lines.append(line.strip())
    
    total_biz_counter += 1

    new_loading = math.floor( 100 * (total_biz_counter / 209393) )
    if new_loading > old_loading:
      old_loading = new_loading
      end = time.perf_counter()
      print(str(old_loading) + '% -> ' + str(format(end-start, '.2f')) + 's')
    
  print('selected ' + str(total_biz_selected) + ' reviews from ' + city)
  print('trimmed off ' + str(format(100 - ((total_biz_selected/total_biz_counter)*100), '.1f')) + '% from the total entries')

  # save lines to new file
  newfile_name = biz_filename + '_' + city + extention
  newfile = open(newfile_name, encoding='utf8', mode='w')
  json.dump(lines, newfile, indent=2)
  newfile.close()
  print(newfile_name)

  return newfile_name


def getBusinesses(filename):
  reviews_file = open(filename, encoding='utf8', mode='r')
  data = json.load(reviews_file)
  biz_dict = {}
  for i in data:
    biz_id = json.loads(i)['business_id']
    biz_dict[biz_id] = True

  return biz_dict


def isBizListed(biz_id, biz_dict):
  return biz_dict.get(biz_id, False)


def trim_features(file):
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
  newfile_name = 'resources/Mississauga/businesses' + extention
  newfile = open(newfile_name, encoding='utf8', mode='w')
  json.dump(trimmed_businesses, newfile, indent=2)
  newfile.close()
  
  return newfile_name


def countCity(file, city):
  counter = 0

  while(True):

    line = file.readline()
    if not line:
      break
    
    if json.loads(line)['city'] == city:
      counter += 1
  
  return counter


def save_to_csv(file):
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

  with open('./resources/Mississauga/businesses.csv', 'w', newline='', encoding='utf-8') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(rows)


if __name__ == '__main__':
  '''
  # get dict of biz ids
  start = time.perf_counter()
  biz_dict = getBusinesses(reviews_filename + extention)
  end = time.perf_counter()
  print('time to get biz dict: ' + str(end-start) + 's')
  print('nr of unique businesses: ' + str(len(biz_dict)))

  # select businesses from file
  file = open(biz_filename + extention, encoding='utf8', mode='r')
  trim(file, biz_dict)
  file.close()
  '''

  '''
  file = open(trimmed_filename + extention, encoding='utf8', mode='r')
  trim_features(file)
  file.close()
  '''
  
  '''
  # select all businesses from Las Vegas
  file = open(biz_filename + extention, encoding='utf8', mode='r')
  city_filename = selectFromCity(file, 'Las Vegas')
  file.close()

  # trim the useless features
  file = open('./resources/yelp_academic_dataset_business_Las Vegas.json', encoding='utf8', mode='r')
  trim_features(file)
  file.close()
  '''

  # select all businesses from Mississauga
  file = open(biz_filename + extention, encoding='utf8', mode='r')
  city_filename = selectFromCity(file, 'Mississauga')
  file.close()

  # trim the useless features
  file = open(city_filename, encoding='utf8', mode='r')
  filename = trim_features(file)
  file.close()

  print('saved to json')

  # save to csv
  file = open(filename, encoding='utf8', mode='r')
  save_to_csv(file)
  file.close()

  print('saved to csv')