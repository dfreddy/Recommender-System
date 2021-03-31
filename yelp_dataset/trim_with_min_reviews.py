import json, time, math, csv
import pandas as pd

'''
    selecting all businesses from Toronto
'''

city = 'Toronto'
biz_filename = 'resources/yelp_academic_dataset_business.json'

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
  newfile_name = 'resources/trim by review count/'+city+'/businesses_before_first_trimming.json'
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
  newfile_name = 'resources/trim by review count/'+city+'/businesses_first_trim.json'
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
  biz_file = open(filename, encoding='utf8', mode='r')
  data = json.load(biz_file)
  biz_list = []
  for i in data:
    biz_list.append(json.loads(i)['business_id'])

  return biz_lis


if __name__ == '__main__':

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
  biz_list = getBusinesses(biz_filename + extention)
  # select reviews from file
  file = open(reviews_filename + extention, encoding='utf8', mode='r')
  trimmed_filename = trimByBusinesses(file, set(biz_list))
  file.close()
  # trim off useless features
  file = open(trimmed_filename, encoding='utf8', mode='r')
  filename = trim_features(file)
  file.close()
  print('trimmed features...')
  # save to csv
  file = open(filename, encoding='utf8', mode='r')
  reviews_csv_filename = save_reviews_to_csv(file)
  file.close()
  print('saved to csv')