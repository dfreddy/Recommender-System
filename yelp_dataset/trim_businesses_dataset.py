import json, time, math

'''
    selecting all businesses with valid reviews
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
  newfile.write(json.dumps(lines, indent=2))
  newfile.close()
  print(newfile_name)

  return


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
      'business_id': b['business_id'],
      'city': b['city'],
      'state': b['state'],
      'stars': b['stars'],
      'review_count': b['review_count'],
      'is_open': b['is_open'],
      'attributes': b['attributes'],
      'categories': b['categories']
    })
    
  # save businesses to new file
  newfile_name = 'resources/businesses' + extention
  newfile = open(newfile_name, encoding='utf8', mode='w')
  newfile.write(json.dumps(trimmed_businesses, indent=2))
  newfile.close()
  print(newfile_name)
  return


def main():

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

  file = open(trimmed_filename + extention, encoding='utf8', mode='r')
  trim_features(file)
  file.close()


main()