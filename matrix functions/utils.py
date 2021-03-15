import json, pprint, time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tensorflow.core.framework import summary_pb2


# SVD INFERENCE FUNCTIONS

def inference_svd(item_a_batch, item_b_batch, item_num, dim=5, device="/cpu:0"):
  with tf.device('/cpu:0'):
    bias_global = tf.get_variable('bias_global', shape=[])
    embd_bias_item_a = tf.get_variable("embd_bias_item_a", shape=[item_num])
    embd_bias_item_b = tf.get_variable("embd_bias_item_b", shape=[item_num])
    bias_item_a = tf.nn.embedding_lookup(embd_bias_item_a, item_a_batch, name='bias_item_a')
    bias_item_b = tf.nn.embedding_lookup(embd_bias_item_b, item_b_batch, name='bias_item_b')
    
    embd_item_a = tf.get_variable('embd_item_a', shape=[item_num, dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
    embd_item_b = tf.get_variable('embd_item_b', shape=[item_num, dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
    item_a = tf.nn.embedding_lookup(embd_item_a, item_a_batch, name='embedding_item_a')
    item_b = tf.nn.embedding_lookup(embd_item_b, item_b_batch, name='embedding_item_b')

  with tf.device(device):
    infer = tf.reduce_sum(tf.multiply(item_a, item_b), 1)
    infer = tf.add(infer, bias_global)
    infer = tf.add(infer, bias_item_a)
    infer = tf.add(infer, bias_item_b, name='svd_inference')

    regularizer = tf.add(tf.nn.l2_loss(item_a), tf.nn.l2_loss(item_b), name='svd_regularizer')

  return

def optimization_function():

  return


# FUNCTIONS OVER THE CSV DATASETS
city = 'Mississauga'

def getAllRatingsForItem(item, reviews_df=None):
  '''
      Returns all ratings for a certain item, in the format:
        { "user": rating, ... }
  '''

  # start = time.perf_counter()

  ratings = {}

  if reviews_df is None:
    reviews_df = pd.read_csv('../yelp_dataset/resources/'+city+'/reviews.csv')

  ratings_df = reviews_df[reviews_df['business'].str.contains(item)]
  for index, row in ratings_df.iterrows():
    ratings[row['user']] = row['rating']

  return ratings


def getAllRatingsForAllItems(city):
  '''
      Returns all ratings for all items, in the format:
        { "item": { "user": rating, ... }, ... }
  '''

  filename = '../yelp_dataset/resources/'+city+'/user_ratings_by_item.json'

  try:
    file = open(filename, encoding='utf8', mode='r')
  except IOError:
    saveAllRatingsForAllItems(city)
    file = open(filename, encoding='utf8', mode='r')
  finally:
    return json.load(file)

  return None


def saveAllRatingsForAllItems(city):
  '''
      Saves all ratings for all items into a json, in the format:
        { "item": { "user": rating, ... }, ... }
  '''

  all_ratings = {}
  counter = 0

  items_df = pd.read_csv('../yelp_dataset/resources/'+city+'/businesses.csv')
  reviews_df = pd.read_csv('../yelp_dataset/resources/'+city+'/reviews.csv')
  
  start = time.perf_counter()

  for index, row in items_df.iterrows():
    
    ratings = getAllRatingsForItem(row['business'], reviews_df)
    all_ratings[row['id']] = ratings

    counter += 1
    if counter % 500 == 0:
      print(counter)

  end = time.perf_counter()
  print(str(format((end-start)/60, '.3f')) + 'm')
  print(str(format((end-start)/60/3518, '.6f')) + 'm per')

  file = open('../yelp_dataset/resources/'+city+'/user_ratings_by_item.json', encoding='utf8', mode='w')
  json.dump(all_ratings, file, indent=3)
  file.close()

  print('finished')


# GENERAL UTILITY FUNCTIONS

def combineItemsToKey(a, b):
  return str(a) + ',' + str(b)


def getItemsFromKey(key):
  items = key.split(',')

  return items[0], items[1] 


def notRepeating(item_a, item_b, dictionary):
  return combineItemsToKey(item_a, item_b) not in dictionary


def clip_value(x):
  return np.clip(x, 1.0, 5.0)


def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])
