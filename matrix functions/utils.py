import json, pprint, time
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tensorflow.core.framework import summary_pb2


# SVD INFERENCE FUNCTIONS

def inference_svd(item_a_batch, item_b_batch, item_num, dim, device):
  with tf.device('/cpu:0'):
    # get biases for the items in the batch
    bias_global = tf.get_variable('bias_global', shape=[])
    embd_bias_item_a = tf.get_variable("embd_bias_item_a", shape=[item_num])
    embd_bias_item_b = tf.get_variable("embd_bias_item_b", shape=[item_num])
    bias_item_a = tf.nn.embedding_lookup(embd_bias_item_a, item_a_batch, name='bias_item_a')
    bias_item_b = tf.nn.embedding_lookup(embd_bias_item_b, item_b_batch, name='bias_item_b')
    
    # get latent values for the items in the batch 
    embd_item_a = tf.get_variable('embd_item_a', shape=[item_num, dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
    embd_item_b = tf.get_variable('embd_item_b', shape=[item_num, dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
    item_a = tf.nn.embedding_lookup(embd_item_a, item_a_batch, name='embedding_item_a')
    item_b = tf.nn.embedding_lookup(embd_item_b, item_b_batch, name='embedding_item_b')

  with tf.device(device):
    # SVD U*S*V calculation
    inference = tf.reduce_sum(tf.multiply(item_a, item_b), 1)
    # inference = tf.add(inference, bias_global)
    inference = tf.add(inference, bias_item_a)
    inference = tf.add(inference, bias_item_b, name='svd_inference')

    '''
    prediction_matrix = tf.matmul(item_a, item_b, transpose_b=True)
    prediction_matrix = tf.add(prediction_matrix, bias_global)
    prediction_matrix = tf.add(prediction_matrix, bias_item_a)
    prediction_matrix = tf.add(prediction_matrix, bias_item_b, name='prediction_matrix')
    '''

    # L2 Norm
    regularizer = tf.add(tf.nn.l2_loss(item_a), tf.nn.l2_loss(item_b), name='svd_regularizer')

  return inference, regularizer, {'U': item_a, 'V': item_b, 'bias_U': bias_item_a, 'bias_V': bias_item_b}

def optimization_function(inference, regularizer, similarity_batch, learning_rate, reg, device):
  global_step = tf.train.get_global_step()
  assert global_step is not None
  with tf.device(device):
    l2_loss_function = tf.nn.l2_loss(tf.subtract(inference, similarity_batch))
    l2_norm = tf.constant(reg, dtype=tf.float32, shape=[], name='l2')
    cost = tf.add(l2_loss_function, tf.multiply(regularizer, l2_norm))
    
    # Optimization done thru derivative calculation using Tensorflow's Adam Optimizer
    train_operation = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

  return cost, train_operation


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

def clip(x):
    return np.clip(x, 0.0, 1.0)


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
