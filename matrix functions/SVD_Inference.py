import Utils, json, time, pprint, csv, Data_IO
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from six import next
from collections import deque

'''
    Calculates the latent vectors for an incomplete item-item similarity matrix using tensorflow
    Based on https://github.com/songgc/TF-recomm
'''

np.random.seed(13575)

BATCH_SIZE = 1000
ITEM_NUM = 3518 # nr of items in the dataset
DIM = 15 # nr of latent features we want
EPOCH_MAX = 100
DEVICE = "/cpu:0"


def get_data():
  '''
      Opens the data .csv and separates it into training and testing datasets
      Changes the data's alphanumeric ids for integer ids
  '''
  items_df = pd.read_csv('../yelp_dataset/resources/Mississauga/businesses.csv', engine='python')
  ids_hash = {}
  for index, row in items_df.iterrows():
    ids_hash[row['business']] = index
  print('got new ids')

  df = Data_IO.csv_to_df('./resources/AMSD_similarity(L=9).csv')
  df = df.replace(ids_hash)
  print('replaced ids')

  rows = len(df)
  df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
  split_index = int(rows * 0.8)

  df_train = df[0:split_index]
  df_test = df[split_index:].reset_index(drop=True)

  return df_train, df_test, ids_hash


def SVD(train_data, test_data):
  '''
      Let S = {Sij} denote an m*m matrix of item similarity,
      where Sij denotes the similarity between user i and j based on ACOS and AMSD,
      and m is the total number of users.

      Matrix factorization represents the item-item similarity matrix S with two low-rank matrices, A and B.
      Ai and Bj are the column vectors and indicate the k-dimensional latent feature vectors of item i and j.
      As such, the value for the predicted similarity S'ij would be the result of Ai.T*Bj

      The optimization function can be defined as the sum of:
        - L2 Loss Function 
        - half the L2 Norm (Frobenius/Euclidian Norm) of the Latent Vectors, multiplied by a penalty lambda
      The L2 Loss Function simply represents the error between the predicted and the real values
      The L2 Norms are added as a regularisation term, in order to let the model generalise well and prevent overfitting
  '''

  # PREPARE BATCH DATA
  iter_train = Data_IO.ShuffleIterator([
    train_data['item_a'],
    train_data['item_b'],
    train_data['similarity']],
    batch_size=BATCH_SIZE
    )
  iter_test = Data_IO.OneEpochIterator([
    test_data['item_a'],
    test_data['item_b'],
    test_data['similarity']],
    batch_size=-1
    )
  samples_per_batch = len(train_data) // BATCH_SIZE
  item_a_batch = tf.placeholder(tf.int32, shape=[None], name='id_item_a')
  item_b_batch = tf.placeholder(tf.int32, shape=[None], name='id_item_b')
  similarity_batch = tf.placeholder(tf.float32, shape=[None])
  
  # PREPARE INFERENCE ALGORITHM
  infer, regularizer = Utils.inference_svd(item_a_batch, item_b_batch, item_num=ITEM_NUM, dim=DIM, device=DEVICE)
  tf.contrib.framework.get_or_create_global_step() # create global_step for the optimizer
  _, train_operation = Utils.optimization_function(infer, regularizer, similarity_batch, learning_rate=0.001, reg=0.05, device=DEVICE)
  init_operation = tf.global_variables_initializer()

  # START TF SESSION
  with tf.Session() as sesh:
    sesh.run(init_operation)
    summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sesh.graph)
    print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
    errors = deque(maxlen=samples_per_batch)
    start = time.time()

    # TRAIN IN BATCHES
    for i in range(EPOCH_MAX * samples_per_batch):
      print('training...')

      if i % samples_per_batch == 0:
        print('---TESTING---')


  return


'''
df_train, df_test = get_data()
SVD(df_train, df_test)
print("Done!")
'''

train, test = get_data()
print(test)