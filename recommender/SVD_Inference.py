import json, time, pprint, csv, Data_IO, Matrix
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from six import next
from collections import deque

'''
    Calculates the latent vectors for an incomplete item-item similarity matrix using tensorflow
    Based on a solution from https://github.com/songgc/TF-recomm
'''

np.random.seed(13575)

BATCH_SIZE = 1000
ITEM_NUM = 2846 # nr of items in the dataset
DIM = 20 # nr of latent features we want
EPOCH_MAX = 100
DEVICE = "/cpu:0"

errors_list = []
learning_rates_list = []

def get_data_df(filename):
  '''
      Opens the data .csv and returns as a dataframe
  '''
  return Data_IO.csv_to_df(filename)


def get_entire_data(df):
  '''
      Returns the entire DataFrame ready to be applied to the SVD calculations
  '''

  rows = len(df)
  df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)

  return Data_IO.OneEpochIterator([
    df['item_a'],
    df['item_b'],
    df['similarity']],
    batch_size=-1
    )


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
    inference = tf.add(inference, bias_global)
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

  return inference, regularizer, {'U': item_a, 'VT': item_b, 'bias_U': bias_item_a, 'bias_V': bias_item_b, 'bias': bias_global}


def optimization_function(inference, regularizer, similarity_batch, learning_rate, reg, device):
  global_step = tf.train.get_global_step()
  assert global_step is not None
  with tf.device(device):
    l2_loss_function = tf.nn.l2_loss(tf.subtract(inference, similarity_batch))
    l2_norm = tf.constant(reg, dtype=tf.float32, shape=[], name='l2')
    cost = tf.add(l2_loss_function, tf.multiply(regularizer, l2_norm))
    zeroes_penalty = tf.nn.l2_loss(tf.subtract(tf.abs(inference), inference))
    cost = tf.add(cost, zeroes_penalty)

    # Optimization done thru derivative calculation using Tensorflow's Adam Optimizer
    train_operation = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

  return cost, train_operation


def get_epoch_data(df):
  '''
      Shuffles the data and separates it into training and testing datasets
  '''

  df = df.sample(frac=1).reset_index(drop=True)
  rows = len(df)
  df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
  split_index = int(rows * 0.8)

  df_train = df[0:split_index]
  df_test = df[split_index:].reset_index(drop=True)

  # PREPARE BATCH DATA
  iter_train = Data_IO.ShuffleIterator([
    df_train['item_a'],
    df_train['item_b'],
    df_train['similarity']],
    batch_size=BATCH_SIZE
    )
  iter_test = Data_IO.OneEpochIterator([
    df_test['item_a'],
    df_test['item_b'],
    df_test['similarity']],
    batch_size=-1
    )
  samples_per_batch = len(df_train) // BATCH_SIZE

  return iter_train, iter_test, samples_per_batch


def SVD(data_df):
  '''
      Let S = {Sij} denote an m*m matrix of item similarity,
      where Sij denotes the similarity between user i and j based on ACOS and AMSD,
      and m is the total number of users.

      SVD Matrix factorization represents the item-item similarity matrix A with two low-rank matrices, U and V;
      and a diagonal matrix S, which describes the strength (bias) of each latent factor.
      Ui and Vj are the column vectors and indicate the k-dimensional latent feature vectors of item i and j.
      As such, the value for the predicted similarity A'ij would be the result of Ui*S*ij*Vj.T

      The optimization function can be defined as the sum of:
        - L2 Loss Function 
        - half the L2 Norm (Frobenius/Euclidian Norm) of the Latent Vectors, multiplied by a penalty lambda
      The L2 Loss Function simply represents the error between the predicted and the real values
      The L2 Norms are added as a regularisation term, in order to let the model generalise well and prevent overfitting
  
      Returns the item-item similarity matrix in the PredictionSVD class format
  '''

  # PREPARE INFERENCE ALGORITHM
  item_a_batch = tf.placeholder(tf.int32, shape=[None], name='id_item_a')
  item_b_batch = tf.placeholder(tf.int32, shape=[None], name='id_item_b')
  similarity_batch = tf.placeholder(tf.float32, shape=[None])

  base_learning_rate = np.float32(0.000001)
  learning_rate = tf.Variable(base_learning_rate, trainable=False)
  new_learning_rate = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
  update_learning_rate = tf.assign(learning_rate, new_learning_rate)
  
  inference, regularizer, prediction_matrix = inference_svd(item_a_batch, item_b_batch, item_num=ITEM_NUM, dim=DIM, device=DEVICE)
  tf.train.get_or_create_global_step() # create global_step for the optimizer
  _, train_operation = optimization_function(inference, regularizer, similarity_batch, learning_rate=learning_rate, reg=0.05, device=DEVICE)
  init_operation = tf.global_variables_initializer()

  # START TF SESSION
  with tf.Session() as sesh:
    sesh.run(init_operation)
    print("{}\t{}\t{}\t{}\t{}".format("epoch", "train_error", "val_error", "elapsed_time", "learning rate"))

    # Initialise the data for the first epoch
    iter_train, iter_test, samples_per_batch = get_epoch_data(data_df)
    errors = deque(maxlen=samples_per_batch)

    time_start = time.time()
    # TRAIN IN BATCHES
    for i in range(EPOCH_MAX * samples_per_batch):
      train_items_a, train_items_b, train_similarity_values = next(iter_train)
      _, train_pred_batch = sesh.run(
        [train_operation, inference],
        feed_dict={
          item_a_batch: train_items_a,
          item_b_batch: train_items_b,
          similarity_batch: train_similarity_values
        })
      errors.append(np.power(train_pred_batch-train_similarity_values, 2))

      # TEST AT THE END OF EACH EPOCH
      if i % samples_per_batch == 0:
        train_error = np.sqrt(np.mean(errors))
        test_error = np.array([])

        for test_items_a, test_items_b, test_similarity_values in iter_test:
          test_pred_batch = sesh.run(
            inference,
            feed_dict={
              item_a_batch: test_items_a,
              item_b_batch: test_items_b
            })
          test_error = np.append(test_error, np.power(test_pred_batch - test_similarity_values, 2))

        time_end = time.time()
        test_error = np.sqrt(np.mean(test_error))
        errors_list.append(test_error)
        epoch = 1 + i // samples_per_batch
        print("{:3d}\t{:f}\t{:f}\t{:0.4f}(s)\t{:0.0e}".format(1 + i // samples_per_batch, train_error, test_error, time_end - time_start, learning_rate.eval()))
        learning_rates_list.append(learning_rate.eval())

        # update learning rate
        '''
        if epoch % 25 == 0: # progressively increasing learning rate
          sesh.run(update_learning_rate, feed_dict={new_learning_rate: learning_rate.eval() * 2})
        
        if epoch == 70:
          sesh.run(update_learning_rate, feed_dict={new_learning_rate: learning_rate.eval() * 10})
          base_learning_rate = base_learning_rate * 10
        '''
        if epoch % 15 == 0: # cyclically increasing/decreasing learning rate
          if learning_rate.eval() <= base_learning_rate:
            sesh.run(update_learning_rate, feed_dict={new_learning_rate: learning_rate.eval() * 10})
          else:
            sesh.run(update_learning_rate, feed_dict={new_learning_rate: learning_rate.eval() / 10})
        
        # Generate new 80:20 of the dataset for the next epoch
        iter_train, iter_test, _ = get_epoch_data(data_df)
        time_start = time.time()

    # Generate the full predictions SVD matrix
    final_items_a = [i for i in range(ITEM_NUM)]
    final_items_b = [i for i in range(ITEM_NUM)]
    final_prediction = Matrix.PredictionSVD(sesh.run(
      prediction_matrix,
      feed_dict={
        item_a_batch: final_items_a,
        item_b_batch: final_items_b
      }))

    print(f'Validation error = {errors_list[-1]}')
    print(f'Mean similarity = {data_df["similarity"].mean()}')

    print(errors_list)

    return final_prediction

  return


def get_similarity_matrix():
  filename = './resources/sims/ACOS.csv'
  # filename = './resources/test.csv'
  data_df = get_data_df(filename)
  
  return SVD(data_df)


if __name__ == '__main__':
  filename = './resources/sims/COS_no1s.csv'
  #filename = './resources/sims/AMSD.csv'
  #filename = './resources/sims/ACOS.csv'

  data_df = get_data_df(filename)
  final_prediction = SVD(data_df)
  print("Training done!")
  #final_prediction.log()
  final_prediction.save()
