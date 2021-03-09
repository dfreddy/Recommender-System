import Utils, json, time, pprint, csv, Data_IO
import pandas as pd
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
USER_NUM = 6040
ITEM_NUM = 3952
DIM = 15
EPOCH_MAX = 100
DEVICE = "/cpu:0"


def get_data():
  '''
      Opens the data .csv and separates it into training and testing datasets
  '''
  df = Data_IO.csv_to_df('./resources/AMSD_similarity(L=9).csv')

  rows = len(df)
  df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
  split_index = int(rows * 0.8)

  df_train = df[0:split_index]
  df_test = df[split_index:].reset_index(drop=True)

  return df_train, df_test


def svd(train_data, test_data):


  return

'''
df_train, df_test = get_data()
svd(df_train, df_test)
print("Done!")
'''