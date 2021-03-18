from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def csv_to_df(filename):
    # col_names = ["item_a", "item_b", "similarity"]
    df = pd.read_csv(filename, engine='python')

    ''' NOTE
        There might be an issue with the trainer with regards to the item columns not being INTs
    '''

    # convert from float64 to float32 to speed up operations
    df["item_a"] = df["item_a"].astype(np.int32)
    df["item_b"] = df["item_b"].astype(np.int32)
    df["similarity"] = df["similarity"].astype(np.float32)

    return df


class ShuffleIterator(object):
    """
    Randomly generate batches
    """
    def __init__(self, inputs, batch_size=10):
        self.inputs = inputs
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.len = len(self.inputs[0])
        self.inputs = np.transpose(np.vstack([np.array(self.inputs[i]) for i in range(self.num_cols)]))

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ids = np.random.randint(0, self.len, (self.batch_size,))
        out = self.inputs[ids, :]

        return [out[:, i] for i in range(self.num_cols)]


class OneEpochIterator(ShuffleIterator):
    """
    Sequentially generate one-epoch batches, typically for test data
    """
    def __init__(self, inputs, batch_size=10):
        super(OneEpochIterator, self).__init__(inputs, batch_size=batch_size)
        if batch_size > 0:
            self.idx_group = np.array_split(np.arange(self.len), np.ceil(self.len / batch_size))
        else:
            self.idx_group = [np.arange(self.len)]
        self.group_id = 0

    def next(self):
        if self.group_id >= len(self.idx_group):
            self.group_id = 0
            raise StopIteration
        out = self.inputs[self.idx_group[self.group_id], :]
        self.group_id += 1

        return [out[:, i] for i in range(self.num_cols)]