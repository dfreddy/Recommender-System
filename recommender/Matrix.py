import datetime, os
import numpy as np
import pandas as pd

class Matrix(object):
  '''
      Stores a matrix as its SVD factorization matrices
      The matrix format must be as follow:
        [ [U1I1, U1I2, ...], ... ]
  '''

  def __init__(self, matrix):
    self.u, self.s, self.vt = np.linalg.svd(matrix, full_matrices=False)

  # returns the svd approximation of the original matrix
  def get_approx(self):
    return self.u @ np.diag(self.s) @ self.vt

  # returns the value for a certain u and v
  # eg. with a user-item rating matrix, get(user_id, item_id) would return the corresponding rating
  def get(self, x, y):
    return np.sum(self.u[x] * self.s * np.transpose(self.vt)[y])


class PredictionSVD(object):
  '''
      Stores the SVD prediction matrix as arrays of U and V latent features, and U and V biases (S)
  '''
  def __init__(self, prediction=None):

    if prediction != None:
      self.u, self.bias_u, self.vt, self.bias_v, self.bias = prediction['U'], prediction['bias_U'], prediction['VT'], prediction['bias_V'], prediction['bias']
      self.model_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

  def get(self, x, y):
    x = int(x)
    y = int(y)
    return np.float32(self.bias + self.bias_u[x] + self.bias_v[y] + np.sum(np.multiply(self.u[x], self.vt[y])))

  def save(self):
    folder_name = f'./resources/{self.model_id}'
    os.mkdir(folder_name)

    np.save(f'{folder_name}/U', self.u)
    np.save(f'{folder_name}/V', self.vt)
    np.save(f'{folder_name}/BIAS_U', self.bias_u)
    np.save(f'{folder_name}/BIAS_V', self.bias_v)
    np.save(f'{folder_name}/BIAS', self.bias)
    print(f'Saved model: {self.model_id}')

  def log(self):
    print(f'model id: {self.model_id}')
    print('U:')
    print(self.u)
    print('V:')
    print(self.vt)

  def load(self, model_id):
    '''
        Loads model from memory
        model_id input can be string or int

        TODO
        verify if model actually exists in memory
    '''

    self.model_id = model_id
    self.u = np.load(f'./resources/{str(model_id)}/U.npy')
    self.vt = np.load(f'./resources/{str(model_id)}/V.npy')
    self.bias_u = np.load(f'./resources/{str(model_id)}/BIAS_U.npy')
    self.bias_v = np.load(f'./resources/{str(model_id)}/BIAS_V.npy')
    self.bias = np.load(f'./resources/{str(model_id)}/BIAS.npy')
    print('Loaded!')
