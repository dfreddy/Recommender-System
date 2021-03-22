import numpy as np

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
  def __init__(self, prediction):
    self.u, self.bias_u, self.bias_v, self.vt = prediction['U'], prediction['bias_U'], prediction['V'], prediction['bias_V']

  def get_value(self, x, y):
    return self.bias_u[x] + self.bias_v[y] + np.sum(np.multiply(self.u[x], self.v[y]))