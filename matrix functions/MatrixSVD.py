import numpy as np

class MatrixSVD(object):
  '''
      Stores a matrix as its SVD factorization matrices
  '''

  def __init__(self, matrix):
    self.u, self.s, self.vt = np.linalg.svd(matrix, full_matrices=False)

  # returns the svd approximation of the original matrix
  def get_approx(self):
    return self.u @ np.diag(self.s) @ self.vt

  # returns the value for a certain u and v
  # eg. with a user-item rating matrix, get(user_id, item_id) would return the corresponding rating
  def get(self, x, y)
    return np.sum(self.u[x] * self.s * np.transpose(self.vt)[y])