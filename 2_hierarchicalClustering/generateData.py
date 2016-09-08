
import numpy as np

# suppress scientific float notation
np.set_printoptions(precision=5, suppress=True)  

np.random.seed(4711)  
# generate two clusters: a with 100 points, b with 50:
a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
X = np.concatenate((a, b),)
#print X.shape  # 150 samples with 2 dimensions
