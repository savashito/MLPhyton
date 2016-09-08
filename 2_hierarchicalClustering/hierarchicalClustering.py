from generateData import X
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

plt.scatter(X[:,0], X[:,1])
# plt.show()

# generate the linkage matrix
Z = linkage(X, 'ward')
print Z[10:]