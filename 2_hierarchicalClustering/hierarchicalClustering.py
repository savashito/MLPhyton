from generateData import X,np
from matplotlib import pyplot as plt
# from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import mixture
import itertools

plt.scatter(X[:,0], X[:,1])

dpgmm = mixture.DPGMM(n_components=5, covariance_type='full')
dpgmm.fit(X)
Y_ = dpgmm.predict(X)
means = dpgmm.means_
color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
for i, (mean,color) in enumerate(zip(means,color_iter)):
	if not np.any(Y_ == i):
		continue
	plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
	print mean
plt.show()



# for i, (clf, title) in enumerate([(gmm, 'GMM'),
# 								  (dpgmm, 'Dirichlet Process GMM')]):



# generate the linkage matrix
# Z = linkage(X, 'ward')
# print Z[10:]