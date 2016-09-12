# import numpy as np
from generateData import m_true,f_true,b_true,np,plt,x,y,yerr
from maximumLikelihood import lnlike
from sklearn import mixture
import corner

f = open('MCMC_samples', 'r')
truths=[m_true, b_true, np.log(f_true)]
X = np.load(f)

dpgmm = mixture.DPGMM(n_components=5, covariance_type='full')
dpgmm.fit(X)
means = dpgmm.means_
Y_ = dpgmm.predict(X)
# np.any(Y_ == i):

# print means[2]
# print truths
# lnlike(means[3], x, y, yerr)
# select the max log likelihood
# lnlike(truths, x, y, yerr)
m3,b3,f3 = means[3]
y3 = m3*x+b3
y_true = m_true*x+b_true
plt.errorbar(x,y,yerr=yerr, fmt='o')
plt.plot(x,y3,'r--')
plt.plot(x,y_true,'r')
plt.figure()
fig = corner.corner(X, labels=["$m$", "$b$", "$\ln\,f$"],truths=[m_true, b_true, np.log(f_true)])
plt.show()
# import pdb; pdb.set_trace()
