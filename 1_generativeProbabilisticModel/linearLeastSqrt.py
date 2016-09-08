# -*- coding: utf-8 -*-
# http://arxiv.org/pdf/1008.4686v1.pdf
# find a better tutorial for this
from generateData import x,y,yerr,np,plt,m_true,b_true

A = np.vstack((np.ones_like(x), x)).T
C = np.diag(yerr * yerr)
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
y_ls = m_ls*x+b_ls
plt.plot(x,y_ls,'b--')

print("""Least-squares results:
	m = {0} ± {1} (truth: {2})
	b = {3} ± {4} (truth: {5})
""".format(m_ls, np.sqrt(cov[1, 1]), m_true, b_ls, np.sqrt(cov[0, 0]), b_true))
plt.show()