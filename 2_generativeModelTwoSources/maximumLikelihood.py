import scipy.optimize as op
from generateData import x,y,yerr,np,plt,m_true,b_true,f_true

def lnlike(theta, x, y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

nll = lambda *args: -lnlike(*args)

# plt.show()
# result = op.minimize(nll, [m_true,b_true, np.log(f_true)], args=(x, y, yerr))

result = op.minimize(nll, [2, 8, np.log(1)], args=(x, y, yerr))
m_ml, b_ml, lnf_ml = result["x"]

print("""Maximum likelihood result:
	m = {0} (truth: {1})
	b = {2} (truth: {3})
	f = {4} (truth: {5})
""".format(m_ml, m_true, b_ml, b_true, np.exp(lnf_ml), f_true))
y_ml = m_ml*x+b_ml
plt.plot(x,y_ml,'b--')
# import pdb; pdb.set_trace()
# plt.show()
