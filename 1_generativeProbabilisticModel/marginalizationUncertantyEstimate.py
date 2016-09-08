from maximumLikelihood import lnlike,result
from generateData import x,y,yerr,m_true,f_true,b_true,np,plt
from matplotlib.ticker import MaxNLocator
import emcee

def lnprior(theta):
	m, b, lnf = theta
	if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
		return 0.0
	return -np.inf
	

def lnprob(theta, x, y, yerr):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, x, y, yerr)

ndim, nwalkers = 3, 100
pos = [ result["x"] + 3*np.random.randn(ndim) for i in range(nwalkers)]
# pos = [ result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
print("Running MCMC...")
sampler.run_mcmc(pos, 5000)
print("Done.")

plt.clf()
plt.close()
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].axhline(m_true, color="#888888", lw=2)
axes[0].set_ylabel("$m$")

axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].axhline(b_true, color="#888888", lw=2)
axes[1].set_ylabel("$b$")

axes[2].plot(np.exp(sampler.chain[:, :, 2]).T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].axhline(f_true, color="#888888", lw=2)
axes[2].set_ylabel("$f$")
axes[2].set_xlabel("step number")

fig.tight_layout(h_pad=0.0)

import corner
# Make the triangle plot.
burnin = 1000
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"],
					  truths=[m_true, b_true, np.log(f_true)])
fig.savefig("line-triangle.png")

# Plot some samples onto the data.
plt.figure()
xl = np.array([0, 10])
for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
	plt.plot(xl, m*xl+b, color="k", alpha=0.1)
plt.plot(xl, m_true*xl+b_true, color="r", lw=2, alpha=0.8)
plt.errorbar(x, y, yerr=yerr, fmt=".k")
plt.ylim(-9, 9)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.tight_layout()
plt.savefig("line-mcmc.png")

# Compute the quantiles.
samples[:, 2] = np.exp(samples[:, 2])
m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
							 zip(*np.percentile(samples, [16, 50, 84],
												axis=0)))
print("""MCMC result:
	m = {0[0]} +{0[1]} -{0[2]} (truth: {1})
	b = {2[0]} +{2[1]} -{2[2]} (truth: {3})
	f = {4[0]} +{4[1]} -{4[2]} (truth: {5})
""".format(m_mcmc, m_true, b_mcmc, b_true, f_mcmc, f_true))
plt.show()