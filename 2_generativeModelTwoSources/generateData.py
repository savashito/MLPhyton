import matplotlib.pyplot as plt
import numpy as np

# np.random.seed(1)
np.random.seed(123)
# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534
m2_true = 0.9594
b2_true = -5
plt.figure()

# Generate some synthetic data from the model.
N = 50
x = np.sort(10*np.random.rand(N))
yerr = 0.1+0.5*np.random.rand(N)
y = m_true*x+b_true
y2 = m2_true*x+b2_true

plt.plot(x,y)
plt.plot(x,y2)
y += np.abs(f_true*y) * np.random.randn(N)
# error reading the data
# y += np.random.randn(N)
# data is also stochastic but has a mysterious bias
y += yerr * np.random.randn(N)
y2 += yerr * np.random.randn(N)
y = np.hstack([y,y2])
x = np.hstack([x,x])
yerr=np.hstack([yerr,yerr])
print y.shape,x.shape
plt.errorbar(x,y,yerr=yerr, fmt='o')
# plt.errorbar(x,y2,yerr=yerr, fmt='o')
# plt.show()