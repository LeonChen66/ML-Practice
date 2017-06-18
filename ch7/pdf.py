from scipy.misc import comb
import math
import numpy as np
import matplotlib.pyplot as plt

def ensemble_error(n_classifier, error):
    k_start = math.ceil(n_classifier/2.0)
    probs = [comb(n_classifier,k)*error**k*(1-error)**(n_classifier-k) for k in range(k_start, n_classifier +1)]

    return sum(probs)

ensemble_error(n_classifier=11, error=0.25)
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]
plt.plot(error_range, ens_errors, label='Ensemble error', linewidth=2)
plt.plot(error_range, error_range, linestyle='--', label='Base error')
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid()
plt.show()

np.argmax(np.bincount([0,0,1],weights=[0.2,0.2,0.6]))
ex = np.array([[0.9,0.1],[0.8,0.2],[0.4,0.6]])
p = np.average(ex, axis=0, weights=[0.2,0.2,0.6])
print(p)
print(np.argmax(p))