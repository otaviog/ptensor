import matplotlib.pyplot as plt
import numpy as np

def to_cpx(x):
    return x[..., 0] + 1j * x[..., 1]


npz = np.load('test.npz')
sines = npz['sines']
print(sines.shape)
# import ipdb; ipdb.set_trace()
freq = npz['freq']
S0 = to_cpx(freq[0])

rec = npz['rec']
import ipdb; ipdb.set_trace()
# plt.plot(data[0, :128])
# plt.show()

