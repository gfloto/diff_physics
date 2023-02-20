import numpy as np

N = 16

for k in (1,15):
    for n in range(N):
        # exp -i 2 pi k n / N
        spin = np.exp(-2j*np.pi*k*n/N)
        print(k*n/N, spin)
    print()