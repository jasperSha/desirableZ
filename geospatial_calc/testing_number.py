from numba import cuda
from numba import jit
import numpy as np



x=[16,16,16,15,15,28,15,18,25,15,18,25,30,25,22,30,22,38,40,38,30,22,20,35,33,35]
y=[50,49,48,45,40,14,15,15,20,32,33,20,20,20,25,30,38,20,28,33,50,48,40,30,35,36]


matrix = list(zip(x, y))

a = np.asarray(matrix, dtype='int64')


print(a)
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def go_fast(a): # Function is compiled to machine code when called the first time
    trace = 0.0
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace              # Numba likes NumPy broadcasting

print(go_fast(a))