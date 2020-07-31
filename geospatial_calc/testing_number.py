from numba import cuda
from numba import jit
import numpy as np



# x=[16,16,16,15,15,28,15,18,25,15,18,25,30,25,22,30,22,38,40,38,30,22,20,35,33,35]
# y=[50,49,48,45,40,14,15,15,20,32,33,20,20,20,25,30,38,20,28,33,50,48,40,30,35,36]


# matrix = list(zip(x, y))

# a = np.asarray(matrix, dtype='int64')


# print(a)
# @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
# def go_fast(a): # Function is compiled to machine code when called the first time
#     trace = 0.0
#     for i in range(a.shape[0]):   # Numba likes loops
#         trace += np.tanh(a[i, i]) # Numba likes NumPy functions
#     return a + trace              # Numba likes NumPy broadcasting

# print(go_fast(a))
'''
Using Silverman's Rule of Thumb bandwidth estimation formula, extrapolated to
two dimensions, we derive the formula for determining h as such:
    1. Calculate the mean center of input points
    2. Calculate distance from mean center for all points
    3. Calculate median of all the distances (Dm)
    4. Calculate Standard Distance (simply standard deviation of the distances) as SD
    
    5. Finally, h = 0.9 * min(SD, sqrt(1/ln(2))*Dm) * n**-0.2,
       where Dm is the median distance.
       '''

import matplotlib.pyplot as plt
import mplleaflet
# from kerneldensity import kernelbandwidth

#POINT DATASET
x=[-118.45, -118.42, -118.35]
y=[33.98, 33.95, 33.94]
xy = np.vstack((x, y)).T
print(xy)
centroid = np.array((sum(x) / len(x), sum(y) / len(y)))
print(centroid)
# h = kernelbandwidth(xy, centroid)
# print(h)
plt.scatter(x,y,s=50, alpha=0.5)
# plt.show()
mplleaflet.show(tiles='cartodb_positron')
# print(xy.shape)
# print(xy[0])
# print(xy[0,][0], ' \n', xy[0,][1])



