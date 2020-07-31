import matplotlib.pyplot as plt
import numpy as np
import math
from random import random
from random import seed
from numba import cuda

FUNCTION TO CALCULATE INTENSITY WITH QUARTIC KERNEL
def kde_quartic(d,h):
    dn=d/h
    P=(15/16)*(1-dn**2)**2
    return P
seed(1)
x, y = [], []
for _ in range(10):
    value = random()*50
    x.append(int(value))
    
for _ in range(10):
    value = random()*50
    y.append(int(value))

print(x)
print(y)

grid_size = 1
h = 10

#GETTING X,Y MIN AND MAX
x_min=min(x)
x_max=max(x)
y_min=min(y)
y_max=max(y)

#CONSTRUCT GRID
x_grid=np.arange(x_min-h,x_max+h,grid_size)
y_grid=np.arange(y_min-h,y_max+h,grid_size)
x_mesh,y_mesh=np.meshgrid(x_grid,y_grid)

#GRID CENTER POINT MESH
xc=x_mesh+(grid_size/2)
yc=y_mesh+(grid_size/2)

#PROCESSING
intensity_list=[]
#j will iterate along the count of y-coordinates
for j in range(len(xc)):
    intensity_row=[]
    #k along the count of x-coords
    for k in range(len(xc[0])):
        kde_value_list=[]
        for i in range(len(x)):
            #CALCULATE DISTANCE
            d=math.sqrt((xc[j][k]-x[i])**2+(yc[j][k]-y[i])**2) 
            if d<=h:
                p=kde_quartic(d,h)
            else:
                p=0
            kde_value_list.append(p)
        #SUM ALL INTENSITY VALUE
        p_total=sum(kde_value_list)
        intensity_row.append(p_total)
    intensity_list.append(intensity_row)

# print(x_mesh)
# print(y_mesh)
print(xc)
print(xc[0])
print(xc[0][0])
print(len(xc))
print(len(xc[0]))

plt.plot(x,y,'ro')

@cuda.jit
def my_kernel(io_array):
    #thread id in a 1-D block
    tx = cuda.threadIdx.x
    
    #Block id in a 1-D grid
    ty = cuda.blockIdx.x
    
    #Block width (# of threads per block)
    bw = cuda.blockDim.x
    
    #Compute flattened index inside array
    pos = tx + ty * bw
    if pos < io_array.size: #Check bounds
        io_array[pos] *= 2 #Computation


if __name__ == '__main__':
    # Create the data array - usually initialized some other way
    data = np.ones(512)
    
    # Set the number of threads in a block
    threadsperblock = 32 
    
    # Calculate the number of thread blocks in the grid
    blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock
    
    # Now start the kernel
    my_kernel[blockspergrid, threadsperblock](data)
    
    # Print the result
    print(data)








