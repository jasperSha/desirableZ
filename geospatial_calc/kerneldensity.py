import matplotlib.pyplot as plt
import numpy as np
import math
import numpy as np
import sys
from numba import njit, jit

#display entire numpy array
np.set_printoptions(threshold=sys.maxsize)


#POINT DATASET
x=[16,16,16,15,15,28,15,18,25,15,18,25,30,25,22,30,22,38,40,38,30,22,20,35,33,35]
y=[50,49,48,45,40,14,15,15,20,32,33,20,20,20,25,30,38,20,28,33,50,48,40,30,35,36]

#convert to numpy arrays
x = np.asarray(x, dtype=np.float64)
y = np.asarray(y, dtype=np.float64)

#DEFINE GRID SIZE AND RADIUS(h)
#grid_size 3rd decimal for (longitudinal) area of a large field
grid_size=5
#h is our kernel radius to determine cluster influence
h=10

#GETTING X,Y MIN AND MAX
x_min=min(x)
x_max=max(x)
y_min=min(y)
y_max=max(y)

#CONSTRUCT GRID
x_grid=np.arange(x_min-h,x_max+h,grid_size)
y_grid=np.arange(y_min-h,y_max+h,grid_size)
x_mesh,y_mesh=np.meshgrid(x_grid,y_grid)

#GRID CENTER POINT

xc=x_mesh+(grid_size/2)

yc=y_mesh+(grid_size/2)

'''
Using the quartic kernel distribution for the density estimation.

The quartic kernel distribution function weighs near points more than far points, and
while the falloff is gradual, it is not as gradual as a normal distribution is;
thus the quartic function is determined to be more closely
aligned with the real life distribution of crime, e.g. high-crime areas that are
adjacent to more upscale neighborhoods with the dropoff being quite dramatic (at least
                                                                              more dramatic than what a
                                                                              normal distribution might
                                                                              follow.)
The quartic kernel is also less granular than a negative exponential or triangular
distribution, which have much more dramatic falloffs but produce finer variations;
due to computational/optimization concerns, the quartic also has the advantage of the
other two, as despite their greater accuracy, using those distributions for a dataset
of over a million points is impractical.

Quartic function equation:
dn = distance / h, where h is the chosen bandwidth
then,
P(x) = KWI*(15/16)*(1-dn**2)**2

The determination of bandwidth has the most impact on the resultant output however.

Initially adaptive kernel bandwidth determination was to be used, but as this
iteration of data analysis is limited by only having crime data from LA county,
and because that dataset is NOT sparsely populated, using a static kernel bandwidth 
will be optimal.

Using Silverman's Rule of Thumb bandwidth estimation formula, extrapolated to
two dimensions, we derive the formula for determining h as such:
    1. Calculate the mean center of input points, weighted by features (in this case
                                                                        by crime type)
    2. Calculate distance from mean center for all points
    3. Calculate median of all the distances
    4. Calculate Standard Distance (simply standard deviation of the distances) as SD
    
    5. Finally, h = 0.9 * min(SD, sqrt(1/ln(2))*Dm) * n**-0.2,
       where Dm is the median distance.
       
However, as there are multiple clusters in the LA county crime dataset, we'll use
apply this bandwidth estimation to each cluster (as determined before using our silhouette
                                                 analysis algorithm),
and get a different bandwidth for each cluster, and then essentially divide the
dataset into each cluster and apply the kernel density to each with their respective
bandwidths.

'''
# @jit(nopython=True)
def kde_quartic(d,h):
    #normalize distance(d) by dividing by radius length(h)
    dn=d/h
    
    P=(15/16)*(1-dn**2)**2
    return P

#PROCESSING
# @jit(nopython=True)
def generate_intensity(x, y, xc, yc):
    intensity_list=[]
    for j in range(len(xc)):
        intensity_row=[]
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
    return intensity_list

#HEATMAP OUTPUT
# for row in intensity_list:
#     print(row)
intensity_list = generate_intensity(x, y, xc, yc)
intensity=np.array(intensity_list)
print(intensity)
plt.pcolormesh(x_mesh,y_mesh,intensity)
plt.plot(x,y,'ro')
plt.colorbar()
plt.show()