from kerneldensity import centroidnp, kernelbandwidth
from silhouette import full_crime_compile
from pointdensity import * 
from shapely.wkt import loads
import time
import matplotlib.pyplot as plt
import pandas as pd

#pre-compiled crime
os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/geospatial_calc/')
crime_df = pd.read_csv('full_crime_df.csv')
crime_df['geometry'] = crime_df['geometry'].apply(loads)
crime_coords = np.array(list(crime_df.geometry.apply(lambda x: (x.x, x.y))))


centroid = centroidnp(crime_coords)

h = kernelbandwidth(crime_coords, centroid)
radius = latToKm(h)

radius = np.round(radius, 3)


density_grid = point_density(events=crime_df, coords=crime_coords, radius=radius, gridsize=0.1)

#using log normal for skewed data
# density_grid[:,2] = np.log(density_grid[:,2])


# #(still using original density_grid)
# x, y = density_grid[:,1], density_grid[:,0]
# f, ax = plt.subplots()
# points = ax.scatter(x, y, s=0.1, c=density_grid[:,2])
# f.colorbar(points)



#write density to csv
os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/geospatial_calc')
cols = ['latitude', 'longitude', 'density', 'dateavg']
df = pd.DataFrame(density_grid, columns=cols)
df.to_csv('crime_density_rh_gridsize_1.csv', index=False)

