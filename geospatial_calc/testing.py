# from kerneldensity import centroidnp, kernelbandwidth
from pointdensity import * 
from shapely.wkt import loads
import time
import matplotlib.pyplot as plt
from colour import Color
red = Color("red")
colors = list(red.range_to(Color("green"),13753))


os.chdir('/home/jaspersha/Projects/HeatMap/desirableZ/geospatial_calc/')
checking_crime_df = pd.read_csv('full_crime_df')


crime_df['geometry'] = crime_df['geometry'].apply(loads)
crime_coords = np.array(list(crime_df.geometry.apply(lambda x: (x.x, x.y))))


centroid = centroidnp(crime_coords)

h = kernelbandwidth(crime_coords, centroid)
radius = latToKm(h)

radius = np.round(radius, 3)

radius = 2.25
start = time.time()
density_grid = point_density(events=crime_df, coords=crime_coords, radius=radius, gridsize=0.2)
end = time.time()
print('gridsize = 0.1 runtime for pointdensity: ', end - start)

_, idx, counts = np.unique(density_grid[:,2], axis=0, return_index=True, return_counts=True)

densities = density_grid[idx]

x, y = densities[:,1], densities[:,0]
f, ax = plt.subplots()
points = ax.scatter(x, y, s=1, c=densities[:,2])
f.colorbar(points)