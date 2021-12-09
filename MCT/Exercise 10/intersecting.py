import numpy as np
import math

samples = 100000
cnt = 0

# Overlap of two spheres:  radius of larger one is bigr, radius of smaller one
# is r.  Distance between them is d
bigr = 1.0
d = 1.1
r = 0.6

# calculating a better frame from intersections
hcap = (r-bigr+d)*(r+bigr-d) / (2*d)
intersect = np.sqrt(hcap*(2*bigr-hcap))

# set up integration ranges in x, y, z:
xmin = d-r
xmax = bigr
ymax = intersect
zmax = intersect
ymin = -ymax
zmin = -zmax

# loop over samples
for i in range(0,samples):
    rxyz = np.random.random(3)

    # scale the samples
    rxyz[0] = rxyz[0]*(xmax-xmin) + xmin
    rxyz[1] = rxyz[1]*(ymax-ymin) + ymin
    rxyz[2] = rxyz[2]*(zmax-zmin) + zmin

    if(sum([a*a for a in rxyz])<bigr**2):
        if((rxyz[0]-d)**2 + rxyz[1]**2+rxyz[2]**2 < r**2):
            cnt=cnt+1

volume = cnt/samples*(xmax-xmin)*(ymax-ymin)*(zmax-zmin)

exact = math.pi * (bigr+r-d)**2 * (d**2 + 2*d*r -3*r*r + 2*d*bigr + 6*r*bigr - 3*bigr**2)/(12*d)

print(volume, exact)

