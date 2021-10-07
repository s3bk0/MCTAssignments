# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 21:13:42 2021

@author: Sebastian
"""
# sample solution to week 1 problem in Modern Computational Techniques 2020

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# set up plot
fig, ax = plt.subplots(figsize=(5, 3))
ax.set(xlim=(0, 1), ylim=(-0.15, 0.15))

# number of unknowns in x-grid & x-spacing
npts = 100 ; dx = 1.0/(npts+1) 

# pick time spacing (arbitrary but must be commensurate with dx)
dt = 0.005 ; nt = 400

# and speed
c = 1.0

# set up three arrays which represent the displacement at "now" "last time"
# and "next time".

x = np.empty(npts)
now = np.zeros(npts)
last = np.zeros(npts)
nxt = np.zeros(npts)

# I think to get animation working I need to save all the data:
alld = np.empty((npts,nt+2))

# as initial condition, let's have a gaussian shape centred on position x=0.3
# (our endpoints are defined to be at x=0 and x=1 with displacement fixed at
# u=0).  We will stick in some offset, width and amplitude constants for the
# gaussian since it's our initial condition to choose

# in my way of thinking point 0 is at x=0 and point npts+1 is at x=1

for i in range(1,npts+1):
    last[i-1] = 0.1*math.exp(-((dx*i-0.3))**2/0.01)
    x[i-1]=i*dx
    
x = np.linspace(0, 1, npts+2)[1:-1]
last = np.sin(np.pi*x)

alld[:,0] = last

# set the "now" displacement vector to be based on du/dt.
# in the simplest case, this can be 0 (for a pure plucked string)

dudt = 0.0
now[:] = dudt*dt + last
alld[:,0] = now

# now having set up we can apply the derived equation over many loops

# plot first line
line, = ax.plot(x,last)

for time in range(0,nt):
    print("time = ",time)
# plot "last" so that we first plot the t=0 displacement
    for i in range(1,npts+1):
        if i==1:            
            nxt[i-1] = c**2 * dt**2 / dx**2 * (-2*now[i-1]+now[i]) + \
            2*now[i-1]-last[i-1]
        elif i==npts:
            nxt[i-1] = c**2 * dt**2 / dx**2 * (now[i-2]-2*now[i-1]) + \
            2*now[i-1]-last[i-1]
        else:
            nxt[i-1] = c**2 * dt**2 / dx**2 * (now[i-2]-2*now[i-1]+now[i]) + \
            2*now[i-1]-last[i-1]

    # done one time step, so discard old "last" vector, and put "now" in there
    # then overwrite "now" with nxt
    last[:] = now
    now[:] = nxt
    alld[:,time+1]=last

# t = np.linspace(0,10, nt)
# alld = np.sin(np.pi*(x[:,None]-t[None, :]))

# do animaiton of results
def animate(i):
    line.set_ydata(alld[:,i])
    ax.set_title('frame '+str(i))

anim = FuncAnimation(fig,animate,interval=20,frames=nt)
plt.draw()
plt.show()
