# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 18:32:09 2021

@author: Sebastian
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

def triangle(x, maxpos=0.5, maxval=1):
    
    def up(x, maxpos, maxval):
        return maxval / maxpos * x
    
    def down(x, maxpos, maxval):
        return maxval - maxval / (1 - maxpos) * (x - maxpos)
    
    return np.piecewise(x, [x<maxpos, x>=maxpos], [up, down], 
                        maxpos, maxval)

# difference representing second derivative
def seconddiff(values):
    array = np.concatenate(([0], values, [0]))
    return np.array([array[i+1] - 2*array[i] + array[i-1] 
            for i in range(1,len(array)-1)] ) / dx**2
  
xsamples = 100
tsamples = 600
xmax = 2

c = 1.5 # m/s

dt = 5e-3 #s
xvals = np.linspace(0, xmax, xsamples+2)[1:-1]
tvals = np.array([i*dt for i in range(tsamples)])

dx = xvals[1] - xvals[0]

displacement = np.zeros((tsamples, xsamples))

#boundary conditions
displacement[0,:] = np.zeros(xvals.shape) # np.sin(np.pi*xvals)*0.3 #np.zeros(xvals.shape) 
velocity = -norm.pdf(xvals,1,0.1) #+ np.sin(1.5 * np.pi*xvals)
                                                 
displacement[1,:] = displacement[0,:] + dt * velocity


# for i in range(tsamples-2):
#     # update displacements
#     displacement[i+1,:] = velocity * dt + displacement[i]
    
#     # update velocities
#     velocity = velocity + seconddiff(displacement[i]) * dt * c**2
    

for i in range(1, tsamples-1):
    displacement[i+1] = c**2 * dt**2 * seconddiff(displacement[i]) + 2 * displacement[i,:] - displacement[i-1,:]
    
#set up animation
fig = plt.figure()
ax = plt.gca()
plt.xlim(0, xmax)
# plt.ylim(np.min(displacement), np.max(displacement))
plt.ylim(-1,1)
line, = ax.plot([], [])

def animate(i):
    line.set_data(xvals, displacement[i,:])
    
line_ani = animation.FuncAnimation(fig, animate, interval=1, 
                                    frames=tsamples)
# plt.plot(xvals, displacement[0])



    
