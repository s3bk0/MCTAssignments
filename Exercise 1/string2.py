# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 20:52:13 2021

@author: Sebastian
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def seconddiff(values):
    array = np.concatenate(([0], values, [0]))
    return np.array([array[i+1] - 2*array[i] + array[i-1] 
            for i in range(1,len(array)-1)] ) / dx**2

xsamples = 100
tsamples = 1000

c = 100 # m/s

alld = np.zeros((tsamples+1, xsamples))
x = np.linspace(0, 1, xsamples+2)[1:-1]


dt = 0.00005
dx = x[1] - x[0]

t = np.arange(0, tsamples, dt)

last = np.sin(np.pi*x)
dudt = np.zeros(xsamples)

now = np.zeros(xsamples)
nxt = np.zeros(xsamples)

now[:] = dudt * dt + last

# for i in range(1, tsamples-1):
#     u[i+1] = c**2 * dt**2 * seconddiff(u[i]) + 2*u[i] - u[i-1]

for time in range(0,tsamples):
    print("time = ",time)
# plot "last" so that we first plot the t=0 displacement
    for i in range(1,xsamples+1):
        if i==1:            
            nxt[i-1] = c**2 * dt**2 / dx**2 * (-2*now[i-1]+now[i]) + \
            2*now[i-1]-last[i-1]
        elif i==xsamples:
            nxt[i-1] = c**2 * dt**2 / dx**2 * (now[i-2]-2*now[i-1]) + \
            2*now[i-1]-last[i-1]
        else:
            nxt[i-1] = c**2 * dt**2 / dx**2 * (now[i-2]-2*now[i-1]+now[i]) + \
            2*now[i-1]-last[i-1]
    
    last[:] = now
    now[:] = nxt
    alld[time+1,:] = last
    
fig, ax = plt.subplots()
ax.set(xlim=(0,1), ylim=(-2,2))
line, = ax.plot([], [])

alld = np.concatenate((np.zeros((alld.shape[0],1)), alld, np.zeros((alld.shape[0],1))), axis=1)

def animate(i):
    line.set_data(np.linspace(0,1,xsamples+2), alld[i,:])
    # ax.set_title(f'frame {i}')
    
anim = animation.FuncAnimation(fig, animate, interval=10, frames=tsamples)
plt.draw()
plt.show()
    


