# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 21:10:07 2021

@author: Sebastian
"""

import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = np.array((12, 6.75)) * 0.75
# plt.rcParams["font.family"] = "sans"
plt.rcParams["axes.labelsize"] = "medium"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.linewidth"] = 1.3
plt.rcParams["lines.linewidth"] = 2.0

#transforms linear index k to grid indices
def lin2quad(k):
    return k % Ngrid,  k // Ngrid

def quad2lin(i, j):
    return Ngrid * j + i 

def tri2lin(i, j):
    return i*(i+1) / 2 + j

def lin2tri(k):
    rest = k
    i = 0
    j = 0
    while rest > i:
        rest -= i + 1
        j = rest
        i += 1
    return i, j

Ngrid = 50

lap = np.zeros((Ngrid**2, Ngrid**2))
v = np.zeros(Ngrid**2)

# grid coordinates
y, x = np.mgrid[10:0:-Ngrid*1j, 0:10:-Ngrid*1j]

left = np.ones(Ngrid)*100
right = np.ones(Ngrid)
upper = np.ones(Ngrid)*0#*np.sin(2*np.pi*x[0]/10)*100
lower = np.ones(Ngrid)

charges = np.zeros(Ngrid**2)
# charges[quad2lin(Ngrid//2+10, Ngrid//2)] = -100
# charges[quad2lin(Ngrid//2-10, Ngrid//2)] = +100

# array to set up dirichlet boundary conditions within the square
potential = np.zeros((Ngrid, Ngrid))
# potential[:,:] = np.diag(np.ones(Ngrid)*100)


for k in range( Ngrid**2):
    lap[k,k] = -4
    
    #grid indices
    i, j = lin2quad(k)
    
    if potential[i, j] != 0:
        continue
    
    # look up
    if i==0:
        v[k] = v[k] - upper[j]
    elif potential[i-1, j] != 0:
        v[k] -= potential[i-1, j]
    else:
        lap[k,quad2lin(i-1, j)] = 1.0

    # look left
    if j==0:
        v[k] = v[k] - left[i]
    elif potential[i, j-1] != 0:
        v[k] -= potential[i, j-1]
    else:
        lap[k,quad2lin(i, j-1)] = 1.0

    # look below
    if i==Ngrid-1:
        v[k] = v[k] - lower[j]
    elif potential[i+1, j] != 0:
        v[k] -= potential[i+1, j]
    else:
        lap[k,quad2lin(i+1, j)] = 1.0

    # look right
    if j==Ngrid-1:
        v[k] = v[k] - right[i]
    elif potential[i, j+1] != 0:
        v[k] -= potential[i, j+1]
    else:
        lap[k,quad2lin(i, j+1)] = 1.0
        
v += charges

grid = lg.solve(lap, v)
grid = np.reshape(grid, (Ngrid, Ngrid), order='F')
grid += potential

gradx, grady = np.gradient(grid, x[0,:], y[:,0])

# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (16,8))
fig, ax1 = plt.subplots(figsize = (16, 8))
im = ax1.pcolormesh(x, y, grid, shading='auto')
cb = plt.colorbar(im, ax=ax1)
cb.set_label('Potential V in Volt')

ax1.quiver(x, y, -gradx*100, -grady*100)


    