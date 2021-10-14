# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:59:53 2021

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
    return int(i*(i+1) / 2 + j)

def lin2tri(k):
    rest = k
    i = 0
    j = 0
    while rest > i:
        rest -= i + 1
        j = rest
        i += 1
    return i, j

Ngrid = 100

#number of elements
Nel = tri2lin(Ngrid-1, Ngrid-1) + 1

lap = np.zeros((Nel, Nel))
v = np.zeros(Nel)

# grid coordinates
y, x = np.mgrid[10:0:-Ngrid*1j, 0:10:-Ngrid*1j]
potential = np.ones((Ngrid, Ngrid))*np.NAN

left = -np.ones(Ngrid)*100
diagonal = np.ones(Ngrid+1) *100
lower = np.ones(Ngrid)*0

charges = np.zeros(Nel)
charges[tri2lin(Ngrid//2+10, Ngrid//2)] = -100
charges[tri2lin(Ngrid//2-10, Ngrid//2)] = +100

for k in range( Nel):
    lap[k,k] = -4
    
    #grid indices
    i, j = lin2tri(k)
    
    # look up and right
    if i==j:
        v[k] = v[k] - diagonal[i]
        v[k] = v[k] - diagonal[i+1]
    else:
        lap[k,tri2lin(i-1, j)] = 1.0
        lap[k,tri2lin(i, j+1)] = 1.0

    # look left
    if j==0:
        v[k] = v[k] - left[i]
    else:
        lap[k,tri2lin(i, j-1)] = 1.0

    # look below
    if i==Ngrid-1:
        v[k] = v[k] - lower[j]
    else:
        lap[k,tri2lin(i+1, j)] = 1.0

        
v += charges
grid = lg.solve(lap, v)

for k, gval in enumerate(grid):
    potential[lin2tri(k)] = gval



fig, ax = plt.subplots(figsize = (10,8))
im = ax.pcolormesh(x, y, potential, shading='auto')
cb = plt.colorbar(im, ax=ax)
cb.set_label('Potential V in Volt')

plt.show()
    