# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:04:09 2019

@author: khershberger
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Setup a plot such that only the bottom spine is shown
def setup(ax):
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which='major', width=1.00)
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', width=0.75)
    ax.tick_params(which='minor', length=2.5)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.patch.set_alpha(0.0)


plt.figure(figsize=(8, 6))

# Fixed Locator
ax = plt.subplot(2, 1, 1)
setup(ax)
majors = vds_meas
ax.set_xlim((0,25))
ax.xaxis.set_major_locator(ticker.FixedLocator(majors))
plt.xticks(rotation=90)
#minors = np.linspace(0, 1, 11)[1:-1]
#ax.xaxis.set_minor_locator(ticker.FixedLocator(minors))
ax.text(0.0, 0.1, "Measured Vd values", fontsize=14,
        transform=ax.transAxes)

# Fixed Locator
ax = plt.subplot(2, 1, 2)
setup(ax)
majors = vgs_meas
ax.set_xlim((-5,8))
ax.xaxis.set_major_locator(ticker.FixedLocator(majors))
plt.xticks(rotation=90)
#minors = np.linspace(0, 1, 11)[1:-1]
#ax.xaxis.set_minor_locator(ticker.FixedLocator(minors))
ax.text(0.0, 0.1, "Measured Vg values", fontsize=14,
        transform=ax.transAxes)