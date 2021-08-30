# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:15:20 2019

@author: kyleh
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget

logger = logging.getLogger(__name__)

def toClipboard(fig):
  QApplication.clipboard().setImage(QWidget.grab(fig.canvas).toImage())
  print('figure %d copied to clipboard' % fig.number)

def to_clipboard(fig):
    logger.warning('to_clipboard() deprecated.  Use toClipboard instead.')
    toClipboard(fig)

def plotWithMultiX(table, **kwargs):
    labelsize = kwargs.pop('labelsize', None)
    ax = table.plot(**kwargs)
    # This is a ghetto hack to detect if a numeric axis was plotted

    #if ax.get_xticks()[0] == 0:
    if len(ax.get_xlabel().split(','))>1:
        ax.set_xticks(range(len(table)))
        # Converting to array here is done to force more sane precision
        labels = [str(np.array(item)) for item in table.index.tolist()]
        ax.set_xticklabels(labels, rotation=90)
        if labelsize is None:
            labelsizeh = round(144.0/len(table))
            labelsizev = round(144.0/ max(list(map(len, labels))))
            labelsize = min([labelsizeh, labelsizev])
        ax.tick_params(axis='x', labelsize=labelsize)

    ax.grid(True)
    ax.get_figure().tight_layout()
    return ax

def calculateTextProp(ax, tickLabels):
    """Calculates appropriate text/tick parameters based on size"""
    # Get labels for inner most index
    axesWidth = ax.get_window_extent().width
    nTicks = len(tickLabels)
    
    maxTickLabelLength = np.amax([ len(str(tmp)) for tmp in tickLabels])
    logger.debug('nticks = {:d}'.format(nTicks))
    logger.debug('max label length = {:d}'.format(maxTickLabelLength))
    
    # Check to see if these tick labels will overlap
    tickCharacterWidth = axesWidth / (nTicks * maxTickLabelLength)
    if tickCharacterWidth < 10:
        # Too many ticks, we'll rotate by 90 and scale font
        tickRotation = 90
        # Compute font size to keep vertical size down
        tickFontSize = 50 / maxTickLabelLength
        tickFontSizeMax = min(12, (axesWidth / nTicks) - 5)
        if tickFontSize > tickFontSizeMax:
            tickFontSize = tickFontSizeMax
    else:
        tickRotation = 0
        tickFontSize = 12

    logger.debug('tickRotation = ' + str(tickRotation))
    logger.debug('tickFontSize = ' + str(tickFontSize))
  
    # Get actual size of labels
    # This actually draws a text object, measures it size, and then removes it
    t = plt.text(0.5,0.5, maxTickLabelLength*'0', size=tickFontSize)
    r = ax.get_figure().canvas.get_renderer()
    bb = t.get_window_extent(renderer=r)
    t.remove()
    
    # Increment position counter accordingly
    if tickRotation == 90:
        positionIncrement = bb.width + 2
    else:
        positionIncrement = bb.height + 2
  
    return tickRotation, tickFontSize, positionIncrement


def createAxisTicks(mindex, level):
    # Compute Major & Minor tick locations for given index level
    # Also creates tick text labels
    ticksMajor = [0]
    ticksMinor = []

    lastLabel = mindex[0][level]
    lastLabelPosition = 0
    
    tickLabels = [lastLabel]
    
    for k, index in enumerate(mindex):
        #print('{:d}: {:s} == {:s}'.format(k,item[level],xLast))

        label = index[level]
        if label != lastLabel:
            ticksMajor.append(k-0.5)
            ticksMinor.append( (k-0.5+lastLabelPosition)/2.0 )
            tickLabels.append( label )

            lastLabel = label
            lastLabelPosition = k-0.5

    ticksMajor.append(k)
    ticksMinor.append( (k + lastLabelPosition) / 2.0 )

    return ticksMajor, ticksMinor, tickLabels


def fixMultiAxisLabel(ax, xindex):
    """ Fixes axis label for multi-parameter axes"""

    axesWidth = ax.get_window_extent().width
    axesHeight = ax.get_window_extent().height
    logger.debug('axes dim = {:g} x {:g}'.format(axesWidth,axesHeight))

    naxis = xindex.nlevels

    # Get labels for inner most index
    # This will be displayed suing the built in axes
    tickLabels = [str(item[naxis-1]) for item in xindex.tolist()]
    
    tickRotation, tickFontSize, positionIncrement = calculateTextProp(ax, tickLabels)

    ax.set_xticks(range(len(tickLabels)))
    ax.set_xticklabels(tickLabels, 
                    rotation=tickRotation, 
                    fontsize=tickFontSize)
    
    position = positionIncrement
    
    def setupXaxis(ax, position):
        """What does this do?"""
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('outward', position))
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
            ax.spines["bottom"].set_visible(False)

    # Now we handle the remaining index levels
    for k in reversed(range(naxis-1)):
        logger.debug('axis level = {:s}'.format(xindex.names[k]))
        logger.debug('position = ' + str(position))
        ax2 = ax.twiny()            # Create new x-axis
        setupXaxis(ax2, position)   # Setup x-axis properties
        tmaj,tmin,tlab = createAxisTicks(xindex.tolist(), k)

        tickRotation, tickFontSize, positionIncrement = calculateTextProp(ax, tlab)

        ax2.set_xticks(tmaj)
        ax2.set_xticklabels('')
        ax2.set_xticks(tmin, minor=True)
        ax2.set_xticklabels(tlab, minor=True,
                            rotation=tickRotation, 
                            fontsize=tickFontSize)
        ax2.tick_params(axis='x', which='minor', length = 0)
        ax2.tick_params(axis='x', which='major', length = 10)
        position = position + positionIncrement

    # Convert out pixel position to axes coordinates
    position = position + 30
    positionAxes = position / axesHeight
    logger.debug('position = ' + str(position))
    logger.debug('positionAxes = ' + str(positionAxes))
    
    ax.xaxis.set_label_coords(0.5, -positionAxes)

def plotMulti2(pt,
              valueField, 
              title="Default Title",
              ylabel="Y-Axis",
              ylim=None,
              ynticks=10,
              cycler=None,
              savepng=False):
    """Creates a plot with a multi-parameter X-axis
    
    pt -- DataFrame/ PivotTable to plot
    valueField -- String or (String, aggfunc)"""
    
    logger.debug('plotMulti(): {:s}'.format(title))
    logger.info('Plotting {:s}'.format(title))

    def setupXaxis(ax, position):
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('outward', position))
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
            ax.spines["bottom"].set_visible(False)


    logger.debug('plotMulti(): ' + title)      
    
    if not isinstance(valueField, tuple):
        valueField = (valueField, 'mean')
    
    # Setup the plotting
    fig = plt.figure(figsize=(8,10), dpi=80, facecolor='w', edgecolor='k')
    fig.canvas.set_window_title(title)
    #ax1 = fig.add_subplot(1,1,1)
    ax1 = fig.add_axes([0.1,0.3, 0.8, 0.5])
    ax1.set_title(title)

    if cycler is not None:
        ax1.set_prop_cycle(cycler)
    
    maxAggregation = np.max(pt[(valueField[0], 'count')])
    maxAggregation = 0

    indexFields   = pt.index.names
    legendFields2 = pt[[valueField]].columns.names  # Double brackets to make sure we get a Dataframe not a Series
    if len(legendFields2) == 1:
        legendFields2 = [valueField]
        
    #dfToPlot = pt[valueField].unstack(indexFields)  # Not needed
    dfToPlot = pt[[valueField]]
    
    dfToPlot.plot(ax=ax1)
    dfToPlot = dfToPlot
    

    axesWidth = ax1.get_window_extent().width
    axesHeight = ax1.get_window_extent().height
    logger.debug('axes dim = {:g} x {:g}'.format(axesWidth,axesHeight))
    naxis = len(axisField)

    # Get labels for inner most index
    tickLabels = [str(item[naxis-1]) for item in dfToPlot.index.tolist()]
    
    tickRotation, tickFontSize, positionIncrement = calculateTextProp(fig, axesWidth, tickLabels)

    ax1.set_xticks(range(len(dfToPlot)))
    ax1.set_xticklabels(tickLabels, 
                    rotation=tickRotation, 
                    fontsize=tickFontSize)
    
    position = positionIncrement

    # Now we handle the remaining index levels
    for k in reversed(range(naxis-1)):
        logger.debug('axis level = {:s}'.format(dfToPlot.index.names[k]))
        logger.debug('position = ' + str(position))
        ax2 = ax1.twiny()
        setupXaxis(ax2, position)
        tmaj,tmin,tlab = createAxisTicks(dfToPlot.index.tolist(), k)

        tickRotation, tickFontSize, positionIncrement = calculateTextProp(fig, axesWidth, tlab)

        ax2.set_xticks(tmaj)
        ax2.set_xticklabels('')
        ax2.set_xticks(tmin, minor=True)
        ax2.set_xticklabels(tlab, minor=True,
                            rotation=tickRotation, 
                            fontsize=tickFontSize)
        ax2.tick_params(axis='x', which='minor', length = 0)
        ax2.tick_params(axis='x', which='major', length = 10)
        position = position + positionIncrement

    # Convert out pixel position to axes coordinates
    position = position + 20
    positionAxes = position / axesHeight
    logger.debug('position = ' + str(position))
    logger.debug('positionAxes = ' + str(positionAxes))
    
    ax1.xaxis.set_label_coords(0.5, -positionAxes)
    legendItems = len(ax1.get_legend_handles_labels()[0])
    if legendItems <= 15:
        legendProp = {'size':10}
        legendCol = 3
    elif legendItems <= 27:
        legendProp = {'size':8}
        legendCol = 3
    else:
        legendProp = {'size':6}
        legendCol = 3
            
    # Convert out pixel position to axes coordinates
    position = position + 15
    positionAxes = position / axesHeight
    logger.debug('position = ' + str(position))
    logger.debug('positionAxes = ' + str(positionAxes))

    ax1.legend(loc='upper center', 
               bbox_to_anchor=(0.5,-positionAxes), 
               ncol=legendCol,
               prop=legendProp)
    
    textLine = 0.02
    textIncrement = 0.02
    fig.text(0.025,textLine, 'Filters: {:s}'.format(str(filterSpec)))
    textLine += textIncrement
    fig.text(0.025,textLine, 'Max Aggregation: {:g}'.format(maxAggregation))
    
    if ylim is not None:
        ax1.set_ylim(ylim)
    
    limits = ax1.get_ylim()
    ax1.set_yticks(np.linspace( limits[0], limits[1], ynticks+1))

    ax1.set_ylabel(ylabel)    
    ax1.grid(True)
    
    fig.show()
    if savepng:
        filename = path + '/' + title
        fig.savefig(filename + '.png', dpi=300)
        
        #writeTex(filename, title, spec)        
    
    figlist.append(fig)
    return fig,ax1
