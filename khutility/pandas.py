# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:42:45 2019

@author: kyleh
"""

from matplotlib.pyplot import subplots
import pandas as pd

def getColumn(df, name):
    """ THis gets a column from a Datafram regardless if it is an index or data column"""
    
    # First figur out if this is a data or index column
    if name in df.columns:
        return df[name].values
    elif name in df.index.names:
        return df.index.get_level_values(name).values
    else:
        raise ValueError('Name is not a data or index column')
    
def plotFamily(df,x,y,ax=None,**kwargs):
    if ax is None:
        f,ax = subplots()

    if isinstance(df, pd.core.groupby.generic.DataFrameGroupBy):
        # print('Plotting group')
        for name,group in df:
            # print('Plotting {:s}'.format(str(name)))
            ax = group.plot(x,y,ax=ax, label=str(name), **kwargs)
        return ax
    else:
        return df.plot(x,y,ax=ax,**kwargs)