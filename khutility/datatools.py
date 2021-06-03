# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 08:28:46 2019

@author: khershberger

Utility to merge multiple data files into a single file.
Features:
    Column mapping
    Interpolation

"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os.path
import re

from khutility.filetools import extractFromFilename

class DataMerger():
    def __init__(self):
        self.df = pd.DataFrame()
        self.extractors = {}
        
    def load(self, filelist, columnMap=None, columnInterp=None, interpStep=1., columnGroup=None, handler=None):
        for fidx,file in enumerate(filelist):
            if isinstance(file, str):
                fname = file
            else:
                fname = file['name']
                sheet_name = file.get('sheet', None)
            print('Loading {:s}'.format(fname))
            
            loader = {}
            loader['.xlsx'] = lambda f: pd.read_excel(f, sheet_name=sheet_name)
            loader['.csv'] = lambda f: pd.read_csv(f)
            
            ext = os.path.splitext(fname)[1]
            dfin = loader[ext](fname)
            
            # Add file information
            dfin['idxFile'] = fidx
            dfin['filename'] = os.path.basename(fname)
            
            if handler is not None:
                print('Running handler')
                dfin = handler(dfin)
            
            # Rename columns
            if columnMap is not None:
                print('Renaming Columns')
                dfin.rename(columns=columnMap, inplace=True)
        
            # Run extractors
            print('Running extractors')
            for key,val in self.extractors.items():
                dfin[key] = extractFromFilename(val, fname)
        
            # Interpolate data
            print('Interpolating')
            if columnInterp is None:
                self.df = dfin
            else:
                out = []
                groups = dfin.groupby(columnGroup)
                for k,g in groups:
                    # print('Processing group {:s}'.format(str(k)))
                    cols = list(g.columns)
                    cols.remove(columnInterp)
                    xold = g[columnInterp].values
                    
                    idx_start = 0
                    idx_stop  = -1
                    
                    # Check for monotonicity
                    if np.any(np.diff(xold) < 0):
                        # Find highest monotonic point
                        tmp = np.nonzero(np.diff(xold) < 0.)[0]
                        idx_stop = tmp[0]
                        print('Non-monotonic for group: {:s}  Using x-axis sub-range {:g} - {:g}'.format(str(k), xold[idx_start], xold[idx_stop]))
                    
                    xnew_start = np.ceil(np.min(xold[idx_start:idx_stop]/interpStep))*interpStep
                    xnew_stop = np.floor(np.max(xold[idx_start:idx_stop]/interpStep))*interpStep + 1e-6
                    xnew = np.round(np.arange(xnew_start, xnew_stop, interpStep),2)  # To resolve floating point precision errors
                    
                    dtmp = {}

                    for c in g.columns:
                        if c == columnInterp:
                            # If this is the column being interpolated along just use xnew
                            dtmp[c] = xnew
                            continue
                        # df.print('Interpolating Column {:s}'.format(c))
                        try:
                            dtmp[c] = interp1d(xold[idx_start:idx_stop], g[c][idx_start:idx_stop], kind='linear')(xnew)
                        except ValueError:
                            # Assuming this is a static data column
                            # Just broadcast data from first element
                            # print('Copying ', c)
                            dtmp[c] = [g.iloc[0][c]] * len(xnew)
                    out.append(dtmp)
            
                # Reconstruct dataframe
                for d in out:
                    self.df = pd.concat([self.df, pd.DataFrame.from_dict(d)], ignore_index=True, sort=False)
    def data(self):
        return self.df
    
    def addExtractor(self, column, pattern):
        self.extractors[column] = pattern

# Custom data handlers
        # sheet = 'DATA'
        # dumbsuffix = '1-750.1_C4_2_SN11_2.45GHz_25C_15mA_0mA_50mA_12.03V_0V_12V_contour'
    # # Simplify the stupid column names
    # columns = dfin.columns
    # colmap = {}
    
    # for cname in columns:
    #     #bleh = cname.split(' - ')
    #     bleh = cname.replace(dumbsuffix, '')
    #     bleh = bleh.replace(' - ', '')
        
    #     colmap[cname] = bleh