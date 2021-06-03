# -*- coding: utf-8 -*-
"""
Created on Tue May 11 12:08:44 2021

@author: khershberger
"""

import numpy as np

from khutility.numeric import complexns
from khutility.datatools import DataMerger
from khutility.pandas import plotFamily

from khutility.filetools import find
from khutility.filetools import extractFromFilename

filelist = []
# filelist = [{'name': r'Q:\LAB\Data\Modeling\temp\load pull pivot.xlsx', 'sheet':'DATA'}]
# filelist = [{'name': r'C:\Users\khershberger\Documents\TEMP\load pull pivot.xlsx', 'sheet':'DATA'}]
filelist += [ r'Q:\LAB\Data\Modeling\05 Loadpul v2\L865W08_R19\L865W08_R19-F2E_25C_20210507T141942_LP.csv', 
              r'Q:\LAB\Data\Modeling\04 Loadpull v1\L865W08_R19\L865W08_R19-F2E_25C_20210506T101605_LP.csv',
              ]


filelist += find(r'Q:\LAB\Data\Modeling\07 Loadpull v3', 
                patternInclude='_LP.csv',
                patternExclude='$a')

def convertComplexColumns(df):
    # Convert textual complex into true complex
    keysToConvert = ['S11', 'S12', 'S21', 'S22', 'A1', 'B1', 'GammaH2']
    for key in keysToConvert:
        if key in df:
            print('Converting {:s} to complex'.format(key))
            df[key] = complexns(df[key])
    return df

columnMap = {'IDQ':'ID_set',
             'S21 Mag':'Gain',
             'S21 Phase': 'Phase'}

# columnGroup = ['RUN','Frequency', 'idxGamma', 'ID_set']
columnGroup = ['idxFile', 'Frequency', 'idxGamma', 'ID_set']

# columnInterp = None
columnInterp = 'PDelivered'
dm = DataMerger()
dm.addExtractor('Version', '[_-]([A-I][0-9][A-F])_')
dm.addExtractor('Reticle', '_(R[0-9]{2})[_-]')
dm.addExtractor('Date', '([0-9]{8}T[0-9]{6})')
dm.load(filelist, columnInterp=columnInterp, interpStep=1., 
        columnGroup=columnGroup,
        columnMap=columnMap,
        handler = convertComplexColumns)

df = dm.data()

# Construct Gamma column
df['Gamma'] = df['Gamma Mag'] * np.exp(1j*df['Gamma Phase'])


# idx = (df['RUN'].values=='F2E_DEBUG_1') * (np.isclose(7.0, df['Frequency']))
idx = (np.isclose(7.0, df['Frequency']))
dfg = df.loc[idx].groupby(columnGroup)

plotFamily(dfg, 'PDelivered', 'Gain')

# df.to_csv(r'Q:\LAB\Data\Modeling\07 Loadpull v3\out.csv', index=None)
