# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:05:33 2021

@author: khershberger
"""

from numpy import vectorize

def complexns_core(strin):
    """
    Improved conversion of string representation of complex numbers to float
    """

    try:
        return complex(strin)
    except ValueError:
        pass
    
    # Do the simple fixes
    tmp = strin.replace(' ', '').replace('*','')
    tmp = tmp.lower().replace('i','j')
    
    try:
        return complex(strin)
    except ValueError:
        pass

    # Try to move 'j' to end    
    loc = tmp.find('j')
    if loc:
        #tmp[1] = tmp[1][1:-1] + tmp[1][0]
        tmp = tmp[:loc] + tmp[(loc+1):] + 'j'
    return complex(tmp)
    
complex = vectorize(complexns_core)