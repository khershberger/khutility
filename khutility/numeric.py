# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:05:33 2021

@author: khershberger
"""

from numpy import vectorize
from re import search
from math import floor, log10

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
    
complexns = vectorize(complexns_core)

def engToFloat(s, debug=False):
    m = search('^[ ]*([-+mn])?([0-9]+)([a-df-zA-DF-Z\.])?([0-9]*)[eE]?(-?[0-9]*)(.*)$', s)
    
    if m is None or len(m.groups()) != 6:
        raise ValueError('Input not recognized as number')
        
    [sSign, sWhole, sRadix, sFraction, sExponent, sExtra] = m.groups()
    
    if len(sExtra) != 0:
        raise ValueError('Extraneous characters found after number')
    
    if debug:
        print(f'[{sSign}] [{sWhole}] [{sRadix}] [{sFraction}] [{sExponent}] [{sExtra}]')
    
    mult = {'G':1e9,
            'M':1e6,
            'k':1e3,
            'r':1,
            '.':1,
            None:1,
            'm':1e-3,
            'u':1e-6,
            'n':1e-9,
            'p':1e-12,
            'f':1e-15,
            'a':1e-18}.get(sRadix)
    
    if mult is None:
        raise ValueError('Unknown radix symbol: ' + str(sRadix))
    
    mapSign = {'m':-1, 'n':-1, '-':-1, '+':1}
    
    sign = mapSign.get(sSign, 1)
    
    exp = 0
    if len(sExponent) != 0:
        exp = int(sExponent)
        
    # Reconstrut the number    
    val = sign * mult * float(sWhole + '.' + sFraction) * 10**exp
    return val
    
def floatToEng( x, sigfig=3, fmtstr=None, si=True):
    '''
    Original code from: https://stackoverflow.com/a/19270863
    Modified since then
    
    Returns float/int value <x> formatted in a simplified engineering format -
    using an exponent that is a multiple of 3.

    sigfig: Number of significatn figures
    fmtstr: printf-style string used to format the value before the exponent.

    si: if true, use SI suffix for exponent, e.g. k instead of e3, n instead of
    e-9 etc.

    E.g. with format='%.2f':
        1.23e-08 => 12.30e-9
             123 => 123.00
          1230.0 => 1.23e3
      -1230000.0 => -1.23e6

    and with si=True:
          1230.0 => 1.23k
      -1230000.0 => -1.23M
    '''
    
    sign = ''
    if x < 0:
        x = -x
        sign = '-'
    exp = int( floor( log10( x)))
    exp3 = exp - ( exp % 3)
    x3 = x / ( 10 ** exp3)

    if si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
        exp3_text = 'yzafpnum kMGTPEZY'[ round((exp3 - (-24)) / 3)]
    elif exp3 == 0:
        exp3_text = ''
    else:
        exp3_text = 'e%s' % exp3
        
    if fmtstr is None:
        fmtstr = '%.{:d}f'.format(sigfig - 1 + exp - exp3)
        # print(fmtstr)

    return ( '%s'+fmtstr+'%s') % ( sign, x3, exp3_text)