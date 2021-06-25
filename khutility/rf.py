# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 16:52:52 2019

@author: kyleh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import logging
import timeit

#logging.basicConfig(format='%(asctime)s %(message)s')
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def evm_test():
    pin = np.array([-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1])
    vref = np.array([-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,-0,0,0])
    vmeas_mag   = np.array([0.388208,0.435562,0.488691,0.548302,0.615199,0.690284,0.774567,0.86921,0.975524,1.09501,1.22938,1.38057,1.55077,1.74244,1.95833,2.20151,2.4754,2.78388,3.13147,3.52383,3.96812,4.47208,5.04201,5.68003,6.36363,7.01236,7.57652,8.03131,8.42095,8.6897,8.83684,8.93731])
    vmeas_phase = np.array([75.5874,75.566,75.547,75.5265,75.494,75.4446,75.3933,75.3271,75.2512,75.1623,75.0596,74.9533,74.8485,74.7478,74.6585,74.5885,74.5431,74.5162,74.5071,74.5075,74.4968,74.4426,74.3026,74.0644,73.7472,73.2589,72.5501,71.3563,69.5733,68.0518,67.122,66.336])

    #pout = 20.0*np.log10(vmeas_mag)
    pout = 10.0*np.log10(vmeas_mag**2/(2*50.0)) + 30
    
    phase = vmeas_phase

    evm_pin1, evm1 = calcevm_interp(pin, pout, phase, 1, -10, 6, method='a')
    evm_pin2, evm2 = calcevm_interp(pin, pout, phase, 1, -10, 6, method='b')
    evm3 = calcevm(pin, pout, phase, 1, -10, 6)
    
    evm_ofdm    = 20.0 * np.log10(calcevm(pin,pout,phase,1,-10,8, ccdf=ccdfOfdm, allow_clipping=True))
    evm_qpsk    = 20.0 * np.log10(calcevm(pin,pout,phase,1,-10,5, ccdf=ccdfQpsk, allow_clipping=True))
    evm_twotone = 20.0 * np.log10(calcevm(pin,pout,phase,1,-10,3, ccdf=ccdfTwoTone, allow_clipping=True))

def pdfOfdm(x):
    return np.exp(-x)

def ccdfOfdm(x):
    return np.exp(-x)

def ccdfTwoTone(x):
    # Need to hard cap values at 2
    x[x > 2.0] = 2.0
    logger.debug(f'{x}')
    return 1 - np.arcsin(x/2) / (np.pi/2)

def ccdfQpskCreator():
    data = np.genfromtxt('qpsk_ccdf.csv',delimiter=',', skip_header=1)
    return interpolate.interp1d(data[:,0], data[:,1], kind='linear', bounds_error=False)
ccdfQpsk = ccdfQpskCreator()

def calcevm_interp(pin, pout, phase, step, llimit, hlimit, ccdf=None, method='a'):
    if (llimit > 0):
        raise ValueError('Lower limit must be less than 0')
    if (hlimit < 0):
        raise ValueError('Upper limit must be greater than 0')
    if ( (llimit % step != 0) or (hlimit % step) != 0):
        raise ValueError('Uppwer and Lower limts must be multiple of step')

    if ccdf is None:
        def ccdf(x):
            return np.exp(-x)
    
    ln10 = np.log(10)
    
    # Generate new list of pin points to use for subsequent calculations and data
    # Fix points to even multiples of step size.
    pin_avg = np.arange(np.ceil(min(pin)/step), np.floor(max(pin)/step)+1, 1) * step
    
    pin_num_total = len(pin_avg)

    pin_num_valid = pin_num_total - int(np.floor(hlimit/step))
    
    # Interpolate inputs to be on standard step
    poutinterp = np.interp(pin_avg, pin, pout)
    gaininterp = poutinterp - pin_avg
    phaseinterp = np.interp(pin_avg, pin, phase)

    # This isn'ta ctually needed
    #dgain  = gaininterp - gaininterp[0]
    #dphase = phaseinterp - phaseinterp[0]
    
    evm      = np.array([np.float('nan')] * pin_num_total)
    evm_amam = evm.copy()
    evm_ampm = evm.copy()
    
    # r is the relative power offset used in the evm euqtion
    
    rValuesDb = np.arange(llimit, hlimit+step, step)
    #rNumPoints = round((hlimit-llimit)/step + 1)
    rNumPoints = len(rValuesDb)
    #rIndexes = np.arange(rNumPoints)
    rIndexOffset = round(llimit/step)

    # Method 1a  - This is fastest
    #rValues = 10**((rIndexes + rIndexOffset) * step / 10.0)
    # Method 1b - This is even faster
    rValues = 10**(rValuesDb/10.0)
    # Method 2
    #stepfactor = 10**(step/10.0)
    #rValues = 10**(llimit/10.0) * np.array([ stepfactor**n for n in range(rNumPoints)])
    # Method 3  - This is slowest
    #rValues = np.geomspace(10**(llimit/10.0), 10**(hlimit/10.0), num=rNumPoints)
    
    scaleHalfStep = 10**((step/2)/10)
    
    if method == 'a' or method == 'ac':
        # This calculates the spacing between each step and the next
        drValues = 10**(step/10.0) * rValues - rValues  # Uppper
        # This shifts that value to represent the spacing between -step/2 and +step/2
        if (method == 'ac'):
            drValues /= scaleHalfStep
        ## Sloppy method
        rProbabilities = drValues*pdf(rValues)
    if method == 'b':
        # More accurater method?
        # CDF(x1) - CDF(x0) = (1 - ccdf(x1)) - (1 - ccdf(x0)) = -1 * (ccdf(x1) - ccdf(x0))
        rProbabilities = -1 * (ccdf(rValues*scaleHalfStep) - ccdf(rValues/scaleHalfStep))

    for k1 in range(pin_num_valid):
        logger.debug(f'New pin level k1={k1} pin={pin[k1]}')
        evm_amam[k1] = 0
        evm_ampm[k1] = 0
        
        for idxpdf in range(rNumPoints):
            kTmp = k1 + idxpdf + rIndexOffset
            logger.debug(f'Integral:  idxpdf={idxpdf} kTmp={kTmp}' )
            if (kTmp >= 0) and (kTmp < pin_num_total):
                # This was the old way of doing things
                # Re-calculated pdf every single time
                #r  = rValues[idxpdf]
                #dr = drValues[idxpdf]
                #logger.debug(f'r = {r}  dr = {dr} dgain[kTmp]={gaininterp[kTmp]} dgain[k1]={gaininterp[k1]}')
                #evm_amam[k1] += dr*pdf(r)*r*((gaininterp[kTmp] - gaininterp[k1])**2)
                #evm_ampm[k1] += dr*pdf(r)*r*((phaseinterp[kTmp] - phaseinterp[k1])**2)
                
                # New method uses pre-calculated probabilities
                r  = rValues[idxpdf]
                evm_amam[k1] += rProbabilities[idxpdf]*r*((gaininterp[kTmp] - gaininterp[k1])**2)
                evm_ampm[k1] += rProbabilities[idxpdf]*r*((phaseinterp[kTmp] - phaseinterp[k1])**2)
                    
        
        evm_amam[k1] = ln10 / 20.0   * np.sqrt(evm_amam[k1])
        evm_ampm[k1] = np.pi / 180.0 * np.sqrt(evm_ampm[k1])
        evm[k1] = np.sqrt(evm_amam[k1]**2 + evm_ampm[k1]**2)

    return pin_avg, evm

# This version of the function makes no assumptions about the number of points
# or their spacing in the input data.
def calcevm(pin, pout, phase, step, llimit, hlimit, ccdf=None, allow_clipping=False):
    """
    Calculates EVM and outputs using same pin values as the input
    """
    if (llimit > 0):
        raise ValueError('Lower limit must be greater than 0')
    if (hlimit < 0):
        raise ValueError('Upper limit must be less than 0')
    if ( (llimit % step != 0) or (hlimit % step) != 0):
        raise ValueError('Uppwer and Lower limts must be multiple of step')
    
    if ccdf is None:
        def ccdf(x):
            return np.exp(-x)
    
    ln10 = np.log(10)
    
    pinNum = len(pin)
    minPin = np.min(pin)
    maxPin = np.max(pin)
    
    # Generate interpolation function
    # Fill values
    if allow_clipping:
        # This will generate a  hard-clip for Pout
        fill_value =  ([np.float('nan'), np.float('nan')],[pout[pinNum-1], phase[pinNum-1]])
    else:
        fill_value = np.float('nan')
    
    ifcn = interpolate.interp1d(pin, np.stack((pout,phase)), kind='cubic', 
                                bounds_error=False, fill_value=fill_value)

    evm      = np.array([np.float('nan')] * len(pin))
    evm_amam = evm.copy()
    evm_ampm = evm.copy()
    
    rValuesDb = np.arange(llimit, hlimit+step, step)
    rValues = 10**(rValuesDb/10.0)
    rNumPoints = len(rValues)
    rIndexOffset = round(llimit/step)
    
    scaleHalfStep = 10**((step/2)/10)

    # CDF(x1) - CDF(x0) = (1 - ccdf(x1)) - (1 - ccdf(x0)) = -1 * (ccdf(x1) - ccdf(x0))
    rProbabilities = -1 * (ccdf(rValues*scaleHalfStep) - ccdf(rValues/scaleHalfStep))
    for k1 in range(len(pin)):
        logger.debug(f'New pin level k1={k1} pin={pin[k1]}')

        # First, make sure there is enough head-room for this power level
        if (pin[k1] + hlimit <= maxPin) or allow_clipping:
            evm_amam[k1] = 0.0
            evm_ampm[k1] = 0.0
        
            (poutNom, phaseNom) = ifcn(pin[k1])
            gainNom = poutNom - pin[k1]
            for idxpdf in range(rNumPoints):
                pdb = pin[k1] + rValuesDb[idxpdf]
                
                # Determine if we need data below minimum input power
                # If so, just use whatever the minimum power level is
                # Essentially creating a flat extrapolation on the lower end
                # of the power sweep
                if (pdb >= minPin):
                    pinRel = pin[k1] + rValuesDb[idxpdf]
                else:
                    pinRel = minPin

                (poutRel, phaseRel) = ifcn(pinRel)
                gainRel = poutRel - pinRel

                r  = rValues[idxpdf]
                evm_amam[k1] += rProbabilities[idxpdf]*r*((gainRel - gainNom)**2)
                evm_ampm[k1] += rProbabilities[idxpdf]*r*((phaseRel - phaseNom)**2)
                    
            evm_amam[k1] = ln10 / 20.0   * np.sqrt(evm_amam[k1])
            evm_ampm[k1] = np.pi / 180.0 * np.sqrt(evm_ampm[k1])
            evm[k1] = np.sqrt(evm_amam[k1]**2 + evm_ampm[k1]**2)

    return evm

