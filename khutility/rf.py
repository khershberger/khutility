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
    """
    Returns function that provides QPSK CCDF value for given relative power level.
    """
    
    # ccdf1: looks to be more correct, but not quite the right shape
    # ccdf2: No clue where this came from, it looks pretty wrong though.
    
    ccdf1 = {
        'r_db': np.array([-42.232, -41.75 , -41.267, -40.784, -40.301, -39.818, -39.335,
           -38.852, -38.37 , -37.887, -37.404, -36.921, -36.438, -35.955,
           -35.472, -34.99 , -34.507, -34.024, -33.541, -33.058, -32.575,
           -32.092, -31.61 , -31.127, -30.644, -30.161, -29.678, -29.195,
           -28.712, -28.229, -27.747, -27.264, -26.781, -26.298, -25.815,
           -25.332, -24.849, -24.367, -23.884, -23.401, -22.918, -22.435,
           -21.952, -21.469, -20.987, -20.504, -20.021, -19.538, -19.055,
           -18.572, -18.089, -17.606, -17.124, -16.641, -16.158, -15.675,
           -15.192, -14.709, -14.226, -13.744, -13.261, -12.778, -12.295,
           -11.812, -11.329, -10.846, -10.364,  -9.881,  -9.398,  -8.915,
            -8.432,  -7.949,  -7.466,  -6.983,  -6.501,  -6.018,  -5.535,
            -5.052,  -4.569,  -4.086,  -3.603,  -3.121,  -2.638,  -2.155,
            -1.672,  -1.189,  -0.706,  -0.223,   0.259,   0.742,   1.225,
             1.708,   2.191,   2.674,   3.157,   3.639,   4.122,   4.605,
             5.088,   5.571]),
        'CCDF': np.array([9.9998e-01, 9.9998e-01, 9.9998e-01, 9.9998e-01, 9.9998e-01,
           9.9998e-01, 9.9998e-01, 9.9998e-01, 9.9998e-01, 9.9998e-01,
           9.9996e-01, 9.9996e-01, 9.9996e-01, 9.9996e-01, 9.9996e-01,
           9.9994e-01, 9.9990e-01, 9.9986e-01, 9.9984e-01, 9.9984e-01,
           9.9982e-01, 9.9974e-01, 9.9972e-01, 9.9972e-01, 9.9965e-01,
           9.9963e-01, 9.9963e-01, 9.9951e-01, 9.9945e-01, 9.9941e-01,
           9.9933e-01, 9.9927e-01, 9.9915e-01, 9.9898e-01, 9.9886e-01,
           9.9878e-01, 9.9858e-01, 9.9843e-01, 9.9829e-01, 9.9811e-01,
           9.9782e-01, 9.9754e-01, 9.9721e-01, 9.9699e-01, 9.9672e-01,
           9.9628e-01, 9.9567e-01, 9.9504e-01, 9.9436e-01, 9.9390e-01,
           9.9322e-01, 9.9255e-01, 9.9158e-01, 9.9046e-01, 9.8936e-01,
           9.8804e-01, 9.8682e-01, 9.8543e-01, 9.8370e-01, 9.8189e-01,
           9.8018e-01, 9.7815e-01, 9.7562e-01, 9.7333e-01, 9.7066e-01,
           9.6720e-01, 9.6441e-01, 9.6032e-01, 9.5609e-01, 9.5233e-01,
           9.4696e-01, 9.4142e-01, 9.3583e-01, 9.2852e-01, 9.2120e-01,
           9.1302e-01, 9.0455e-01, 8.9479e-01, 8.8408e-01, 8.7123e-01,
           8.5560e-01, 8.3517e-01, 8.1047e-01, 7.7883e-01, 7.3673e-01,
           6.8229e-01, 6.1276e-01, 5.2787e-01, 4.3229e-01, 3.3253e-01,
           2.3769e-01, 1.5803e-01, 9.7420e-02, 5.6260e-02, 3.2430e-02,
           1.7050e-02, 8.9900e-03, 3.5800e-03, 1.3800e-03, 2.2000e-04])
        }
    ccdf3 = {
        'r_db': np.array([-70. , -60. , -50. , -40. , -30. , -20. , -19. , -18. , -17. ,
           -16. , -15. , -14. , -13. , -12. , -11. , -10. ,  -9. ,  -8. ,
            -7. ,  -6. ,  -5. ,  -4. ,  -3. ,  -2. ,  -1. ,   0. ,   0.2,
             0.4,   0.6,   0.8,   1. ,   1.2,   1.4,   1.6,   1.8,   2. ,
             2.1,   2.2,   2.3,   2.4,   2.5,   2.6,   2.7,   2.8,   2.9,
             3. ,   3.1,   3.2,   3.3]),
        'CCDF': np.array([1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
           1.00000000e+00, 9.99900000e-01, 9.99758108e-01, 9.98016000e-01,
           9.93048649e-01, 9.84172973e-01, 9.71464865e-01, 9.63232432e-01,
           9.61800000e-01, 9.61800000e-01, 9.61800000e-01, 9.61800000e-01,
           9.61800000e-01, 9.61800000e-01, 9.55255405e-01, 9.30129730e-01,
           9.23700000e-01, 9.23700000e-01, 8.84141892e-01, 7.69064865e-01,
           7.01686486e-01, 5.42421317e-01, 4.76364865e-01, 4.22156757e-01,
           3.76571622e-01, 3.25716216e-01, 2.66231081e-01, 2.26022973e-01,
           1.97120270e-01, 1.56397297e-01, 1.04521622e-01, 7.84540540e-02,
           7.14243240e-02, 6.14297300e-02, 4.93432430e-02, 3.63810810e-02,
           2.46243240e-02, 1.65716220e-02, 1.40621620e-02, 1.21540540e-02,
           8.93783800e-03, 5.90405400e-03, 3.56756800e-03, 1.45405400e-03,
           2.10811000e-04])
        }
    return interpolate.interp1d(ccdf1['r_db'], ccdf1['CCDF'], kind='linear', bounds_error=False)
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

