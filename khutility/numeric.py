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
    tmp = strin.replace(" ", "").replace("*", "")
    tmp = tmp.lower().replace("i", "j")

    try:
        return complex(strin)
    except ValueError:
        pass

    # Try to move 'j' to end
    loc = tmp.find("j")
    if loc:
        # tmp[1] = tmp[1][1:-1] + tmp[1][0]
        tmp = tmp[:loc] + tmp[(loc + 1) :] + "j"
    return complex(tmp)


complexns = vectorize(complexns_core)

prefix_si = {
    "G": 1e9,
    "M": 1e6,
    "k": 1e3,
    "r": 1,
    ".": 1,
    None: 1,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "a": 1e-18,
}


def engToFloat(s, debug=False):
    m = search(
        "^[ ]*([-+mn])?([0-9]+)([a-df-zA-DF-Z\.])?([0-9]*)[eE]?(-?[0-9]*)(.*)$", s
    )

    if m is None or len(m.groups()) != 6:
        raise ValueError("Input not recognized as number")

    [sSign, sWhole, sRadix, sFraction, sExponent, sExtra] = m.groups()

    if len(sExtra) != 0:
        raise ValueError("Extraneous characters found after number")

    if debug:
        print(f"[{sSign}] [{sWhole}] [{sRadix}] [{sFraction}] [{sExponent}] [{sExtra}]")

    mult = prefix_si.get(sRadix)

    if mult is None:
        raise ValueError("Unknown radix symbol: " + str(sRadix))

    mapSign = {"m": -1, "n": -1, "-": -1, "+": 1}

    sign = mapSign.get(sSign, 1)

    exp = 0
    if len(sExponent) != 0:
        exp = int(sExponent)

    # Reconstrut the number
    val = sign * mult * float(sWhole + "." + sFraction) * 10 ** exp
    return val


def floatToEng(x, sigfig=3, si=True, replace_decimal=False):
    """
    Original code from: https://stackoverflow.com/a/19270863
    Modified since then
    
    Returns float/int value x formatted in a simplified engineering format -
    using an exponent that is a multiple of 3.

    sigfig: Number of significatn figures
    si: if true, use SI suffix for exponent, e.g. k instead of e3, n instead of
    e-9 etc.
    replace_decimal: If true, si suffix will be used for the decimal point

    With si=True:
          1230.0 => 1.23k      (sigfig=3)
          1230.0 => 1k         (sigfig=1)
      -1230000.0 => -1.23M

    With si=True and replace_decimal=True:
          1230.0 => 1k23       (sigfig=3)
          1230.0 => 1k         (sigfig=1)
      -1230000.0 => -1M23
    
    """
    sign = ""
    if x < 0:
        x = -x
        sign = "-"
    e = int(floor(log10(x)))
    e3 = e - (e % 3)

    # Figure out unit symbol
    if si and e3 >= -24 and e3 <= 24 and e3 != 0:
        e3_text = "yzafpnum kMGTPEZY"[round((e3 - (-24)) / 3)]
    elif e3 == 0:
        e3_text = ""
    else:
        e3_text = "e%s" % e3

    # Convert to scientific notation with desired precision
    s = f"{{:.{sigfig-1}e}}".format(x)

    # Remove decimal & strip off trailing e-notation
    s = s[: s.find("e")].replace(".", "")

    # Now figure out where to place decimal
    position = e - e3 + 1
    # print(f"{e=} {e3=} {position=} {s=}")

    if position < sigfig:
        if replace_decimal and si:
            decimal = e3_text
            unit = ""
        else:
            decimal = "."
            unit = e3_text

        s = s[0:position] + decimal + s[position:] + unit
    else:
        # Pad with zeros
        s += "0" * (position - sigfig) + e3_text

    # Add sign & multiplier symbol
    s = sign + s

    return s
