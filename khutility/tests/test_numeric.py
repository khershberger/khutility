# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:34:59 2021

@author: khershberger
"""

from math import pi
import unittest

import numeric


def isClose(a, b):
    return abs(a - b) < 1e-18


class TestNumericMethods(unittest.TestCase):
    def test_complexns(self):
        fcn = numeric.complexns

        self.assertTrue(isClose(fcn("1+i2"), complex(1.0, 2.0)))
        self.assertTrue(isClose(fcn("1+j2"), complex(1.0, 2.0)))
        self.assertTrue(isClose(fcn("1+2i"), complex(1.0, 2.0)))
        self.assertTrue(isClose(fcn("1+2j"), complex(1.0, 2.0)))
        # self.assertTrue(isClose(fcn('1i'), complex(1.0, 0.0)))
        # self.assertTrue(isClose(fcn('1j'), complex(1.0, 0.0)))
        # self.assertTrue(isClose(fcn('i1'), complex(1.0, 0.0)))
        # self.assertTrue(isClose(fcn('j1'), complex(1.0, 0.0)))

    def test_engToFloat(self):
        fcn = numeric.engToFloat

        self.assertRaises(ValueError, fcn, "asdf")
        self.assertRaises(ValueError, fcn, "1q2")
        self.assertRaises(ValueError, fcn, "1 2")
        self.assertRaises(ValueError, fcn, "1.23 2")
        self.assertRaises(ValueError, fcn, "-  1n234")

        self.assertTrue(isClose(fcn("-4.123"), -4.123))
        self.assertTrue(isClose(fcn("1"), 1.0))
        self.assertTrue(isClose(fcn("  1G234"), 1.234e9))
        self.assertTrue(isClose(fcn("  1M234"), 1.234e6))
        self.assertTrue(isClose(fcn("  1k234"), 1.234e3))
        self.assertTrue(isClose(fcn("  1.234"), 1.234))
        self.assertTrue(isClose(fcn("  1r234"), 1.234))
        self.assertTrue(isClose(fcn("  1m234"), 1.234e-3))
        self.assertTrue(isClose(fcn("  1u234"), 1.234e-6))
        self.assertTrue(isClose(fcn("  1n234"), 1.234e-9))
        self.assertTrue(isClose(fcn("  1p234"), 1.234e-12))
        self.assertTrue(isClose(fcn("  1f234"), 1.234e-15))
        self.assertTrue(isClose(fcn("  1a234"), 1.234e-18))

        self.assertTrue(isClose(fcn("  -1u234"), -1.234e-6))

    def test_floatToEng(self):
        fcn = numeric.floatToEng

        # self.assertRaises(ValueError, fcn, 'sadf')

        # Test prefixes
        self.assertEqual(fcn(6.282e18, sigfig=2, si=True), "6.3E")
        self.assertEqual(fcn(6.282e15, sigfig=2, si=True), "6.3P")
        self.assertEqual(fcn(6.282e12, sigfig=2, si=True), "6.3T")
        self.assertEqual(fcn(6.282e9, sigfig=2, si=True), "6.3G")
        self.assertEqual(fcn(6.282e6, sigfig=2, si=True), "6.3M")
        self.assertEqual(fcn(6.282e3, sigfig=2, si=True), "6.3k")
        self.assertEqual(fcn(6.282e0, sigfig=2, si=True), "6.3")
        self.assertEqual(fcn(6.282e-3, sigfig=2, si=True), "6.3m")
        self.assertEqual(fcn(6.282e-6, sigfig=2, si=True), "6.3u")
        self.assertEqual(fcn(6.282e-9, sigfig=2, si=True), "6.3n")
        self.assertEqual(fcn(6.282e-12, sigfig=2, si=True), "6.3p")
        self.assertEqual(fcn(6.282e-15, sigfig=2, si=True), "6.3f")
        self.assertEqual(fcn(6.282e-18, sigfig=2, si=True), "6.3a")

        # Test sign
        self.assertEqual(fcn(-6.282e3, sigfig=2, si=True), "-6.3k")

        # Test sigfigs
        self.assertEqual(fcn(pi * 1e3, sigfig=1, si=True), "3k")
        self.assertEqual(fcn(pi * 1e3, sigfig=2, si=True), "3.1k")
        self.assertEqual(fcn(pi * 1e3, sigfig=3, si=True), "3.14k")
        self.assertEqual(fcn(pi * 1e3, sigfig=4, si=True), "3.142k")
        self.assertEqual(fcn(pi * 1e3, sigfig=5, si=True), "3.1416k")
        self.assertEqual(fcn(pi * 1e3, sigfig=6, si=True), "3.14159k")
        self.assertEqual(fcn(pi * 1e-3, sigfig=1, si=True), "3m")
        self.assertEqual(fcn(pi * 1e-3, sigfig=2, si=True), "3.1m")
        self.assertEqual(fcn(pi * 1e-3, sigfig=3, si=True), "3.14m")
        self.assertEqual(fcn(pi * 1e-3, sigfig=4, si=True), "3.142m")
        self.assertEqual(fcn(pi * 1e-3, sigfig=5, si=True), "3.1416m")
        self.assertEqual(fcn(pi * 1e-3, sigfig=6, si=True), "3.14159m")

        self.assertEqual(fcn(123456, sigfig=1, si=True), "100k")
        self.assertEqual(fcn(123456, sigfig=2, si=True), "120k")
        self.assertEqual(fcn(123456, sigfig=3, si=True), "123k")
        self.assertEqual(fcn(123456, sigfig=4, si=True), "123.5k")
        self.assertEqual(fcn(123456, sigfig=5, si=True), "123.46k")
        self.assertEqual(fcn(123456, sigfig=6, si=True), "123.456k")

        self.assertEqual(fcn(0.123456, sigfig=1, si=True), "100m")
        self.assertEqual(fcn(0.123456, sigfig=2, si=True), "120m")
        self.assertEqual(fcn(0.123456, sigfig=3, si=True), "123m")
        self.assertEqual(fcn(0.123456, sigfig=4, si=True), "123.5m")
        self.assertEqual(fcn(0.123456, sigfig=5, si=True), "123.46m")
        self.assertEqual(fcn(0.123456, sigfig=6, si=True), "123.456m")

        # Test decimal placement
        self.assertEqual(fcn(pi * 1e3, sigfig=4, si=True), "3.142k")
        self.assertEqual(fcn(pi * 1e4, sigfig=4, si=True), "31.42k")
        self.assertEqual(fcn(pi * 1e5, sigfig=4, si=True), "314.2k")
        self.assertEqual(fcn(pi * 1e6, sigfig=4, si=True), "3.142M")

        self.assertEqual(
            fcn(pi * 1e3, sigfig=4, si=True, replace_decimal=True), "3k142"
        )
        self.assertEqual(
            fcn(pi * 1e4, sigfig=4, si=True, replace_decimal=True), "31k42"
        )
        self.assertEqual(
            fcn(pi * 1e5, sigfig=4, si=True, replace_decimal=True), "314k2"
        )
        self.assertEqual(
            fcn(pi * 1e6, sigfig=4, si=True, replace_decimal=True), "3M142"
        )


if __name__ == "__main__":
    unittest.main()
