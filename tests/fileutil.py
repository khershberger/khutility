# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 12:04:34 2021

@author: khershberger
"""

import unittest
import os

from khutility import fileutil

class TestFiletoolsMethods(unittest.TestCase):
    def test_find(self):
        find = fileutil.find
        path = '.'

        # Check to make sure empty exclude works
        self.assertLess(0, len(find(path, patternInclude='.', patternExclude='(?!x)x')))
        self.assertLess(0, len(find(path, patternInclude='.', patternExclude='')))
        self.assertLess(0, len(find(path, patternInclude='.', patternExclude=None)))
        
        # Check that empty include works
        self.assertLess(0, len(find(path, patternInclude='', patternExclude='(?!x)x')))
        self.assertLess(0, len(find(path, patternInclude=None, patternExclude='(?!x)x')))
        self.assertLess(0, len(find(path, patternInclude='.', patternExclude='(?!x)x')))

        # Check that unmatchable include works
        self.assertEqual(0, len(find(path, patternInclude='(?!x)x', patternExclude='(?!x)x')))

if __name__ == '__main__':
    unittest.main()