# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:28:26 2021

@author: khershberger
"""

import os
import re
import logging

logger = logging.getLogger(__name__)


def find(path, patternInclude=".*", patternExclude="(?!x)x"):
    """
    Parameters
    ----------
    path : String
        Path to begin sedrch in
    patternInclude : String, optional
        Regular expression to compare filenames against
        Matches will be included in output
        Ex:
            'tdms$': load ALL .tdms files found in path
    patternExclude : String, optional
        Regular expression to compare filenames against
        Matches will never be included in output
        Ex:
            
    Returns
    -------
    List of filenames relative to specified path

    """

    # See if an empty include/exclude patterns were passed in and replace
    # with an appropriate regex
    if patternInclude is None:
        patternInclude = "."  # Match anything
    if patternExclude is None or patternExclude == "":
        patternExclude = "(?!x)x"  # Match nothing

    reInclude = re.compile(patternInclude)
    reExclude = re.compile(patternExclude)

    filelist = []
    for t in os.walk(path):
        logger.info("Searching dir %s", t[0])
        # Iterate through filename list
        filelist += [
            os.path.join(t[0], x)
            for x in t[2]
            if reInclude.search(x) and not reExclude.search(x)
        ]

    return filelist


def extractFromFilename(pattern, s, unknown="UNK"):
    # check input
    if not isinstance(s, str):
        return unknown

    if isinstance(pattern, re.Pattern):
        m = pattern.findall(s)
    else:
        m = re.findall(pattern, s)

    if m:
        return m[-1]  # Return last occurance
    else:
        return unknown
