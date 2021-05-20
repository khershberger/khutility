# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:28:26 2021

@author: khershberger
"""

import os
import re
import logging

logger = logging.getLogger(__name__)

def find(path, patternInclude='.*', patternExclude='$a'):
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
    
    reInclude = re.compile(patternInclude)
    reExclude = re.compile(patternExclude)
    
    filelist = []
    for t in os.walk(path):
        logger.info('Searching dir %s', t[0])
        # Iterate through filename list
        filelist += [ os.path.join(t[0], x) for x in t[2] if reInclude.search(x) and not reExclude.search(x)]

    return filelist