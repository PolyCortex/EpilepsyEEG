#!/usr/bin/env python

# file: src/sys_tools/nedc_text_tools.py
#
# usage:
#  import nedc_text_tools as ntt
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#------------------------------------------------------------------------------

# import required system modules
#

# import required NEDC modules
#

#------------------------------------------------------------------------------
#                                                                              
# define important constants
#                                                                              
#------------------------------------------------------------------------------

# define a constant that controls the amount of precision used
# to check floating point numbers
#
MAX_PRECISION = int(10)

# define a constant used to indicate a null choice
#
NULL_CLASS = "***"

#------------------------------------------------------------------------------
#                                                                              
# functions are listed here
#                                                                              
#------------------------------------------------------------------------------

# function: nedc_first_substring
#
# arguments:
#  strings: list of strings (input)
#  substring: the substring to be matched (input)
#
# return: the index of the match in strings
#
# This function finds the index of the first string in strings that
# contains the substring. This is similar to running strstr on each
# element of the input list.
#
def first_substring(strings_a, substring_a):
    return next(i for i, string in enumerate(strings_a) if \
                substring_a in string)
#
# end of function

# function: nedc_first_string
#
# arguments:
#  strings: list of strings (input)
#  substring: the string to be matched (input)
#
# return: the index of the match in strings
#
# This function finds the index of the first string in strings that
# contains an exact match. This is similar to running strstr on each
# element of the input list.
#
def first_string(strings_a, tstring_a):
    return next(i for i, string in enumerate(strings_a) if \
                tstring_a == string)
#
# end of function

#
# end of file
                
