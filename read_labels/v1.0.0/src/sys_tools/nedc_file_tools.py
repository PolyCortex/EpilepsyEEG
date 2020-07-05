#!/usr/bin/env python
#
# file: src/sys_tools/nedc_file_tools.py
#                                                                              
# usage:                                                                       
#  import nedc_file_tools as nft
#                                                                              
# This class contains a collection of functions that deal with file handling
#------------------------------------------------------------------------------
#                                                                             
# imports are listed here                                                     
#                                                                             
#------------------------------------------------------------------------------

# import system modules
#
import os
import re
import sys
import errno
from collections import OrderedDict

#------------------------------------------------------------------------------
#                                                                              
# global variables are listed here                                             
#                                                                              
#------------------------------------------------------------------------------

# file processing constants
#
DELIM_COMMENT = '#'
DELIM_SPACE = ' '
DELIM_NULL = ''
DELIM_NEWLINE = '\n'
DELIM_EQUAL = '='
DELIM_BOPEN = '{'
DELIM_BCLOSE = '}'
DELIM_OPEN = '['
DELIM_CLOSE = ']'
DELIM_COLON = ':'
DELIM_SEMI = ';'
DELIM_COMMA = ','
DELIM_SLASH = '/'
DELIM_USCORE = '_'
DELIM_SQUOTE = '\''
DELIM_QUOTE = '"'
DELIM_CARRIAGE = '\r'

# parameter file constants
#
DELIM_VERSION = "version"
PFILE_VERSION = "params_v1.0.0"

# output file constants
#
NEDC_DEF_EXT = ".txt"
NEDC_DEF_DIR = "."

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

# function: get_flist
#                                                                          
# arguments:
#  fname: full pathname of a filelist file
#                                              
# return:
#  flist: a list of filenames
#
# This function opens a file and reads filenames. It ignores comment
# lines and blank lines.
#
def get_flist(fname_a):

    # declare local variables
    #
    flist = []

    # open the file
    #
    try: 
        fp = open(fname_a, "r") 
    except IOError: 
        print("%s (%s: %s): file not found (%s)" \
            % (sys.argv[0], __name__, "get_flist", fname_a))
        return None

    # iterate over lines
    #
    for line in fp:

        # remove spaces and newline chars
        #
        line = line.replace(DELIM_SPACE, DELIM_NULL) \
                   .replace(DELIM_NEWLINE, DELIM_NULL)

        # check if the line starts with comments
        #
        if line.startswith(DELIM_COMMENT) or len(line) == 0:
            pass
        else:
            flist.append(line)

    # close the file
    #
    fp.close()

    # exit gracefully
    #
    return flist
#                                                                          
# end of function

# function: load_parameters
#                                                                          
# arguments:
#  pfile: path of a paramter file
#  keyword: section of the parameter file to load
#                                              
# return: an ordereddict, containing the values in the section
#
def load_parameters(pfile_a, keyword_a):

    # declare local variables
    #
    values = OrderedDict()

    # make sure the file is a parameter file
    #
    if get_version(pfile_a) != PFILE_VERSION:
        return None

    # open the file
    #
    try: 
        fp = open(pfile_a, "r") 
    except ioerror: 
        print("%s (%s: %s): file not found (%s)" \
            % (sys.argv[0], __name__, "load_parameters", pfile_a))
        return None

    # loop over all lines in the file
    #
    flag_pblock = False
    for line in fp:

        # clean up the line
        #
        str = (line.replace(DELIM_SPACE, DELIM_NULL)) \
                   .replace(DELIM_NEWLINE, DELIM_NULL)

        # throw away commented and blank lines
        #
        if (str.startswith(DELIM_COMMENT) == True) or (len(str) == 0):
            pass

        # check for the beginning of a parameter block
        #
        elif (str.startswith(keyword_a) == True) and (DELIM_BOPEN in str):
            flag_pblock = True

        # check for the end of a parameter block:
        #  note that we exit if we hit the end of the parameter block
        #
        elif (flag_pblock == True) and (DELIM_BCLOSE in str):
            fp.close();
            return values

        # otherwise, if the parameter block has started, decode a parameter
        #  by splitting and assigning to a dictionary
        #
        elif (flag_pblock == True):
            parts = str.split(DELIM_EQUAL)
            values[parts[0]] = parts[1]

    # make sure we found a block
    #
    if flag_pblock == False:
        fp.close()
        print("%s (%s: %s): invalid parameter file (%s)" \
            % (sys.argv[0], __name__, "load_parameters", pfile_a))
        return None
    
    # exit gracefully
    #
    return values
#                                                                          
# end of function

# function: load_montage
#
# arguments:
#  pfile: path of a parameter file
#  keyword: section of the parameter file to load
#
# return: an OrderedDict, containing the montage information
#
# This method is a more specific implementation of load_parameters, used
# specifically for loading a montage section
#
def load_montage(pfile_a, keyword_a):

    # declare local variables
    #
    montage = OrderedDict()

    # make sure the file is a parameter file
    #
    if get_version(pfile_a) != PFILE_VERSION:
        return None

    # open the file
    #
    try:
        fp = open(pfile_a, 'r')
    except:
        print("%s (%s: %s): error opening montage file (%s)" % \
            (sys.argv[0], __name__, "load_montage", pfile_a))

    # loop over all lines in the file
    #
    flag_pblock = False
    for line in fp:

        # clean up the line
        #
        check = line.replace(DELIM_NEWLINE, DELIM_NULL) \
                    .replace(DELIM_SPACE, DELIM_NULL)

        # throw away commented and blank lines
        #
        if check.startswith(DELIM_COMMENT) or len(check) == 0:
            pass

        # check for the beginning of a parameter block
        #
        elif check.startswith(keyword_a) and DELIM_BOPEN in check:
            flag_pblock = True

        # check for the end of a parameter block:
        #  note that we exit if we hit the end of the parameter block
        #
        elif flag_pblock and DELIM_BCLOSE in check:
            fp.close()
            return montage

        # otherwise, if the parameter block has started, decode a parameter
        #  by splitting and assigning to a dict
        #
        elif flag_pblock:
            parts = line.replace(DELIM_NEWLINE, DELIM_NULL).split(DELIM_EQUAL)
            montage[parts[0].strip()] = \
                [parts[1].split(DELIM_COLON)[0].strip(), parts[1].strip()]
            
    # make sure we found a block
    #
    if flag_pblock == False:
        fp.close()
        print("%s (%s: %s): invalid montage file (%s)" \
            % (sys.argv[0], __name__, "load_montage", pfile_a))
        return None

    # exit gracefully
    #
    return montage
#
# end of function

# function: get_version
#
# arguments:
#  fname: input filename
#
# return: a string containing the type
#
# this function opens a file, reads the magic sequence and returns
# the string.
#
def get_version(fname_a):
    
    # open the file
    #
    try: 
        fp = open(fname_a, "r") 
    except IOError: 
        print("%s (%s: %s): file not found (%s)" \
            % (sys.argv[0], __name__, "get_version", fname_a))
        return None

    # iterate over lines until we find the magic string
    #
    for line in fp:

        # remove spaces and newline chars
        #
        line = (line.lower()).replace(DELIM_SPACE, DELIM_NULL) \
                             .replace(DELIM_NEWLINE, DELIM_NULL)

        # break on the first non-comment and non-blank line
        #
        if (line.startswith(DELIM_COMMENT) == False) and len(line) != 0:
            break

    # close the file
    #
    fp.close()

    # split the line
    #
    parts = line.split(DELIM_EQUAL)

    # check the version
    #
    if (len(parts) == 0) or (parts[0] != DELIM_VERSION):
        return None

    # exit gracefully
    #
    return parts[1]
#                                                                          
# end of function

# function: generate_map
#                                                                          
# arguments:
#  pblock: a dictionary containing a parameter block
#                                              
# return:
#  pmap: a parameter file map
#
# This function converts a dictionary returned from load_parameters to
# a dictionary containing a parameter map. Note that is lowercases the
# map so that text is normalized.
#
def generate_map(pblock_a):

    # declare local variables
    #
    pmap = OrderedDict()

    # loop over the input, split the line and assign it to pmap
    #
    for key in pblock_a:
        lkey = key.lower()
        pmap[lkey] = pblock_a[key].split(DELIM_COMMA)
        pmap[lkey] = [x.lower() for x in pmap[lkey]]

    # exit gracefully
    #
    return pmap
#
# end of function

# function: permute_map
#                                                                          
# arguments:
#  map: the input map
#                                              
# return:
#  pmap: an inverted map
#
# this function permutes a map so symbol lookups can go fast.
#
def permute_map(map_a):

    # declare local variables
    #
    pmap = OrderedDict()

    # loop over the input map:
    #  note there is some redundancy here, but every event should
    #  have only one output symbol
    #
    for sym in map_a:
        for event in map_a[sym]:
            pmap[event] = sym
 
    # exit gracefully                                                     
    #                                                                      
    return pmap
#
# end of function

# function: map_events
#                                                                          
# arguments:
#  elist: a list of events
#  pmap: a permuted map (look up symbols to be converted)
#                                              
# return:
#  mlist: a list of mapped events
#
# this function maps event labels to mapped values.
#
def map_events(elist_a, pmap):

    # loop over the input list
    #
    mlist = []
    i = int(0)
    for event in elist_a:

        # copy the event
        #
        mlist.append([event[0], event[1], OrderedDict()]);

        # change the label
        #
        for key in event[2]:
            mlist[i][2][pmap[key]] = event[2][key]

        # increment the counter
        #
        i += int(1)
 
    # exit gracefully                                                     
    #                                                                      
    return mlist
#
# end of function

# function: make_fname
#                                                                          
# arguments:
#
#  odir: the output directory that will hold the file
#  fname: the output filename
#                                              
# return:
#  fname: a filename that is a concatenation of odir and fname
#
def make_fname(odir_a, fname_a):

    # string any trailing slashes
    #
    odir = odir_a
    if odir[-1] == DELIM_SLASH:
        odir = odir[:-1]

    # ceate the fill pathname
    #
    fname = odir + DELIM_SLASH + fname_a

    # exit gracefully                                                     
    #                                                                      
    return fname
#                                                                          
# end of function

# function: make_ofile
#
# arguments:
#  fname: input filename
#  ext: output file extension
#  odir: output directory
#  rdir: replace directory
#
# return: the output filename
#
# This method creates an output file name based on the input arguments
#
def make_ofile(fname_a, ext_a=NEDC_DEF_EXT, odir_a=NEDC_DEF_DIR, rdir_a=None):
        
    # get absolute file name
    #
    abs_name = os.path.abspath(os.path.realpath(
        os.path.expanduser(fname_a)))

    # replace extension with ext_a
    #
    ofile = os.path.join(os.path.dirname(abs_name),
                         os.path.basename(abs_name).split('.')[0]
                         + '.' + ext_a)

    # get absolute path of odir
    #
    odir = os.path.abspath(os.path.realpath(os.path.expanduser(odir_a)))

    # if the replace directory is valid and specified
    #
    if rdir_a is not None and rdir_a in ofile:

        # get absolute path of rdir
        #
        rdir = os.path.abspath(os.path.realpath(
            os.path.expanduser(rdir_a)))

        # replace the replace directory portion of path with 
        # the output directory
        #
        ofile = ofile.replace(rdir, odir)

    # if the replace directory is not valid or specified
    #
    else:

        # append basename of ofile to output directory
        #
        ofile = os.path.join(odir, os.path.basename(ofile))

    # exit gracefully
    #
    return ofile
#
# end of function
 
# function: make_fp
#                                                                          
# arguments:
#
#  fname: the filename
#                                              
# return:
#  fp: a file pointer
#
def make_fp(fname_a):

    # open the file
    #
    try:
        fp = open(fname_a, "w")
    except:
        print("%s (%s: %s): error opening summary file (%s)" % \
            (sys.argv[0], __name__, "make_fp", fname_a))
        return None
 
    # exit gracefully                                                     
    #                                                                      
    return fp
#
# end of function

# function: make_dir
#
# arguments:
#  path: new directory path (input)
#
# return: none
#
# This function emulates the Unix command "mkdir -p". It creates
# a directory tree, recursing through each level automatically.
# If the directory already exists, it continues past that level.
#
def make_dir(path_a):

    # use a system call to make a directory
    #
    try:
        os.makedirs(path_a)

    # if the directory exists, and error is thrown (and caught)
    #
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path_a):
            pass
        else: raise

    # exit gracefully
    #
    return True
#
# end of function

# function: get_fullpath
#
# arguments:
#  path: path to directory or file
#
# return: the full path to directory/file path argument
#
# This function returns the full pathname for a file. It expands
# environment variables.
#
def get_fullpath(path_a):
    return os.path.abspath(os.path.realpath(
        os.path.expanduser(os.path.expandvars(path_a))))
#
# end of function

#                                                                              
# end of file 

