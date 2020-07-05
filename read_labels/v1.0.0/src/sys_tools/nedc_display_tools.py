#!/usr/bin/env python
#
# file: src/sys_tools/nedc_display_tools.py
#
# usage:
#  import nedc_display_tools as ndt
#
# This class contains a collection of functions that provide functionality
# to display information in meaningful tables and other mediums.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#                                                                             
# imports are listed here                                                     
#                                                                             
#------------------------------------------------------------------------------

# import reqired system modules
#
import os
import sys

# import required NEDC modules
#
from . import nedc_file_tools as nft
from . import nedc_ann_tools as nat
from . import nedc_text_tools as ntt

#------------------------------------------------------------------------------
#                                                                              
# global variables are listed here                                             
#                                                                              
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

# function: format_hyp
#
# arguments:
#  ref: the references events as a list
#  hyp: the hypothesis events as a list
#
# return:
#  refo: a string displaying the alignment of the reference
#  hypo: a string displaying the alignment of the hypothesis
#  hits: the number of correct
#  subs: the number of substitution errors
#  inss: the number of insertion errors
#  dels: the number of deletion errors
#
# This function displays all the results in output report.
#
def format_hyp(ref_a, hyp_a):
    
    # declare return values
    #
    hits = int(0)
    subs = int(0)
    inss = int(0)
    dels = int(0)
    
    # find the max label length and increment by 1
    #
    maxl = int(0)
    for lbl in ref_a:
        if len(lbl) > maxl:
            maxl = len(lbl)
    for lbl in hyp_a:
        if len(lbl) > maxl:
            maxl = len(lbl)
    maxl += 1

    # loop over the input: skip the first and last label
    #
    refo = ""
    hypo = ""

    for i in range(1, len(ref_a)-1):

        # save a copy of the input
        #
        lbl_r = ref_a[i]
        lbl_h = hyp_a[i]

        # count the errors
        #
        if (ref_a[i] == ntt.NULL_CLASS) and (hyp_a[i] != ntt.NULL_CLASS):
            inss += int(1)
            lbl_h = hyp_a[i].upper()
        elif (ref_a[i] != ntt.NULL_CLASS) and (hyp_a[i] == ntt.NULL_CLASS):
            dels += int(1)
            lbl_r = ref_a[i].upper()
        elif (ref_a[i] != hyp_a[i]):
            subs += int(1)
            lbl_r = ref_a[i].upper()
            lbl_h = hyp_a[i].upper()
        else:
            hits += int(1)
            
        # append the strings
        #
        refo += ("%*s " % (maxl, lbl_r))
        hypo += ("%*s " % (maxl, lbl_h))

    # exit gracefully
    #
    return (refo, hypo, hits, subs, inss, dels)
#
# end of function

# function: create_table
#
# arguments:
#  cnf: confusion matrix
#
# return: 
#  header: header for the table
#  tbl: a table formatted for print_table
#
# This function transforms a confusion matrix into a format
# required for print_table.
#
def create_table(cnf_a):
        
    # declare local variables
    #
    tbl = []

    # create the header output by loop over the second index of the input
    #
    header = ['Ref/Hyp:']
    for key1 in cnf_a:
        for key2 in cnf_a[key1]:
            header.append(key2)
        break

    # loop over each key and then each row
    #
    counter = int(0)
    for key1 in cnf_a:

        # append the header
        #
        tbl.append([key1])

        # compute the sum of the entries in the row
        #
        sum = float(0)
        for key2 in cnf_a[key1]:
            sum += float(cnf_a[key1][key2])
        
        # transfer counts and percentages to the output table:
        #  note there is a chance the counts are zero due to a bad map
        #
        for key2 in cnf_a[key1]:
            if sum == 0:
                val1 = float(0.0)
                val2 = float(0.0)
            else:
                val1 = float(cnf_a[key1][key2])
                val2 = float(cnf_a[key1][key2]) / sum * 100.0
            tbl[counter].append([val1, val2])

        # increment the counter
        #
        counter += 1
            
    # exit gracefully
    #
    return header, tbl
#
# end of function

# function: print_table
#
# arguments: 
#  title: the title of the table
#  headers: the column headers
#  data: a list containing the row-by-row entries
#  fmt_lab: the format specification for a label (e..g, "%10s")
#  fmt_cnt: the format specification for the 1st value (e.g., "%8.2f")
#  fmt_pct: the format specification for the 2nd value (e.g., "%6.2f")
#  fp: the file pointer to write to
#
# return: a logical value indicating the status
#
# This function prints a table formatted in a relatively standard way.
# For example:
#
#     title = "This is the title"
#     headers = ["Ref/Hyp:", "Correct", "Incorrect"]
#     data = ["seiz:", [ 8.00, 53.33],  [7.00, 46.67]],\
#            ["bckg:", [18.00, 75.00],  [6.00, 25.00]],\
#            ["pled:", [ 0.00,  0.00],  [0.00,  0.00]]]
#
# results in this output:
#
#              This is the title             
#  Ref/Hyp:     Correct          Incorrect     
#     seiz:    8.00 ( 53.33%)    7.00 ( 46.67%)
#     bckg:   18.00 ( 75.00%)    6.00 ( 25.00%)
#     pled:    0.00 (  0.00%)    0.00 (  0.00%)
#
# fmt_lab is the format used for the row and column headings. fmt_cnt is
# used for the first number in each cell, which is usually an unnormalized
# number such as a count. fmt_pct is the format for the percentage.
#
def print_table(title_a, headers_a, data_a,
                fmt_lab_a, fmt_cnt_a, fmt_pct_a, fp_a):
    
    # get the number of rows and colums for the numeric data:
    #  the data structure contains two header rows
    #  the data structure contains one column for headers
    #
    nrows = len(data_a);
    ncols = len(headers_a) - 1

    # get the width of each colum and compute the total width:
    #  the width of the percentage column includes "()" and two spaces
    #
    width_lab = int(float(fmt_lab_a[1:-1]))
    width_cell = int(float(fmt_cnt_a[1:-1]))
    width_pct = int(float(fmt_pct_a[1:-1]))
    width_paren = 4
    total_width_cell = width_cell + width_pct + width_paren
    total_width_table = width_lab + \
                        ncols * (width_cell + width_pct + width_paren);

    # print the title
    #
    fp_a.write("%s".center(total_width_table - len(title_a)) % title_a)
    fp_a.write("\n")

    # print the first heading label right-aligned
    #
    fp_a.write("%*s" % (width_lab, headers_a[0]))

    # print the next ncols labels center-aligned:
    #  add a newline at the end
    #
    for i in range(1, ncols + 1):

        # compute the number of spaces needed to center-align
        #
        num_spaces = total_width_cell - len(headers_a[i])
        num_spaces_2 = int(num_spaces / 2)

        # write spaces, header, spaces
        #
        fp_a.write("%s" % " " * num_spaces_2)
        fp_a.write("%s" % headers_a[i])
        fp_a.write("%s" % " " * (num_spaces - num_spaces_2))
    fp_a.write("\n")

    # write the rows with numeric data:
    #  note that "%%" is needed to print a percent
    #
    fmt_str = fmt_cnt_a + " (" + fmt_pct_a + "%%)"

    for d in data_a:

        # write the row label
        #
        fp_a.write("%*s" % (width_lab, d[0] + nft.DELIM_COLON))

        # write the numeric data and then add a new line
        #
        for j in range(1,ncols + 1):
            fp_a.write(fmt_str % (d[j][0], d[j][1]))
        fp_a.write("\n")

    # exit gracefully
    #
    return True
#
# end of function

#
# end of file
