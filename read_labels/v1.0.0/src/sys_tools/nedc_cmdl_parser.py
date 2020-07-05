#!/usr/bin/env python

# file: src/sys_tools/nedc_cmdl_parser.py
#
# usage:
#  import nedc_cmdl_parser as ncp
#  parser = ncp.CommandLineParser(usage_file_a, help_file_a)
#
#------------------------------------------------------------------------------

# import required system modules
#
import os
import sys
import argparse

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

# class: CommandLineParser
#
# This class inherits the argparse ArgumentParser object.
# This class is used to edit the format_help method found in the argparse
# module. It defines the help method for this specific tool.
#
class CommandLineParser(argparse.ArgumentParser):

    # method: Constructor
    #
    # arguments:
    #  usage_file: a short explanation of the command that is printed when 
    #              it is run without argument.
    #  help_file: a full explanation of the command that is printed when 
    #             it is run with -help argument.
    #
    # return: None
    #
    def __init__(self, usage_file_a, help_file_a):

        # declare class data
        #
        self.usage_file_d = usage_file_a
        self.help_file_d = help_file_a

        # declare an argument parser
        #
        argp_d = argparse.ArgumentParser.__init__(self)

        # exit gracefully
        #
        return None
    #                                                                          
    # end of method

    # method: set_usage
    #
    # arguments:
    #  fname: a file containing a one line usage message
    #
    # return: none
    #
    # This method is used to set the usage message. It is called
    # from the arg parser class.
    #
    def set_usage(self, fname_a):
        self.usage_file_d = fname_a
        return True
    #                                                                          
    # end of method

    # method: set_help
    #
    # arguments:
    #  fname: the full pathname of the help file
    #
    # return: a logical value indicating status
    #
    # This method is used to set the help message. It is called
    # from the arg parser class.
    #
    def set_help(self, fname_a):
        self.help_file_d = help_file_a
        return True
    #                                                                          
    # end of method

    # method: print_usage
    #
    # arguments: none
    #
    # return: a logical value indicating status
    #
    # This method is used to print the usage message from a file. Note that
    # this method must not return, so it exits directly.
    #
    def print_usage(self, file=sys.stderr):
        
        # open the file
        #
        try: 
            fp = open(self.usage_file_d, "r") 
        except IOError: 
            print("%s (%s: %s): file not found (%s)" \
                % (sys.argv[0], __name__, "print_usage", fname_a))
            return False

        # print the file
        #
        usage_file = fp.read()
        print(usage_file)

        # exit ungracefully
        #
        exit(-1)

    #                                                                          
    # end of method

    # method: format_help
    #
    # arguments: none
    #
    # return:
    #  help_file: string containing text from help file
    #
    # This class is used to define the specific help message to be used.
    #
    def format_help(self):

        # open the file
        #
        try: 
            fp = open(self.help_file_d, "r") 
        except IOError: 
            print("%s (%s: %s): file not found (%s)" \
                % (sys.argv[0], __name__, "format_help", self.help_file_d))
            return False

        # read the help file
        #
        help_file = fp.read()
        
        # close the file
        #
        fp.close()

        # exit gracefully
        #
        return help_file
    #                                                                          
    # end of method
#
# end of file
