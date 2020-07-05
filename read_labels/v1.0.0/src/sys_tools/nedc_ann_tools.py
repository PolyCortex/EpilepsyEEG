#!/usr/bin/env python
#
# file: src/sys_tools/nedc_ann_tools.py
#
# usage:
#  import nedc_ann_tools as nat
#
# This class contains a collection of methods that provide 
# the infrastructure for processing annotation-related data.
#------------------------------------------------------------------------------

# import reqired system modules
#
import os
import sys
from collections import OrderedDict

# import required NEDC modules
#
from . import nedc_file_tools as nft
from . import nedc_text_tools as ntt

#------------------------------------------------------------------------------
#                                                                              
# global variables are listed here                                             
#                                                                              
#------------------------------------------------------------------------------

# define a data structure that encapsulates all file types:
#  we use this data structure to access lower-level objects. the key
#  is the type name, the first value is the magic sequence that should
#  appear in the file and the second value is the name of class data member
#  that is used to dynamically bind the subclass based on the type.
# 
#  we also need a list of supported versions for utilities to use.
#
FTYPE_LBL = 'lbl_v1.0.0'
FTYPE_TSE = 'tse_v1.0.0'
FTYPES = {'lbl': [FTYPE_LBL, 'lbl_d'], 'tse': [FTYPE_TSE, 'tse_d']} 
VERSIONS = [FTYPE_LBL, FTYPE_TSE]

#---
# define constants associated with the Annotation class
#

#---
# define constants associated with the Lbl class
#

# define a default montage file
#
DEFAULT_MAP_FNAME = None

# define symbols that appear as keys in an lbl file
#
DELIM_LBL_MONTAGE = 'montage'
DELIM_LBL_NUM_LEVELS = 'number_of_levels'
DELIM_LBL_LEVEL = 'level'
DELIM_LBL_SYMBOL = 'symbols'
DELIM_LBL_LABEL = 'label'

# define a list of characters we need to parse out
#
REM_CHARS = [nft.DELIM_BOPEN, nft.DELIM_BCLOSE, nft.DELIM_NEWLINE, \
             nft.DELIM_SPACE, nft.DELIM_QUOTE, nft.DELIM_SEMI, \
             nft.DELIM_SQUOTE]

#---
# define constants associated with the Tse class
#

#------------------------------------------------------------------------------
#
# classes are listed here:
#  there are four classes in this file arranged in this hierarchy
#   Annotation -> {Tse, Lbl} -> AnnotationGraph
#
#------------------------------------------------------------------------------

# class: AnnotationGraph
#
# This class implements the main data structure used to hold an annotation.
#
class AnnotationGraph:

    # method: constructor
    #
    # arguments: none
    #
    # return: none
    #
    def __init__(self):
        
        # declare a data structure to hold a graph
        #
        self.graph_d = OrderedDict()

        # exit gracefully
        #
        return None
    #
    # end of method

    # method: create
    #
    # arguments:
    #  lev: level of annotation
    #  sub: sublevel of annotation
    #  chan: channel of annotation
    #  start: start time of annotation
    #  stop: stop time of annotation
    #  symbols: dict of symbols/probabilities
    #
    # return: a logical value indicating status
    #
    # This method create an annotation in the AG data structure
    #
    def create(self, lev_a, sub_a, chan_a, start_a, stop_a, symbols_a):

        # try to access sublevel dict at level
        #
        try:
            self.graph_d[lev_a]
            
            # try to access channel dict at level/sublevel
            #
            try:
                self.graph_d[lev_a][sub_a]

                # try to append values to channel key in dict
                #
                try:
                    self.graph_d[lev_a][sub_a][chan_a] \
                        .append([start_a, stop_a, symbols_a])

                # if appending values failed, finish data structure
                #
                except:

                    # create empty list at chan_a key
                    #
                    self.graph_d[lev_a][sub_a][chan_a] = []

                    # append values
                    #
                    self.graph_d[lev_a][sub_a][chan_a] \
                        .append([start_a, stop_a, symbols_a])

            # if accessing channel dict failed, finish data structure
            #
            except:

                # create dict at level/sublevel
                #
                self.graph_d[lev_a][sub_a] = OrderedDict()

                # create empty list at chan_a
                #
                self.graph_d[lev_a][sub_a][chan_a] = []

                # append values
                #
                self.graph_d[lev_a][sub_a][chan_a] \
                    .append([start_a, stop_a, symbols_a])

        # if accessing sublevel failed, finish data structure
        #
        except:

            # create dict at level
            #
            self.graph_d[lev_a] = OrderedDict()

            # create dict at level/sublevel
            #
            self.graph_d[lev_a][sub_a] = OrderedDict()

            # create empty list at level/sublevel/channel
            #
            self.graph_d[lev_a][sub_a][chan_a] = []

            # append values
            #
            self.graph_d[lev_a][sub_a][chan_a] \
                .append([start_a, stop_a, symbols_a])

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: get
    #
    # arguments:
    #  level: level of annotations
    #  sublevel: sublevel of annotations
    # 
    # return: events by channel at level/sublevel
    #
    # This method returns the events stored at the level/sublevel argument
    #
    def get(self, level_a, sublevel_a, channel_a):
        
        # declare local variables
        #
        events = []

        # try to access graph at level/sublevel/channel
        #
        try:
            events = self.graph_d[level_a][sublevel_a][channel_a]

            # exit gracefully
            #
            return events

        # exit (un)gracefully: if failed, return False
        #
        except:
            print("%s (%s: %s): level/sublevel/channel not found (%d/%d/%d)" \
                % (sys.argv[0], __name__, "get", \
                   level_a, sublevel_a, channel_a))
            return False
    #
    # end of method

    # method: sort
    #
    # arguments: none
    #
    # return: a logical value indicating status
    #
    # This method sorts annotations by level, sublevel, 
    # channel, start, and stop times
    #
    def sort(self):

        # sort each level key by min value
        #
        self.graph_d = OrderedDict(
            sorted(list(self.graph_d.items())))

        # iterate over levels
        #
        for lev in self.graph_d:
            
            # sort each sublevel key by min value
            #
            self.graph_d[lev] = OrderedDict(
                sorted(list(self.graph_d[lev].items())))

            # iterate over sublevels
            #
            for sub in self.graph_d[lev]:

                # sort each channel key by min value
                #
                self.graph_d[lev][sub] = OrderedDict(
                    sorted(list(self.graph_d[lev][sub].items())))

                # iterate over channels
                #
                for chan in self.graph_d[lev][sub]:

                    # sort each list of labels by start and stop times
                    #
                    self.graph_d[lev][sub][chan] = sorted(
                        self.graph_d[lev][sub][chan], key=lambda x:
                        (x[0], x[1]))
                #
                # end of for
            #
            # end of for
        #
        # end of for
        
        # exit gracefully
        #
        return True
    #
    # end of method

    # method: add
    # 
    # arguments:
    #  dur: duration of events
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a logical value indicating status
    #
    # This method adds events of type sym.
    #
    def add(self, dur_a, sym_a, level_a, sublevel_a):

        # try to access level/sublevel
        #
        try:
            self.graph_d[level_a][sublevel_a]
        except:
            print("%s (%s: %s): level/sublevel not found (%d/%d)" \
                % (sys.argv[0], __name__, "add", level_a, sublevel_a))
            return False

        # variable to store what time in the file we are at
        #
        mark = 0.0

        # make sure events are sorted
        #
        self.sort()

        # iterate over channels at level/sublevel
        #
        for chan in self.graph_d[level_a][sublevel_a]:

            # reset list to store events
            #
            events = []

            # iterate over events at each channel
            #
            for event in self.graph_d[level_a][sublevel_a][chan]:

                # ignore if the start or stop time is past the duration
                #
                if (event[0] > dur_a) or (event[1] > dur_a):
                    pass

                # ignore if the start time is bigger than the stop time
                #
                elif event[0] > event[1]:
                    pass

                # ignore if the start time equals the stop time
                #
                elif event[0] == event[1]:
                    pass

                # if the beginning of the event is not at the mark
                #
                elif event[0] != mark:

                    # create event from mark->starttime
                    #
                    events.append([mark, event[0], {sym_a: 1.0}])
                
                    # add event after mark->starttime
                    #
                    events.append(event)

                    # set mark to the stop time
                    #
                    mark = event[1]
                
                # if the beginning of the event is at the mark
                #
                else:
                
                    # store this event
                    #
                    events.append(event)

                    # set mark to the stop time
                    #
                    mark = event[1]
            #
            # end of for
    
            # after iterating through all events, if mark is not at dur_a
            #
            if mark != dur_a:

                # create event from mark->dur_a
                #
                events.append([mark, dur_a, {sym_a: 1.0}])

            # store events as the new events in self.graph_d
            #
            self.graph_d[level_a][sublevel_a][chan] = events
        #
        # end of for

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: delete
    # 
    # arguments:
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a logical value indicating status
    #
    # This method deletes events of type sym
    #
    def delete(self, sym_a, level_a, sublevel_a):

        # try to access level/sublevel
        #
        try:
            self.graph_d[level_a][sublevel_a]
        except:
            print("%s (%s: %s): level/sublevel not found (%d/%d)" \
                % (sys.argv[0], __name__, "delete", level_a, sublevel_a))
            return False

        # iterate over channels at level/sublevel
        #
        for chan in self.graph_d[level_a][sublevel_a]:

            # get events at chan
            #
            events = self.graph_d[level_a][sublevel_a][chan]

            # keep only the events that do not contain sym_a
            #
            events = [e for e in events if sym_a not in list(e[2].keys())]

            # store events in self.graph_d
            #
            self.graph_d[level_a][sublevel_a][chan] = events
        #
        # end of for

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: get_graph
    #
    # arguments: none
    #
    # return: entire graph data structure
    #
    # This method returns the entire graph, instead of a 
    # level/sublevel/channel.
    #
    def get_graph(self):
        return self.graph_d
    #
    # end of method

    # method: set_graph
    #
    # arguments:
    #  graph: graph to set
    #
    # return: a logical value indicating status
    #
    # This method sets the class data to graph_a.
    #
    def set_graph(self, graph_a):
        self.graph_d = graph_a
        return True
    #
    # end of method
#
# end of class

# class: Tse
#
# This class contains methods to manipulate time-synchronous event files.
#
class Tse:
 
    # method: constructor
    #
    # arguments: none
    #
    # return: none
    #
    def __init__(self):

        # declare Graph object, to store annotations
        #
        self.graph_d = AnnotationGraph()

        # exit gracefully
        #
        return None
    #
    # end of method

    # method: load
    #
    # arguments:
    #  fname: annotation filename
    #
    # return: a logical value indicating status
    #
    # This method loads an annotation from a file.
    #
    def load(self, fname_a):

        # open file
        #
        with open(fname_a, 'r') as fp:

            # loop over lines in file
            #
            for line in fp:

                # clean up the line
                #
                line = line.replace(nft.DELIM_NEWLINE, nft.DELIM_NULL) \
                           .replace(nft.DELIM_CARRIAGE, nft.DELIM_NULL)
                check = line.replace(nft.DELIM_SPACE, nft.DELIM_NULL)

                # throw away commented, blank lines, version lines
                #
                if check.startswith(nft.DELIM_COMMENT) or \
                   check.startswith(nft.DELIM_VERSION) or \
                   len(check) == 0:
                    continue

                # split the line
                #
                val = OrderedDict()
                parts = line.split()

                try:
                    # loop over every part, starting after start/stop times
                    #
                    for i in range(2, len(parts), 2):

                        # create dict with label as key, prob as value
                        #
                        val[parts[i]] = float(parts[i+1])

                    # create annotation in AG
                    #
                    self.graph_d.create(int(0), int(0), int(-1), \
                                        float(parts[0]), float(parts[1]), val)
                except:
                    print("%s (%s: %s): invalid annotation (%s)" \
                        % (sys.argv[0], __name__, "load", line))
                    return False
            #
            # end of for
        #
        # end of with

        # make sure graph is sorted after loading
        #
        self.graph_d.sort

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: get
    #
    # arguments:
    #  level: level of annotations to get
    #  sublevel: sublevel of annotations to get
    #
    # return: events at level/sublevel by channel
    #
    # This method gets the annotations stored in the AG at level/sublevel.
    #
    def get(self, level_a, sublevel_a, channel_a):
        events = self.graph_d.get(level_a, sublevel_a, channel_a)
        return events
    #
    # end of method

    # method: display
    #
    # arguments: 
    #  level: level of events
    #  sublevel: sublevel of events
    #  fp: a file pointer
    #
    # return: a logical value indicating status
    #
    # This method displays the events from a flat AG.
    #
    def display(self, level_a, sublevel_a, fp_a = sys.stdout):
        
        # get graph
        #
        graph = self.get_graph()

        # try to access graph at level/sublevel
        #
        try:
            graph[level_a][sublevel_a]
        except:
            sys.stdout.write("%s (%s: %s): level/sublev not in graph (%d/%d)" \
                             % (sys.argv[0], __name__, "display", \
                                level_a, sublevel_a))
            return False

        # iterate over channels at level/sublevel
        #
        for chan in graph[level_a][sublevel_a]:

            # iterate over events for each channel
            #
            for event in graph[level_a][sublevel_a][chan]:
                start = event[0]
                stop = event[1]

                # create a string with all symb/prob pairs
                #
                pstr = ""
                for symb in event[2]:
                    pstr += " %8s %10.4f" % (symb, event[2][symb])

                # display event
                #
                fp_a.write("%10s: %10.4f %10.4f%s\n" % \
                           ('ALL', start, stop, pstr))
            #
            # end of for
        #
        # end of for
            
        # exit gracefully
        #
        return True
    #
    # end of method

    # method: write
    #
    # arguments:
    #  ofile: output file path to write to
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a logical value indicating status
    #
    # This method writes the events to a .tse file
    #
    def write(self, ofile_a, level_a, sublevel_a):

        # make sure graph is sorted
        #
        self.graph_d.sort()

        # get graph
        #
        graph = self.get_graph()

        # try to access the graph at level/sublevel
        #
        try:
            graph[level_a][sublevel_a]
        except:
            print("%s (%s: %s): level/sublevel not in graph (%d/%d)" \
                % (sys.argv[0], __name__, "write", level_a, sublevel_a))
            return False

        # list to collect all events
        #
        events = []

        # iterate over channels at level/sublevel
        #
        for chan in graph[level_a][sublevel_a]:

            # iterate over events for each channel
            #
            for event in graph[level_a][sublevel_a][chan]:

                # store every channel's events in one list
                #
                events.append(event)
            #
            # end of for
        #
        # end of for

        # remove any events that are not unique
        #
        events = get_unique_events(events)

        # open file with write
        #
        with open(ofile_a, 'w') as fp:

            # write version
            #
            fp.write("version = %s\n" % FTYPES['tse'][0])
            fp.write("\n")

            # iterate over events
            #
            for event in events:

                # create symb/prob string from dict
                #
                pstr = ""
                for symb in event[2]:
                    pstr += " %s %.4f" % (symb, event[2][symb])

                # write event
                #
                fp.write("%.4f %.4f%s\n" % (event[0], event[1], pstr))
            #
            # end of for
        #
        # end of with

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: add
    # 
    # arguments:
    #  dur: duration of events
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a logical value indicating status
    #
    # This method adds events of type sym.
    #
    def add(self, dur_a, sym_a, level_a, sublevel_a):
        return self.graph_d.add(dur_a, sym_a, level_a, sublevel_a)
    #
    # end of method

    # method: delete
    # 
    # arguments:
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a logical value indicating status
    #
    # This method deletes events of type sym.
    #
    def delete(self, sym_a, level_a, sublevel_a):
        return self.graph_d.delete(sym_a, level_a, sublevel_a)
    #
    # end of method

    # method: get_graph
    #
    # arguments: none
    # 
    # return: entire graph data structure
    #
    # This method accesses self.graph_d and returns the entire graph structure.
    #
    def get_graph(self):
        return self.graph_d.get_graph()
    #
    # end of method

    # method: set_graph
    #
    # arguments:
    #  graph: graph to set
    #
    # return: a logical value indicating status
    #
    # This method sets the class data to graph_a.
    #
    def set_graph(self, graph_a):
        return self.graph_d.set_graph(graph_a)
    #
    # end of method
#
# end of class

# class: Lbl
# 
# This class implements methods to manipulate label files.
#
class Lbl:
    
    # method: constructor
    #
    # arguments: none
    #
    # return: none
    #
    # This method constructs Ag
    #
    def __init__(self):
        
        # declare variables to store info parsed from lbl file
        #
        self.chan_map_d = OrderedDict({int(-1): 'all'})
        self.montage_lines_d = []
        self.symbol_map_d = OrderedDict()
        self.num_levels_d = int(1)
        self.num_sublevels_d = OrderedDict({int(0): int(1)})

        # declare AG object to store annotations
        #
        self.graph_d = AnnotationGraph()

        # exit gracefully
        #
        return None
    #
    # end of method

    # method: load
    #
    # arguments:
    #  fname: annotation filename
    #
    # return: a logical value indicating status
    #
    # This method loads an annotation from a file.
    #
    def load(self, fname_a):

        # open file
        #
        fp = open(fname_a, 'r')

        # loop over lines in file
        #
        for line in fp:

            # clean up the line
            #
            line = line.replace(nft.DELIM_NEWLINE, nft.DELIM_NULL) \
                   .replace(nft.DELIM_CARRIAGE, nft.DELIM_NULL)

            # parse a single montage definition
            #
            if line.startswith(DELIM_LBL_MONTAGE):
                try:
                    chan_num, name, montage_line = \
                        self.parse_montage(line)
                    self.chan_map_d[chan_num] = name
                    self.montage_lines_d.append(montage_line)
                except:
                    print("%s (%s: %s): error parsing montage (%s)" \
                        % (sys.argv[0], __name__, "load", line))
                    fp.close()
                    return False
            
            # parse the number of levels
            #
            elif line.startswith(DELIM_LBL_NUM_LEVELS):
                try:
                    self.num_levels_d = self.parse_numlevels(line)
                except:
                    print("%s (%s: %s): error parsing number of levels (%s)" \
                        % (sys.argv[0], __name__, "load", line))
                    fp.close()
                    return False

            # parse the number of sublevels at a level
            #
            elif line.startswith(DELIM_LBL_LEVEL):
                try:
                    level, sublevels = self.parse_numsublevels(line)
                    self.num_sublevels_d[level] = sublevels
                    
                except:
                    print("%s (%s: %s): error parsing num of sublevels (%s)" \
                        % (sys.argv[0], __name__, "load", line))
                    fp.close()
                    return False

            # parse symbol definitions at a level
            #
            elif line.startswith(DELIM_LBL_SYMBOL):
                try:
                    level, mapping = self.parse_symboldef(line)
                    self.symbol_map_d[level] = mapping
                except:
                    print("%s (%s: %s): error parsing symbols (%s)" \
                        % (sys.argv[0], __name__, "load", line))
                    fp.close()
                    return False

            # parse a single label
            #
            elif line.startswith(DELIM_LBL_LABEL):
                try:
                    lev, sub, start, stop, chan, symbols = \
                        self.parse_label(line)
                except:
                    print("%s (%s: %s): error parsing label (%s)" \
                        % (sys.argv[0], __name__, "load", line))
                    fp.close()
                    return False

                # create annotation in AG
                #
                status = self.graph_d.create(lev, sub, chan, \
                                             start, stop, symbols)
        #
        # end of for

        # close file
        #
        fp.close()

        # sort labels after loading
        #
        self.graph_d.sort()

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: get
    #
    # arguments:
    #  level: level value
    #  sublevel: sublevel value
    #
    # return: events by channel from AnnotationGraph
    #
    # This method returns the events at level/sublevel
    #
    def get(self, level_a, sublevel_a, channel_a):

        # get events from AG
        #
        events = self.graph_d.get(level_a, sublevel_a, channel_a)

        # exit gracefully
        #
        return events
    #
    # end of method

    # method: display
    #
    # arguments:
    #  level: level of events
    #  sublevel: sublevel of events
    #  fp: a file pointer
    #
    # return: a logical value indicating status
    #
    # This method displays the events from a flat AG
    #
    def display(self, level_a, sublevel_a, fp_a = sys.stdout):

        # get graph
        #
        graph = self.get_graph()

        # try to access level/sublevel
        #
        try:
            graph[level_a][sublevel_a]
        except:
            sys.stdout.write("%s (%s: %s): level/sublevel not found (%d/%d)" \
                             % (sys.argv[0], __name__, "display", \
                                level_a, sublevel_a))
            return False

        # iterate over channels at level/sublevel
        #
        for chan in graph[level_a][sublevel_a]:

            # iterate over events at chan
            #
            for event in graph[level_a][sublevel_a][chan]:

                # find max probability
                #
                max_prob = max(event[2].values())

                # iterate over symbols in dictionary
                #
                for symb in event[2]:
                    
                    # if the value of the symb equals the max prob
                    #
                    if event[2][symb] == max_prob:

                        # set max symb to this symbol
                        #
                        max_symb = symb
                        break
                    #
                    # end of if
                #
                # end of for
                    
                # display event
                #
                fp_a.write("%10s: %10.4f %10.4f %8s %10.4f\n" % \
                           (self.chan_map_d[chan], event[0], event[1], \
                            max_symb, max_prob))
            #
            # end of for
        #
        # end of for

        # exit gracefully
        #
        return True
    #
    # end of method
    
    # method: parse_montage
    #
    # arguments: 
    #  line: line from label file containing a montage channel definition
    #
    # return:
    #  channel_number: an integer containing the channel map number
    #  channel_name: the channel name corresponding to channel_number
    #  montage_line: entire montage def line read from file
    #
    # This method parses a montage line into it's channel name and number.
    # Splitting a line by two values easily allows us to get an exact 
    # value/string from a line of definitions
    #
    def parse_montage(self, line_a):

        # split between '=' and ',' to get channel number
        #
        channel_number = int(line_a.split(nft.DELIM_EQUAL)[1] \
                             .split(nft.DELIM_COMMA)[0].strip())
    
        # split between ',' and ':' to get channel name
        #
        channel_name = line_a.split(nft.DELIM_COMMA)[1] \
                             .split(nft.DELIM_COLON)[0].strip()

        # remove chars from montage line
        #
        montage_line = line_a.strip().strip(nft.DELIM_NEWLINE)

        # exit gracefully
        #
        return [channel_number, channel_name, montage_line]
    #
    # end of method

    # method: parse_numlevels
    #
    # arguments:
    #  line: line from label file containing the number of levels
    #
    # return: an integer containing the number of levels defined in the file
    #
    # This method parses the number of levels in a file.
    #
    def parse_numlevels(self, line_a):

        # split by '=' and remove extra characters
        #
        return int(line_a.split(nft.DELIM_EQUAL)[1].strip())
    #
    # end of method
        
    # method: parse_numsublevels
    #
    # arguments:
    #  line: line from label file containing number of sublevels in level
    #
    # return:
    #  level: level from which amount of sublevels are contained
    #  sublevels: amount of sublevels in particular level
    #
    # This method parses the number of sublevels per level in the file
    #
    def parse_numsublevels(self, line_a):

        # split between '[' and ']' to get level
        #
        level = int(line_a.split(nft.DELIM_OPEN)[1] \
                    .split(nft.DELIM_CLOSE)[0].strip())

        # split by '=' and remove extra characters
        #
        sublevels = int(line_a.split(nft.DELIM_EQUAL)[1].strip())

        # exit gracefully
        #
        return [level, sublevels]
    #
    # end of method

    # method: parse_symboldef
    #
    # arguments:
    #  line: line from label fiel containing symbol definition for a level
    #
    # return:
    #  level: an integer containing the level of this symbol definition
    #  mappings: a dict containing the mapping of symbols for this level
    #
    # This method parses a symbol definition line into a specific level, 
    # the corresponding symbol mapping as a dictionary.
    #
    def parse_symboldef(self, line_a):

        # split by '[' and ']' to get level of symbol map
        #
        level = int(line_a.split(nft.DELIM_OPEN)[1].split(nft.DELIM_CLOSE)[0])

        # remove all characters to remove, and split by ','
        #
        syms = ''.join(c for c in line_a.split(nft.DELIM_EQUAL)[1] \
                       if c not in REM_CHARS)

        symbols = syms.split(nft.DELIM_COMMA)

        # create a dict from string, split by ':'
        #   e.g. '0: seiz' -> mappings[0] = 'seiz'
        #
        mappings = OrderedDict()
        for s in symbols:
            mappings[int(s.split(':')[0])] = s.split(':')[1]

        # exit gracefully
        #
        return [level, mappings]
    #
    # end of method

    # method: parse_label
    #
    # arguments:
    #  line: line from label file containing an annotation label
    #
    # return: all information read from .ag file
    #
    # this method parses a label definition into the values found in the label
    #
    def parse_label(self, line_a):

        # dict to store symbols/probabilities
        #
        symbols = OrderedDict()

        # remove characters to remove, and split data by ','
        #
        line = ''.join(c for c in line_a.split(nft.DELIM_EQUAL)[1] \
                       if c not in REM_CHARS)

        data = line.split(nft.DELIM_COMMA)

        # separate data into specific variables
        #
        level = int(data[0])
        sublevel = int(data[1])
        start = float(data[2])
        stop = float(data[3])

        # the channel value supports either 'all' or channel name
        #
        try:
            channel = int(data[4])
        except:
            channel = int(-1)

        # parse probabilities
        #
        probs = line.split(nft.DELIM_OPEN)[1] \
                    .strip(nft.DELIM_CLOSE).split(nft.DELIM_COMMA)

        # set every prob in probs to type float
        #
        probs = list(map(float, probs))

        # iterate over symbols
        #
        for i in range(len(list(self.symbol_map_d[level].keys()))):

            if probs[i] > 0.0:

                # set each symbol equal to the corresponding probability
                #
                symbols[list(self.symbol_map_d[level].values())[i]] = probs[i]

        # exit gracefully
        #
        return [level, sublevel, start, stop, channel, symbols]
    #
    # end of method

    # method: write
    #
    # arguments:
    #  ofile: output file path to write to
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a logical value indicating status
    #
    # This method writes events to a .lbl file.
    #
    def write(self, ofile_a, level_a, sublevel_a):

        # make sure graph is sorted
        #
        self.graph_d.sort()

        # get graph
        #
        graph = self.get_graph()

        # try to access graph at level/sublevel
        #
        try:
            graph[level_a][sublevel_a]
        except:
            print("%s (%s: %s): level/sublevel not found (%d/%d)" \
                % (sys.argv[0], __name__, "write", level_a, sublevel_a))
            return False

        # open file with write
        #
        with open(ofile_a, 'w') as fp:
                
            # write version
            #
            fp.write("\n")
            fp.write("version = %s\n" % FTYPES['lbl'][0])
            fp.write("\n")
            
            # if montage_lines is blank, we are converting from tse to lbl.
            # 
            # create symbol map from tse symbols
            #
            if len(self.montage_lines_d) == 0:

                # variable to store the number of symbols
                #
                num_symbols = 0

                # create a dictionary at level 0 of symbol map
                #
                self.symbol_map_d[int(0)] = OrderedDict()

                # iterate over all events stored in the 'all' channels
                #
                for event in graph[level_a][sublevel_a][int(-1)]:

                    # iterate over symbols in each event
                    #
                    for symbol in event[2]:

                        # if the symbol is not in the symbol map
                        #
                        if symbol not in list(self.symbol_map_d[0].values()):

                            # map num_symbols interger to symbol
                            #
                            self.symbol_map_d[0][num_symbols] = symbol

                            # increment num_symbols
                            #
                            num_symbols += 1
                        #
                        # end of if
                    #
                    # end of for
                #
                # end of for
            #
            # end of if
                    
            # write montage lines
            #
            for line in self.montage_lines_d:
                fp.write("%s\n" % line)

            fp.write("\n")

            # write number of levels
            #
            fp.write("number_of_levels = %d\n" % self.num_levels_d)
            fp.write("\n")

            # write number of sublevels
            #
            for lev in self.num_sublevels_d:
                fp.write("level[%d] = %d\n" % \
                         (lev, self.num_sublevels_d[lev]))
            fp.write("\n")
            
            # write symbol definitions
            #
            for lev in self.symbol_map_d:
                fp.write("symbols[%d] = %s\n" % \
                         (lev, str(self.symbol_map_d[lev])))
            fp.write("\n")

            # iterate over channels at level/sublevel
            #
            for chan in graph[level_a][sublevel_a]:

                # iterate over events in chan
                #
                for event in graph[level_a][sublevel_a][chan]:
                    
                    # create string for probabilities
                    #
                    pstr = "["

                    # iterate over symbol map
                    #
                    for symb in list(self.symbol_map_d[level_a].values()):

                        # if the symbol is found in the event
                        #
                        if symb in event[2]:
                            pstr += (str(event[2][symb]) + ', ')
                        else:
                            pstr += '0.0, '
                    #
                    # end of for

                    # remove the ', ' from the end of pstr
                    #
                    pstr = pstr[:len(pstr) - 2] + "]}"
                    
                    # write event
                    #
                    fp.write("label = {%d, %d, %.4f, %.4f, %s, %s\n" \
                             % (level_a, sublevel_a, event[0], 
                                event[1], chan, pstr))
                #
                # end of for
            #
            # end of for
        #
        # end of with

        # exit gracefully
        #
        return True
    #
    # end of method

    # method: add
    # 
    # arguments:
    #  dur: duration of events
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a logical value indicating status
    #
    # This method adds events of type sym
    #
    def add(self, dur_a, sym_a, level_a, sublevel_a):
        return self.graph_d.add(dur_a, sym_a, level_a, sublevel_a)

    # method: delete
    # 
    # arguments:
    #  sym: symbol of events
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a logical value indicating status
    #
    # This method deletes events of type sym
    #
    def delete(self, sym_a, level_a, sublevel_a):
        return self.graph_d.delete(sym_a, level_a, sublevel_a)

    # method: get_graph
    #
    # arguments: none
    # 
    # return: entire graph data structure
    #
    # This method accesses self.graph_d and returns the entire graph structure.
    #
    def get_graph(self):
        return self.graph_d.get_graph()
    #
    # end of method

    # method: set_graph
    #
    # arguments:
    #  graph: graph to set
    #
    # return: a logical value indicating status
    #
    # This method sets the class data to graph_a
    #
    def set_graph(self, graph_a):
        return self.graph_d.set_graph(graph_a)
    #
    # end of method
#
# end of class

# class: Annotations
#
# This class is the main class of this file. It contains methods to 
# manipulate the set of supported annotation file formats including
# label (.lbl) and time-synchronous events (.tse) formats.
#
class Annotations:

    # method: constructor
    #
    # arguments: none
    #
    # return: none
    #
    # This method constructs Annotations
    #
    def __init__(self):
        
        # declare variables for each type of file:
        #  these variable names must match the FTYPES declaration.
        #
        self.tse_d = Tse()
        self.lbl_d = Lbl()

        # declare variable to store type of annotations
        #
        self.type_d = None

        # exit gracefully
        #
        return None
    #
    # end of method
        
    # method: load
    #
    # arguments:
    #  fname: annotation filename
    #
    # return: a logical value indicating status
    #
    # This method loads an annotation from a file.
    #
    def load(self, fname_a):

        # reinstantiate objects, this removes the previous loaded annotations
        #
        self.lbl_d = Lbl()
        self.tse_d = Tse()

        # determine the file type
        #
        magic_str = nft.get_version(fname_a)
        self.type_d = self.check_version(magic_str)
        if self.type_d == None:
            print("%s (%s: %s): unknown file type (%s: %s)" % \
            (sys.argv[0], __name__, "load", fname_a, magic_str))
            return False
        
        # load the specific type
        #
        return getattr(self, FTYPES[self.type_d][1]).load(fname_a)
    #
    # end of method

    # method: get
    #
    # arguments:
    #  level: the level value
    #  sublevel: the sublevel value
    #
    # return:
    #  events: a list of ntuples containing the start time, stop time,
    #          a label and a probability.
    #
    # This method returns a flat data structure containing a list of events.
    #
    def get(self, level_a = int(0), sublevel_a = int(0), channel_a = int(-1)):

        if self.type_d is not None:
            events = getattr(self, FTYPES[self.type_d][1]) \
                .get(level_a, sublevel_a, channel_a)
        else:
            sys.stdout.write("%s (%s: %s): no annotation loaded" % \
                             (sys.argv[0], __name__, "get"))
            return False

        # exit gracefully
        #
        return events
    #
    # end of method

    # method: display
    #
    # arguments:
    #  level: level value
    #  sublevel: sublevel value
    #  fp: a file pointer (default = stdout)
    #
    # return: a logical value indicating status
    #
    # This method displays the events at level/sublevel.
    #
    def display(self, level_a = int(0), sublevel_a = int(0), \
                fp_a = sys.stdout):

        if self.type_d is not None:

            # display events at level/sublevel
            #
            status = getattr(self, FTYPES[self.type_d][1]) \
                .display(level_a, sublevel_a, fp_a)
                                                               
        else:
            sys.stdout.write("%s: (%s: %s): no annotations to display" % \
                             (sys.argv[0], __name__, "display"))
            return False

        # exit gracefully
        #
        return status
    #
    # end of method
    
    # method: write
    #
    # arguments:
    #  ofile: output file path to write to
    #  level: level of annotation to write
    #  sublevel: sublevel of annotation to write
    #
    # return: a logical value indicating status
    #
    # This method writes annotations to a specified file.
    #
    def write(self, ofile_a, level_a = int(0), sublevel_a = int(0)):

        # write events at level/sublevel
        #
        if self.type_d is not None:
            status = getattr(self, FTYPES[self.type_d][1]) \
                         .write(ofile_a, level_a, sublevel_a)
        else:
            sys.stdout.write("%s (%s: %s): no annotations to write" % \
                             (sys.argv[0], __name__, "write"))
            status = False

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: add
    #
    # arguments:
    #  dur: duration of file
    #  sym: symbol of event to be added
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a logical value indicating status
    #
    # This method adds events to the current events based on args.
    #
    def add(self, dur_a, sym_a, level_a, sublevel_a):
        
        # add labels to events at level/sublevel
        #
        if self.type_d is not None:
            status = getattr(self, FTYPES[self.type_d][1]) \
                .add(dur_a, sym_a, level_a, sublevel_a,)
        else:
            print("%s (%s: %s): no annotations to add to" \
                % (sys.argv[0], __name__, "add"))
            status = False
        
        # exit gracefully
        #
        return status
    #
    # end of method

    # method: delete
    #
    # arguments:
    #  sym: symbol of event to be deleted
    #  level: level of events
    #  sublevel: sublevel of events
    #
    # return: a logical value indicating status
    #
    # This method deletes all events of type sym
    #
    def delete(self, sym_a, level_a, sublevel_a):

        # delete labels from events at level/sublevel
        #
        if self.type_d is not None:
            status = getattr(self, FTYPES[self.type_d][1]) \
                .delete(sym_a, level_a, sublevel_a)
        else:
            print("%s (%s: %s): no annotations to delete" \
                % (sys.argv[0], __name__, "delete"))
            status = False

        # exit gracefully
        #
        return status
    #
    # end of method

    # method: set_type
    #
    # arguments: 
    #  type: type of ann object to set
    #
    # return: a logical value indicating status
    #
    # This method sets the type and graph in type_a from self.type_d
    # 
    def set_type(self, type_a):

        # set the graph of type_a to the graph of self.type_d
        #
        if self.type_d is not None:
            status = getattr(self, FTYPES[type_a][1]) \
                .set_graph(getattr(self, FTYPES[self.type_d][1]).get_graph())
            self.type_d = type_a
        else:
            print("%s (%s: %s): no graph to set" \
                % (sys.argv[0], __name__, "set_graph"))
            status = False
        
        # exit gracefully
        #
        return status
    #
    # end of method

    # method: get_graph
    #
    # arguments: none
    #
    # return: the entire annotation graph
    #
    # This method returns the entire stored annotation graph
    #
    def get_graph(self):
        
        # if the graph is valid, get it
        #
        if self.type_d is not None:
            graph = getattr(self, FTYPES[self.type_d][1]).get_graph()
        else:
            print("%s (%s: %s): no graph to get" \
                % (sys.argv[0], __name__, "get_graph"))
            graph = None
            
        # exit gracefully
        #
        return graph
    #
    # end of method

    # method: check_version
    #
    # arguments:
    #  magic: a magic sequence
    #
    # return: a character string containing the name of the type
    #
    def check_version(self, magic_a):

        # check for a match
        #
        for key in FTYPES:
            if FTYPES[key][0] == magic_a:
                return key

        # exit (un)gracefully:
        #  if we get this far, there was no match
        #
        return False
    #
    # end of method
#
# end of class

#------------------------------------------------------------------------------
#
# functions listed here
#
#------------------------------------------------------------------------------

# function: get_unique_events
#
# arguments:
#  events: events to aggregate
#
# return: a list of unique events
#
# This method combines events if they are of the same start/stop times.
#
def get_unique_events(events_a):

    # list to store unique events
    #
    events = []

    # make sure events_a are sorted
    #
    events_a = sorted(events_a, key=lambda x: (x[0], x[1]))

    # loop until we have checked all events_a
    #
    while len(events_a) != 0:

        # reset flag
        #
        is_unique = True
        n_start = int(-1)
        n_stop = int(-1)

        # get this event's start/stop times
        #
        start = events_a[0][0]
        stop = events_a[0][1]

        # if we are not at the last event
        #
        if len(events_a) != 1:

            # get next event's start/stop times
            #
            n_start = events_a[1][0]
            n_stop = events_a[1][1]

        # if this event's start/stop times are the same as the next event's,
        #  (only do this if we are not at the last event)
        #
        if (n_start == start) and (n_stop == stop) and (len(events_a) != 1):

            # combine this event's dict with the next event's symbol dict
            #
            for symb in events_a[1][2]:

                # if the symb is not found in this event's dict
                #
                if symb not in events_a[0][2]:

                    # add symb to this event's dict
                    #
                    events_a[0][2][symb] = events_a[1][2][symb]

                # else if the symb in the next event has a higher prob
                #
                elif events_a[1][2][symb] > events_a[0][2][symb]:

                    # update this event's symb with prob from the next event
                    #
                    events_a[0][2][symb] = events_a[1][2][symb]
                #
                # end of if/elif
            #
            # end of for

            # delete the next event, it is not unique
            #
            del events_a[1]
        #
        # end of if

        # loop over unique events
        #
        for unique in events:

            # if the start/stop times of this event is found in unique events
            #
            if (start == unique[0]) and (stop == unique[1]):

                # combine unique event's dict with this event's dict:
                #  iterate over symbs in this event's dict
                #
                for symb in events_a[0][2]:

                    # if the symb is not found in the unique event's dict
                    #
                    if symb not in unique[2]:

                        # add symb to the unique event's dict
                        #
                        unique[2][symb] = events_a[0][2][symb]

                    # else if the symb in this event has a higher prob
                    #
                    elif events_a[0][2][symb] > unique[2][symb]:

                        # update unique event's symb with prob from this event
                        #
                        unique[2][symb] = events_a[0][2][symb]
                    #
                    # end of if/elif
                #
                # end of for

                # delete this event, it is not unique
                #
                del events_a[0]
                is_unique = False
                break
            #
            # end of if
        #
        # end of for

        # if this event is still unique
        #
        if is_unique is True:

            # add this event to the unique events
            #
            events.append(events_a[0])

            # delete this event, it is now stored as unique
            #
            del events_a[0]
        #
        # end of if
    #
    # end of while

    # exit gracefully
    #
    return events
#
# end of function

# function: compare_durations
#
# arguments:
#  l1: the first list of files
#  l2: the second list of files
#
# return: a logical value indicating status
#
# This method goes through two lists of files and compares the durations
# of the annotations. If they don't match, it returns false.
#
def compare_durations(l1_a, l2_a):

    # create an annotation object
    #
    ann = Annotations()

    # check the length of the lists
    #
    if len(l1_a) != len(l2_a):
        return False

    # loop over the lists together
    #
    for l1, l2 in zip(l1_a, l2_a):

        # load the annotations for l1
        #
        if ann.load(l1) == False:
            print("%s (%s: %s): error loading annotation for file (%s)" % \
                (sys.argv[0], __name__, "check_durations", l1))
            return False

        # get the events for l1
        #
        events_l1 = ann.get()
        if events_l1 == None:
            print("%s (%s: %s): error getting annotation (%s)" % \
                (sys.argv[0], __name__, "check_durations", l1))
            return False

        # load the annotations for l2
        #
        if ann.load(l2) == False:
            print("%s (%s: %s): error loading annotation for file (%s)" % \
                (sys.argv[0], __name__, "check_durations", l2))
            return False

        # get the events for l2
        #
        events_l2 = ann.get()
        if events_l2 == None:
            print("%s (%s: %s): error getting annotation (%s)" % \
                (sys.argv[0], __name__, "score", l2))
            return False

        # check the durations
        #
        if round(events_l1[-1][1], ntt.MAX_PRECISION) != \
           round(events_l2[-1][1], ntt.MAX_PRECISION):
            print("%s (%s: %s): durations do not match:" % \
                (sys.argv[0], __name__, "check_durations"))
            print("\t%s (%f)" % (l1, events_l1[-1][1]))
            print("\t%s (%f)" % (l2, events_l2[-1][1]))
            return False

    # exit gracefully
    #
    return True
#
# end of function

# function: load_annotations
#
# arguments:
#  list: a list of filenames
#
# return: a list of lists containing all the annotations
#
# This method loops through a list and collects all the annotations.
#
def load_annotations(list_a, level_a = int(0), sublevel_a = int(0), \
                     channel_a = int(-1)):

    # create an annotation object
    #
    events = []
    ann = Annotations()

    # loop over the list
    #
    for fname in list_a:

        # load the annotations
        #
        if ann.load(fname) == False:
            print("%s (%s: %s): error loading annotation for file (%s)" % \
                (sys.argv[0], __name__, "load_annotations", fname))
            return None

        # get the events
        #
        events_tmp = ann.get(level_a, sublevel_a, channel_a)
        if events_tmp == None:
            print("%s (%s: %s): error getting annotation (%s)" % \
                (sys.argv[0], __name__, "load_annotations", fname))
            return None
        events.append(events_tmp)

    # exit gracefully
    #
    return events
#
# end of function
                
#                                                                              
# end of file 
