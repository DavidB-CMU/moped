################################################################################
#                                                                              
# plotting.py
#
# Copyright: Carnegie Mellon University & Intel Corp.
# Author: Alvaro Collet (acollet@cs.cmu.edu)
#
################################################################################
""" plotting.py: Experiment and Plotting classes to generate nice-looking plots
    from experiment data.
    
    List of Classes:
        Experiment - Import/Export experiment files
        Plotting - Generate plots using pyplot
"""

import commands
import numpy as np
import os.path
import csv
import datetime
import utils

# Default experiment path
EXP_PATH = './'

###############################################################################
#
# Experiment class
#
###############################################################################
class Experiment(list):
    """ An Experiment is a list of dictionaries, where each dictionary is one
        datapoint/result of the experiment. It incorporates some clever logic
        to act as a pseudo-dict when you access Exp['field'] --> array with
        values for 'field' in all datapoints.

        Example:
            Exp = Experiment()
            Exp.LoadFile('test_file.txt')
            
            Exp[i]['result'] --> result for datapoint i of experiment
            Exp[i]['name'] --> name for datapoint i of experiment
            

            Exp.extra_info['name'] --> Name of experiment
            Exp.extra_info['dataset'] --> Dataset used in experiment
        """

    extra_info = None 
    """ Extra info fields (dict). It must contain at least 'name' and 
    'dataset'."""

    # ----------------------------------------------------------------------- #
    def __init__(self, filename=None, folder=None):
        """ Load filename or folder contents in data matrix. """

        super(list, self).__init__()
        self.extra_info = dict()

        if filename is not None:
            self.LoadFile(filename)
        elif folder is not None:
            self.LoadFolder(folder)
    
    # ----------------------------------------------------------------------- #
    def __getitem__(self, key):
        if type(key) is not str:
            return list.__getitem__(self, key)
        else:
            return utils.dict2arr(self, key)

    # ----------------------------------------------------------------------- #
    def __setitem__(self, key, val):
        if type(key) is not str:
            return list.__setitem__(self, key, val)
        else:
            return utils.arr2dict(self, key, val)


    # ----------------------------------------------------------------------- #
    def LoadFolder (self, folder, expr='*.txt'):
        """ Load contents from given folder into structure to be processed.
        
        Usage: Exp.LoadFolder(folder, expr)

        Input:
            folder - str, full path of desired folder. We search for all txt
                     files within folder.
            expr{'*.txt'} - Expression to match within the folder. By default,
                           it is '*.txt'.
        Output:
            Experiment is filled with a list of dictionaries, one dict for each
            datapoint and their features.
            """

        obj_names = commands.getoutput('ls ' + os.path.join(folder, expr))
    
        # Repeat for all files in folder
        for obj_name in obj_names.splitlines():

            Exp = Experiment()
            Exp.LoadFile(obj_name)
            self.extend(Exp)

        # Assume all experiments in folder have the same extra info
        self.extra_info = Exp.extra_info.copy()

    # ----------------------------------------------------------------------- #
    def LoadFile(self, filename):
        """ Load experiment file into structure.
        
        Usage: Experiment.LoadFile(filename)

        Input:
            filename - File to read (including path) 
            
        Output:
            -NONE-, but the structure is loaded in self.

        NOTE: The file format is specified in the 'export' method.
        """
        # Define how we want to export data: strings in quotes, numbers without
        Dialect = csv.excel()
        Dialect.quoting = csv.QUOTE_NONNUMERIC
        Dialect.quotechar = "'"

        with open(filename, 'rt') as fp:

            # Get first few lines of global fields
            while True:
                line = fp.readline()
                if line.find(':') >= 0:    
                    Data = line.rstrip().split(':')
                    if len(Data) > 1:
                        if Data[1].isdigit():
                            self.extra_info[Data[0]] = float(Data[1].strip())
                        else:
                            self.extra_info[Data[0]] = Data[1].strip()
                    else:
                        self.extra_info[Data[0]] = ''
                else:
                    # No more global fields, rewind and go to local fields
                    fp.seek(fp.tell()-len(line))
                    break 

            # Read local fields
            reader = csv.DictReader(fp, dialect = Dialect)
            for datapoint in reader:
                self.append(datapoint)

    # ----------------------------------------------------------------------- #
    def export(self, filename=None, folder=EXP_PATH, 
               overwrite=False, write_extra_info=True):
        """ Export experiment data to single file. 
        
        Usage: Experiment.export(folder, filename, overwrite, write_global)

        Input:
            folder - Str, folder to export datafile. If not given, we use 
                     the global variable EXP_PATH from this module.
            filename - Str, filename to use when saving file. If not given, we
                       use filename = self.name + '_' + self.dataset + '_' + 
                       current date [ + '_' + 'EXP' + extra number if exp
                       already exists].
            overwrite{False} - If True, do not check if experiment already
                               exists. If False, generate a new file.
            write_extra_info{True} - If True, write the fields in
                    self.extra_info as a file header before the experiments.


        FILE FORMAT:
            'global_field_1': VALUE
            'global_field_2': VALUE
            'global_field_3': VALUE
            'local_field_1', 'local_field_2', 'local_field_3', 'local_field_4'
            VALUE_1,VALUE_2,VALUE_3,VALUE_4 --> datapoint 0
            VALUE_1,VALUE_2,VALUE_3,VALUE_4 --> datapoint 1 
            VALUE_1,VALUE_2,VALUE_3,VALUE_4 --> datapoint 2
        """
        # Define how we want to export data: strings in quotes, numbers without
        Dialect = csv.excel()
        Dialect.quoting = csv.QUOTE_NONNUMERIC
        Dialect.quotechar = "'"

        # Generate proper filename if not given
        if filename is None:
            filename = self.extra_info['name'] + '_' + \
                self.extra_info['dataset'] + '_' + \
                datetime.date.strftime(datetime.date.today(), '%d%b%Y') + \
                '.txt'

        # Add extra counter if necessary
        fullname = os.path.join(folder, filename)
        if os.path.isfile(fullname) and not overwrite:
            fullname, ext = os.path.splitext(fullname)
            # No 'EXP' suffix
            if fullname.find('_EXP') < 0:
                fullname = fullname + '_EXP001'
            else:
                # Find 'EXP' suffix with numbers, and increment one
                pos = fullname.find('_EXP')
                new_id = int(fullname[pos+4:]) + 1
                fullname = fullname[:pos] + "{0:0=3}".format(new_id)

        # Now we export all data into a single file
        with open(fullname, 'wt') as fp:

            # Write down header with global Experiment fields
            if write_extra_info:
                global_fields = self.extra_info.keys()
                global_fields.sort()
                for field in global_fields:
                    fp.write(field + ': ' + self.extra_info[field] + '\n')

            # Individual Datapoints
            if len(self) > 0:
                # Get sorted keys for each experiment
                exp_keys = self[0].keys()
                exp_keys.sort()

                # Now write individual experiment stuff
                writer = csv.DictWriter(fp, exp_keys, dialect = Dialect)
                if hasattr(writer, 'writeheader'):
                    # New in python 2.7
                    writer.writeheader()
                else:
                    writeheader = csv.writer(fp, dialect = Dialect)
                    writeheader.writerow(writer.fieldnames)
                writer.writerows(self)

###############################################################################
#
# ExpList class
#
###############################################################################
class ExpList(list):
    """ list of Experiment() objects."""

    type = None 
    """ Base class to generate experiment instances from."""

    # ----------------------------------------------------------------------- #
    def __init__(self, type=Experiment):
        """ Load filename or folder contents in data matrix. 
        
        Usage: E = ExpList(type = Experiment)
               E = ExpList(type = ExpJS)

        Input:
            type - Experiment() class (or derived class) to generate new
                   instances from. 
        Output:
            E - List of type() classes.
        """

        super(list, self).__init__()

        self.type = type 

    # ----------------------------------------------------------------------- #
    def LoadFiles(self, FileList):
        """ Load set of files into a list of experiments.
        
        Usage: ExpList.LoadFiles(['file1.txt', 'file2.txt', 'file3.txt'])
               ExpList.LoadFiles(FileList)

        Input:
            FileList - List of files to load. Each file is loaded as an instance
                       of ExpList.type, using ExpList.type.LoadFile(FileList[i])

        Output:
            Success - Return True if everything loaded correctly, False
                      otherwise.
        """

        # Handle the case of a string and not a list
        if type(FileList) is str:
            self.append(self.type.LoadFile(FileList))
        else:
            # List of files
            for fname in FileList:
                self.append(self.type.LoadFile(fname))

        return True
        
    # ----------------------------------------------------------------------- #
    def LoadFolders(self, FolderList):
        """ Load a set of folders into a list of experiments.
        
        Usage: ExpList.LoadFolders(['folder1', 'folder2', 'folder3'])
               ExpList.LoadFolders(FolderList)

        Input:
            FolderList - List of folders to load. Each folder is loaded as an 
                         instance of ExpList.type, using 
                         ExpList.type.LoadFolder(FolderList[i])

        Output:
            Success - Return True if everything loaded correctly, False
                      otherwise.
        """

        # Handle the case of a string and not a list
        if type(FolderList) is str:
            self.append(self.type.LoadFolder(FolderList))
        else:
            # List of folders
            for folder in FolderList:
                self.append(self.type.LoadFolder(folder))

        return True

    
    # ----------------------------------------------------------------------- #
    def summary(self, func = 'avg'):
        """Show a summary of all experiments. For each experiment, print a list
        of 'experiment name' and the result of Exp[i].func().

        Usage: Summary = ExpList.summary(func = 'avg')

        Input:
            func - string containing a function name to call for each 
                   Experiment instance.

        Output:
            Summary - list of tuples containing (Exp[i].name, Exp[i].func).
                The results are also printed on the screen.
        """

        Summary = list()

        for Exp in self:
            exp_summary = (Exp.extra_info['name'], getattr(Exp, func)())
            Summary.append(exp_summary)
            print "Name: " + exp_summary[0] + ", " + func + ": " + \
                  str(exp_summary[1])

        return Summary
            

###############################################################################
#
# ExpJS class
#
###############################################################################
class ExpJS(Experiment):
    """ An ExpJS is subclass of Experiment with additional functions.

        Example:
            Exp = Experiment()
            Exp.LoadFile('test_file.txt')
            
            Exp[i]['result'] --> result for datapoint i of experiment
            Exp[i]['name'] --> name for datapoint i of experiment
            

            Exp.extra_info['name'] --> Name of experiment
            Exp.extra_info['dataset'] --> Dataset used in experiment
        """

    pascal_ratio = 0.5
    """ Pascal ratio to use in calculations."""

    # ----------------------------------------------------------------------- #
    def avg(self, ratio = None):
        """AVG - find the average performance of an experiment. The formula
        used is:
            Exp.avg(ratio) = np.sum(Exp['ratio'] > ratio) / \
                                    float(Exp['ratio'].size)

            If ratio is not given, the default value of self.pascal_ratio
            is used.
        """
        if not ratio:
            ratio = self.pascal_ratio

        return np.sum(self['ratio'] > ratio) / float(self['ratio'].size)



###############################################################################
#
# Plotting class
#
###############################################################################
class Plotting(object):
    """ Plotting - Class to generate nice plots for experiments """

    opts = dict()
    """ Global plotting options. """

    
    # ----------------------------------------------------------------------- #
    def __init__(self):

        self.opts['legend'] = True
        self.opts['xlabel'] = True 
        self.opts['ylabel'] = True
        self.opts['title'] = True

    # ----------------------------------------------------------------------- #
    def cumhist(self, ExpList, field='result', nBins=100, show=True, **kwargs):
        """ Generate a cumulative histogram plot from multiple Experiments.
        
        Usage: cumhist(ExpList, field, nBins=100, show=True, **kwargs)

        Input:
            ExpList - List of Experiment()
            field{'result'} - Field to plot in each experiment. By default,
                              it is 'result'.
            nBins{100} - Number of bins to use in cumulative histogram
            show{True} - show the graph, or (if false) return a handle to it.
            kwargs available: title, xlabel, ylabel
        """
        from matplotlib import pyplot 

        # New figure
        fig = pyplot.figure()

        StepSize = 1./nBins
        bins_hist = np.arange(0, 1 + StepSize, StepSize)
        for Exp in ExpList:
            hist, bins = np.histogram(Exp[field], bins_hist)
            cumhist = np.cumsum(hist[::-1])[::-1] / float(Exp[field].size)
            pyplot.plot(bins[:-1], cumhist, label = Exp.extra_info['name'], \
                        figure = fig)

        # Default title and labels
        if self.opts['legend']:
            pyplot.legend()
        if self.opts['xlabel']:
            pyplot.xlabel('True Segm. Ratio (S $\cap$ GT) / (S $\cup$ GT)')
        if self.opts['ylabel']:    
            pyplot.ylabel('Frequency')
        if self.opts['title']:
            pyplot.title('Cumulative Histogram')
   
        # Add extra kwargs
        for key in kwargs:
            getattr(pyplot, key)(kwargs[key])
        
        if show:
            pyplot.show()

        return fig

    # ----------------------------------------------------------------------- #
    def line(self, ExpList, field='result', show=True, **kwargs):
        """ Plot raw results (in line form) from multiple Experiments.
        
        Usage: line(ExpList, field, show=True, **kwargs)

        Input:
            ExpList - List of Experiment()
            field{'result'} - Field to plot in each experiment. By default,
                              it is 'result'.
            show{True} - show the graph, or (if false) return a handle to it.
            kwargs available: title, xlabel, ylabel
        """
        from matplotlib import pyplot 

        # New figure
        fig = pyplot.figure()

        for Exp in ExpList: 
            pyplot.plot(Exp[field], label = Exp.extra_info['name'])

        # Default title and labels
        if self.opts['legend']:
            pyplot.legend()
        if self.opts['xlabel']:
            pyplot.xlabel('True Segm. Ratio (S $\cap$ GT) / (S $\cup$ GT)')
        if self.opts['ylabel']:    
            pyplot.ylabel('Frequency')
        if self.opts['title']:
            pyplot.title('Cumulative Histogram')
   
        # Add extra kwargs
        for key in kwargs:
            getattr(pyplot, key)(kwargs[key])
        
        if show:
            pyplot.show()

        return fig
    

