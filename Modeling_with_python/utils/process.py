#! /usr/bin/env python
################################################################################
#                                                                              
# process.py - Run experiments in multiple cores
#
# Copyright: Carnegie Mellon University & Intel Corp.
# Author: Alvaro Collet (acollet@cs.cmu.edu)
#
################################################################################
import optparse
import sys
import os.path
from task import ExecMP 


# --------------------------------------------------------------------------- #
def Process(module, arg, nCpu):
    """Process - Generate list of tasks and execute them. """

    try:
        mod = __import__(module)
    except ImportError:
        modname, ext = os.path.splitext(module)
        mod = __import__(modname)

    if not hasattr(mod, 'GenerateTasks'):
        print "Module must implement function 'GenerateTasks'."

    # Generate Tasks and execute them
    tasks = mod.GenerateTasks(arg)
    output = ExecMP(tasks, num_cpus = nCpu)

    print "Finished processing tasks. Done!"
    return output

# --------------------------------------------------------------------------- #
def parse_args(sysargs):
    # Parse arguments to execute experiments

    # Define options first
    usage = "usage: %prog [-n NCPU] [-a ARG] MODULE1_TO_RUN " + \
            "MODULE2_TO_RUN ...\n" + \
            "Each MODULE_TO_RUN is a python module to be imported, which " + \
            "contains scripts to be executed. The module must implement " + \
            "the function 'module'.GenerateTasks(ARG). See " + \
            "experiment_template.py for an example of module."


    parser = optparse.OptionParser()
    parser.add_option("-n", "--ncpu", type="int", default=None, dest="nCpu", \
                      help = "Number of concurrent processes to launch.")
    parser.add_option("-a", "--arg", dest="arg", default=None, \
                      help = "Argument to initialize set of experiments " +
                      " (e.g. 'index_file.txt').")
    # parser.add_option("-m", "--module", dest="module", \
    #                   help="Python module to be imported, containing scripts "+\
    #                   "to be executed. The module must implement the " + \
    #                   "function module.GenerateTasks(arg).")

    # Now process arguments
    (options, cmd_args) = parser.parse_args(sysargs) 
    if len(cmd_args) < 2:
        parser.error("At least one module is required.")

    return options, cmd_args

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Process.py - Execute script(s) on multiple CPUs
    (options, args) = parse_args(sys.argv)

    # Execute given scripts
    for mod in args[1:]:
        Process(module = mod.strip(), arg = options.arg, nCpu = options.nCpu)



