################################################################################
#                                                                              
# task.py - Wrapper to execute experiments in multiple cores
#
# Copyright: Carnegie Mellon University & Intel Corp.
# Author: Alvaro Collet (acollet@cs.cmu.edu)
#
################################################################################
""" task.py - Easy way of executing tasks in multiple cores.

    Important functions:
        Task - class that encapsulates a single task to run.
        TaskCl - class that encapsulates a single task, when this task is a
            bound method in a class.
        Worker(task) - Execute a single task
        ExecMP(tasks, num_cpus) - Execute all tasks (a list of Task()) in 
            multiple cores. If num_cpus is not given, use all available cores.
"""


################################################################################
#
# Class Task
#
################################################################################

class Task(object):
    """ The Task class is a wrapper to execute jobs in multiprocessing.Pool in
    an easy and simple way. 
    
    Example:
        t = Task(func, (arg1, arg2,...) ) # Init task
        t()                               # Call task --> equivalent to 
                                          # func(arg1, arg2, ...)

    NOTE: In order to improve performance when using this class in
    multiprocessing applications, it is advisable that you DON'T PASS large
    objects as your arguments to the task, since they must be pickled and
    unpickled. It is better if you just pass e.g. a filename and load the file
    within the task.
    WARNING: Remember that, if you want to call a function with one argument,
    it still needs to be encapsulated in a sequence!
    """

    def __init__(self, func, args=(), kwargs={}):
        """ Task initializer. 
        Usage: t = Task(func, args).
        
        Input:
            func - function to execute
            args - SEQUENCE containing regular arguments for func
            kwargs - DICTIONARY containing keyworded arguments for func
        """
        self.func = func
        self.args = args

        if type(kwargs) is dict:
            self.kwargs = kwargs  
        else:
            raise TypeError, "kwargs must be a dictionary!"

    def __call__(self):
        return self.func(*self.args, **self.kwargs)

    def __str__(self):
        return 'Func: ' + self.func.__name__ + ' Args: ' + \
                self.args.__str__()

################################################################################
#
# Class Task
#
################################################################################

class TaskCl(object):
    """ The Task class is a wrapper to execute jobs in multiprocessing.Pool in
    an easy and simple way. 
    
    Example:
        t = TaskCl(obje, func_name, (arg1, arg2,...) ) # Init task
        t() # Call task --> equivalent to obj.func(arg1, arg2, ...)

    NOTE: In order to improve performance when using this class in
    multiprocessing applications, it is advisable that you DON'T PASS large
    objects as your arguments to the task, since they must be pickled and
    unpickled. It is better if you just pass e.g. a filename and load the file
    within the task.
    WARNING: Remember that, if you want to call a function with one argument,
    it still needs to be encapsulated in a sequence!
    """

    def __init__(self, obj, func_name, args=(), kwargs={}):
        """ TaskCl initializer. Workaround to execute bound methods in a 
        task.
        Usage: t = Task(obj, func_name, args).
        
        Input:
            obj - Instance object
            func_name - function name to execute
            args - SEQUENCE containing regular arguments for func_name
            kwargs - DICTIONARY containing keyworded arguments for func_name
        """
        self.obj = obj
        self.func = func_name
        self.args = args

        if type(kwargs) is dict:
            self.kwargs = kwargs  
        else:
            raise TypeError, "kwargs must be a dictionary!"

    def __call__(self):
        return getattr(self.obj, self.func)(*self.args, **self.kwargs)

    def __str__(self):
        return 'Obj: ' + str(self.obj.__class__) + 'Func: ' + self.func + \
                ' Args: ' + self.args.__str__()

# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def Worker(task):
    """A humble worker. Executes given task and returns output.
    
    Usage: Worker(task)

    Input:
        task - instance of class Task with a job ready to be executed

    Output:
        -Whatever the task outputs!- If the task does not return anything,
        Worker returns None.
    """
    return task()

# --------------------------------------------------------------------------- #
# ExecMP
# --------------------------------------------------------------------------- #
def ExecMP(tasks, num_cpus = None):
    """Execute all tasks in a pool of workers.

    Usage: result = ExecMP(tasks, num_cpus)

    Input:
        tasks - list of Task() to execute
        num_cpus - Number of simultaneous cores to use. If None, use all
                   available cores.

    Output:
        result - If necessary, return the results from all tasks in a list.
    """
    import multiprocessing
 
    # How many CPUs?
    pool_size = num_cpus if num_cpus is not None \
                         else multiprocessing.cpu_count()
   
    # Default: use all cores
    # In Python 2.7 we can respawn workers after N tasks, with maxtasksperchild
    pool = multiprocessing.Pool(processes = pool_size)

    # Execute all our tasks in multiple cores
    output = pool.map(Worker, tasks)
    pool.close()
    pool.join()
    
    return output

# --------------------------------------------------------------------------- #
# ExecMP_memfriendly
# --------------------------------------------------------------------------- #
def ExecMP_memfriendly(tasks, num_cpus = None, maxtasksperbatch=None):
    """Execute all tasks in a pool of workers, in several batches. We pay
    a small penalty in performance, but it is potentially much better for
    memory-hungry tasks

    Usage: result = ExecMP_memfriendly(tasks, num_cpus, maxtasksperbatch=None)

    Input:
        tasks - list of Task() to execute
        num_cpus - Number of simultaneous cores to use. If None, use all
                   available cores.
        maxtasksperbatch - Number of tasks to run in each batch of tasks.

    Output:
        result - If necessary, return the results from all tasks in a list.
    """
    import multiprocessing, utils
 
    # How many CPUs?
    pool_size = num_cpus if num_cpus is not None \
                         else multiprocessing.cpu_count()
   
    # Default: use all cores
    # In Python 2.7 we could respawn workers after N tasks with maxtasksperchild
    # pool = multiprocessing.Pool(processes = pool_size)

    if maxtasksperbatch is None:
        maxtasksperbatch = len(tasks)

    output = list()
    nBatches = len(tasks)/maxtasksperbatch
    for nTasks in range(nBatches):
        task_init = nTasks*maxtasksperbatch
        task_end = (nTasks+1)*maxtasksperbatch

        # Execute all our tasks in multiple cores
        pool = multiprocessing.Pool(processes = pool_size)
        out = pool.map(Worker, tasks[task_init:task_end])
        pool.close()
        pool.join()
        pool.terminate() # just in case
        output.extend(out)
        
        for idx in range(task_init, task_end):
            tasks[idx] = None

        # print "Finished with tasks up to " + str(task_end)

    # Now finish off the last few tasks
    task_init = nBatches * maxtasksperbatch
    task_end = len(tasks)
    if task_end > task_init:
        # Execute all our tasks in multiple cores
        pool = multiprocessing.Pool(processes = pool_size)
        out = pool.map(Worker, tasks[task_init:task_end])
        pool.close()
        pool.join()
        pool.terminate() # just in case
        output.extend(out)
        
        for idx in range(task_init, task_end):
            tasks[idx] = None

        # print "Finished with tasks up to " + str(task_end)

    return output

# --------------------------------------------------------------------------- #
# Example of use 
# --------------------------------------------------------------------------- #
def example():

    tasks = list()
    tasks.append(Task(func = sum, args = ((3,4),) ))
    tasks.append(Task(func = sum, args = ((3,5),) ))
    tasks.append(Task(func = sum, args = ((6,8),) ))
    tasks.append(Task(func = sum, args = ((6,8),) ))
    tasks.append(Task(func = sum, args = ((6,8),) ))
    tasks.append(Task(func = sum, args = ((6,8),) ))
    tasks.append(Task(func = sum, args = ((6,8),) ))
    tasks.append(Task(func = sum, args = ((6,8),) ))

    pool_outputs = ExecMP(tasks)

    print pool_outputs


