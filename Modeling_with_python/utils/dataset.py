################################################################################
#                                                                              
# dataset.py
#
# Copyright: Carnegie Mellon University 
# Author: Alvaro Collet (acollet@cs.cmu.edu)
#
################################################################################
""" dataset.py: Python module with generic dataset class (to be subclassed
    by each specific dataset).
"""

import roslib
import os
import yaml
import utils 
import glob

################################################################################
#
# Dataset class
#
################################################################################
class Dataset(object):
    
    pkg_name = None
    """ ROS package name of given dataset"""

    cam_file = None
    """ Camera calibration file for this dataset."""

    cam = None
    """ Camera structure (class Camera()). """
    
    paths = None
    """ Dictionary containing paths to different folders (loaded
    via yaml). If paths begin with '/', they are considered as absolute paths.
    Otherwise, the dataset package name is prepended. """
    
    names = None
    """ Dictionary containing name suffixes for different data types (loaded
    via yaml). """
    
    module = None
    """ Dataset sub-type. Useful if you subclass the dataset class in another
    module. It should be a string containing 'module_name.class'. """

    # ------------------------------------------------------------------------ #
    def __init__(self, yaml_file = None):
        """ Initialization function. ."""
        
        self.names = dict()
        self.paths = dict()

        self.type = __file__
        if yaml_file is not None:
            self.load_yaml(yaml_file)

    # ------------------------------------------------------------------------ #
    def getInfo(self, info_str = None):
        """ dataset introspection: get module name and class of this dataset.

        Usage: module, class = getInfo(info_str = None)

        Input:
            info_str - Info string of type 'module.class'. If not given, we 
            use self.__str__()
        Output:
            module - Module name where this dataset was imported from
            class - Class name of dataset
        """

        if info_str is None:
            # info_str says '<module.class object at X>'
            info_str = self.__str__()
            info_str = info_str[1:info_str.find(' ')] # Now module.class only
            self.module = info_str

        module = info_str[:len(info_str) - info_str[::-1].find('.') - 1]
        cls_name = info_str[len(info_str) - info_str[::-1].find('.'):]

        return module, cls_name

    # ------------------------------------------------------------------------ #
    def load_yaml(self, yaml_file):
        """ Load dataset data from yaml file. The file should contain, at
            least, the dictionary 'paths'.
            
            Usage: dataset.load_yaml(yaml_file)

            Input:
                yaml_file - Input filename of yaml dataset file.
            Output:
                -NONE- but variables in yaml_file are loaded to dataset.
            """
        
        with open(yaml_file, 'r') as fp:
            data = yaml.load(fp)

            for key in data:
                self.__dict__[key] = data[key]

            if self.module is not None:
                mod_name, cls_name = self.getInfo(self.module)
                module = __import__(mod_name)
                self.__class__ = getattr(module, cls_name)
                
    # ------------------------------------------------------------------------ #
    def dump_yaml(self, yaml_file):
        """ Save camera data into yaml file.

            Usage: cam.dump_yaml('yaml_file.yaml')
        """
        
        data = self.__dict__.copy()
        if data.has_key('cam'):
            data.pop('cam')


        with open(yaml_file, 'w') as fp:
            yaml.dump(data, fp)

    # ------------------------------------------------------------------------ #
    def getPath (self, type = None, subtype = ''):
        """Obtain paths for dataset. 
        
        Usage: path = getPath (type = None, subtype = '')

        Input:
            type{None} - If None, return root path of dataset. If type is given,
            return path to the subfolder 'type'. E.g.:
                path = getPath(type = 'img') --> path to images folder
                path = getPath(type = 'depth') --> path to depth images folder
                path = getPath(type = 'output') --> path to output folder
                path = getPath(type = 'summary') --> path to summary folder

        Output:
            path - absolute path for dataset.
        """
        if type is None and not self.paths.has_key('root'):
            if self.pkg_name is not None:
                try:
                    path = roslib.packages.get_pkg_dir(self.pkg_name)
                    # Just query ROS the first time, then cache the value
                    self.paths['root'] = path
                except:
                    print("Could not locate " + self.pkg_name + " using ROS. \
                          Will revert to default")
                    path = os.path.dirname( os.path.abspath(__file__) )
            else:
                path = os.path.dirname( os.path.abspath(__file__) )

        elif type is None:
            path = self.paths['root']

        else:
            try:
                path = self.paths[type]
                if len(path) == 0 or path[0] != '/':
                    path = os.path.join( self.getPath(), path )
            except KeyError:
                raise KeyError, "Could not locate path for type " + type + \
                        "in dataset."

        if subtype is not '':
            path = os.path.join(path, subtype)
        return path

    # ------------------------------------------------------------------------ #
    def getOutputPath (self):
        """Obtain absolute path for base output folder in dataset. 
        
        Usage: path = getOutputPath ()

        Input:
        Output:
            path - absolute output path for dataset.
        
        NOTE: Function kept for compatibility purposes. It is better to call
            getPath (type = 'output').
        """
        return self.getPath(type = 'output')

    # ------------------------------------------------------------------------ #
    def getOutputName (self, basename):
        """Obtain absolute output file name for a given basename.
        
        Usage: output_file = getOutputName(basename)

        Input:
            basename - Base name of object data (e.g.:
                'coffee/coffee_1/coffee_1_1_254')
        Output:
            output_file - filename to store output for given basename

        NOTE: Function kept for compatibility purposes. It is better to call
            getName(basename, type = 'output').
        """
        
        return self.getName(basename, type = 'output')

    # ------------------------------------------------------------------------ #
    def getName (self, basename, type, subtype = ''):
        """Obtain name for image, depth, mask... from basename for this dataset.
        
        Usage: name = getName (basename, type, subtype = '')

        Input:
            basename - Base name of object data (e.g.:
                'coffee/coffee_1/coffee_1_1_254')
            type - 'img', 'depth', 'mask', ...
        Output:
            name - file name for requested object data and type
        """

        if self.names.has_key(type):
            path = os.path.join (self.getPath(type, subtype), \
                                 basename + self.names[type])
            return path
        else:
            # Could not find type
            raise KeyError, "Could not find specified name in dataset."
    
    # ------------------------------------------------------------------------ #
    def getBasename(self, fullname, type, subtype = ''):
        """Obtain basename for fullname image, depth, mask...
        
        Usage: name = getBasename (fullname, type, subtype = '')

        Input:
            fullname - Fullname of object data (e.g.:
                '/dataset/coffee/coffee_1/coffee_1_1_254.png')
            type - file type 'img', 'depth', 'mask', ...
            subtype{''} - (Optional) Subtype within a category ('2D', '3D'...) 
        Output:
            basename - base filename for requested object data and type
        """

        path = self.getPath(type, subtype)
        if fullname.find(path) != 0 or fullname.find(self.names[type]) < 0:
            raise ValueError, "fullname does not seem to be of specified type"
        else:
            basename = fullname[len(path)+1:fullname.find(self.names[type])]
             
        return basename
            
    # ------------------------------------------------------------------------ #
    def getFiles(self, basename, type, subtype = ''):
        """Obtain list of files consistent with pattern for given type. 
        E.g., if we want a list of all sequences, we do:
            seq_files_list = getFiles('*', 'seq')
        
        Usage: name = getFiles (basename, type, subtype)

        Input:
            basename - Base name of object data (wildcards are accepted)
            type - 'img', 'depth', 'mask', ...
        Output:
            name - file name for requested object data and type
        """
        return glob.glob( self.getName(basename, type, subtype) )

    # ------------------------------------------------------------------------ #
    def getTypes(self):
        """ List of available types in this dataset.
        
        Usage: types_list_name, types_list_path = getTypes()

        Input:
            -NONE-
        Output:
            types_list_name - list of all known types of object names in this 
                dataset
            types_list_path - list of all known types of object paths in this
                dataset
        """
        return self.names.keys(), self.paths.keys()

    # ------------------------------------------------------------------------ #
    def loadData (self, basename, type = 'img', subtype = ''):
        """Import data type from file in dataset. We require the basename only. 

        Usage: img, depth, mask = loadData(basename, type = 'img')

        Input:
            basename - base name of file to load (if image is 
                /usr/file_123.jpg, then basename = '/usr/file_123')
            type - Type of data to load. Internally, this function calls the
            specialized functions 'loadType' for a given type. E.g.:
                type = 'img'   --> call loadImg
                type = 'depth' --> call loadDepth

        Output:
            data - Structure containing requested data. Check with the
                specialized loader for details on a specific structure.
        """
        # Get full path and filename to load
        name = self.getName (basename, type, subtype)

        # Load requested data with dynamic function name. Overwrite it if you
        # don't like it!
        if type is not None and type != '':
            func_name = 'load' + type[0].upper() + type[1:]
            # This function exists, so call it and return its data
            if hasattr(self, func_name):
                func = getattr(self, func_name)
                return func(name)
        
        return None
    
    # ------------------------------------------------------------------------ #
    def loadImg (self, name):
        """ Load image from dataset.

        Usage: img = loadImg(img_name)

        Input:
            img_name - Full path of image to load. It's useful to use something
                like: img_name = getName (basename, 'img')
        Output:
            img - Numpy array containing image (dtype = np.uint8)
        """
        pass

    # ------------------------------------------------------------------------ #
    def loadDepth (self, name):
        """ Load depth image from dataset.

        Usage: img = loadDepth(img_name)

        Input:
            img_name - Full path of image to load. It's useful to use something
                like: img_name = getName (basename, 'img')
        Output:
            depth - Numpy array containing depth image (dtype = np.float)
        """
        pass

    # ------------------------------------------------------------------------ #
    def loadMask (self, name):
        """ Load image mask from dataset.

        Usage: mask = loadMask(name)

        Input:
            name - Full path of mask to load. It's useful to use something
                like: mask_name = getName (basename, 'mask')
        Output:
            mask - Numpy array containing mask (dtype = np.uint8)
        """
        pass

    # ------------------------------------------------------------------------ #
    def process (self, basename, output, *args, **kwargs):
        """ Run algorithm on a file in dataset.
                
            Usage: out = dataset.process (basename, output, ...)

            Input:
                basename - Base path of file to use as input
                output - File to save to.
                [...] - Other optional arguments, dataset-dependent
            Output:
                out - structure resulting from processing the file 'basename'
        """
        pass


################################################################################
#
# RGBDDataset class - Generic dataset for rgb + depth files
#
################################################################################
class RGBDDataset(Dataset):

    # ------------------------------------------------------------------------ #
    def genBag(self, bagName, rgb_textfile, depth_textfile, timeout = 100):
        """ Generate ROS bag for group of images/depth images specified in text
        files.

        Usage: genBag (bagName, rgb_textfile, depth_textfile, timeout = 100)

        Input:
            bagName - Path and filename of output bag.
            rgb_textfile - Path to file that contains rgb image names.
            depth_textfile - Path to file that contains depth image names.
            timeout{100} - Max time to spend on a bag file, in seconds.
            
        Output:
            bag - Full path of generated bag file 

        WARNING: bag filename should point to a ***LOCAL*** folder, not
        NFS-mounted or remote folder. Rosbag has lots of issues with remote
        locations!
        
        WARNING 2: It is advisable to run an external roscore before calling
        this tool, otherwise there are issues when calling it multiple times.
        """
        import subprocess
        import shlex
        import time
        import signal
        
        print("Generating bag " + bagName + "... ")
        write_bag = "roslaunch opennisim opennisim2bag.launch" + \
                    " imglist:=" + rgb_textfile + \
                    " depthlist:=" + depth_textfile + \
                    " bagName:=" + bagName

        # Call process silently
        fnull = open(os.devnull, 'w')
        pid = subprocess.Popen(shlex.split(write_bag), stdout = fnull)
        time.sleep(float(timeout))
        print(" Done! \n")
        pid.send_signal(signal.SIGINT)
        fnull.close()

        return bagName 

    # ------------------------------------------------------------------------ #
    def loadSummary (self, name):
        """ Load scene summary from dataset.

        Usage: mask = loadSummary(name)

        Input:
            name - Full path of Summary to load. It's useful to use something
                like: mask_name = getName (basename, 'mask')
        Output:
            JS - JS structure containing only a summary of regions in
                JS.CompactReg.
        """
        return utils.load(name, 'JS')

    # ------------------------------------------------------------------------ #
    def loadSeq (self, name):
        """ Load data Sequence containing info about a subset of images.

        Usage: seq = loadSeq(name)

        Input:
            name - Full path of sequence to load. It's useful to use something
                like: name = getName (basename, 'seq')
        Output:
            seq - RGBDSequence() object containing a list of files and their
                poses.
        """
        return utils.load(name, 'seq')

    # ------------------------------------------------------------------------ #
    def loadOutput(self, name):
        """ Load output from dataset.

        Usage: JS = loadOutput(img_name)

        Input:
            img - Full path of image to load. It's useful to use something
                like: img_name = getName (basename, 'img')
        Output:
            JS - JS structure for this image
        """
        return utils.load(name, 'JS')

    # ---------------------------------------------------------------------------- #
    def rgbdslam_on_bags(self, bags_list):
        """ Execute rgbdslam code on specified bags 

        Usage: rgbdslam_on_bags (bags_list)

        Input:
            bags_list - List of bag filenames (with full path) to run rgbdslam.
                Output files are bags_list[i] + '_g2o.txt'

        Output:
            g2o_files - List of rgbdslam (g2o) outputs for given filename bags.
        """
        import subprocess
        import shlex
        import utils

        g2o_files = list()
        for bag in bags_list: 
            g2o_file = bag + '_g2o.txt'
            locked = utils.lock_file(g2o_file)
            if locked:
                print("Calling RGBDSlam on bag " + bag + "... ")
                launch_str = "roslaunch rgbdslam rgbdslam_bag.launch" + \
                            " bag:=" + bag 
                # Call process silently
                fnull = open(os.devnull, 'w')
                subprocess.call(shlex.split(launch_str), stdout = fnull)
                print(" Done! \n")
                fnull.close()
                utils.unlock_file(g2o_file)
                g2o_files.append(g2o_file)

        return g2o_files

    # ------------------------------------------------------------------------ #
    def parse_all_g2o(self, folder):
        """ Parse folder with g2o files.

        Usage: q_dict = parse_all_g2o(folder)

        Input:
            folder - Input folder containing many g2o files
        
        Output:
            q_dict - dictionary of pairs {filename: list of camera positions}. For
            more info on the list of camera positions, see 'parse_g2o'.
        """
        import glob

        files = glob.glob(os.path.join(folder, '*_g2o.txt'))

        q_dict = dict()
        q_success = dict()
        for f in files:
            q_dict[f], q_success[f] = self.parse_g2o(f)

        return q_dict, q_success

    # ------------------------------------------------------------------------ #
    def parse_g2o(self, g2o_file):
        """ Parse g2o file containing camera poses for multiple images.

        Usage: q_list, success = parse_g2o(g2o_file)

        Input:
            g2o_file - Input filename of text file containing g2o info
        
        Output:
            q_list - List of 7-d quaternion+translation (compatible with
                tf_format), one for each image in the sequence, in IMAGE TO
                WORLD transformation
            success - True/False/None. If True, the g2o file contains as many
                nodes as files in the bag. If False, the g2o file contains less
                nodes than files in the bag. If None, we could not find some 
                file.
        """
        import numpy as np

        q_list = list()
        with open(g2o_file, 'r') as file_:
            for line in file_:
                words = line.split()
                if words[0] == 'VERTEX_SE3:QUAT':
                    # words[1] is the node index
                    # words[8] is the quaternion scalar term
                    # words[5:7] is the quaternion x y z
                    # words[2:4] is the translation
                    q = np.array([float(words[8]), float(words[5]), \
                                  float(words[6]), float(words[7]), \
                                  float(words[2]), float(words[3]), \
                                  float(words[4])])
                    q_list.append(q)

        # Sanity check: verify that this file has the same number of nodes
        # as the corresponding '_rgb.txt' file
        rgb_file = g2o_file[:g2o_file.find('.bag_g2o.txt')] + '_rgb.txt'
        success = None
        if os.path.exists(rgb_file):
            with open(rgb_file, 'r') as rgb:
                lines = 0
                for _ in rgb:
                    lines += 1
            
            if lines == len(q_list):
                success = True
            else:
                success = False
        return q_list, success

    # ------------------------------------------------------------------------ #
    def kinfu(self, basename, type = 'list_ext'):
        """ Use Kinect fusion on list of depth images, and store cloud

        Usage: kinfu(basename, type = 'list_ext')

        Input:
            basename - Path to file that contains depth image names.
            type{'list_ext'} - List file type ('list' or 'list_ext')
            
        Output:
            cloud, poses - Filenames to output cloud and poses from kinfu 

        """ 
        roslib.load_manifest('kinfu')
        import kinfu

        depth_textfile = self.getName(basename, type)

        # Need to read list and export using full names
        fullnames = list()
        with open(depth_textfile, 'rt') as file_:
            for name in file_:
                fullnames.append( self.getName(name.rstrip(), 'depth') )
                
        output_prefix = os.path.join(self.getPath(type), basename) + '_'

        KF = kinfu.KinFuPy('file')
        cloud, poses = KF.kinfu_list(fullnames, output_prefix, opts='singlepass')

        # Store cloud and poses
        utils.save(self.getName(basename, 'cloud'), {'cloud': cloud}, \
                   zipped=True)
        utils.save(self.getName(basename, 'poses'), {'poses': poses}, \
                   zipped=True)
        return cloud, poses

    # ------------------------------------------------------------------------- #
    def reweight(self, basename, subtype = ''):
        """ Reweight segmentation with best known weights.
        
        Usage: dataset.reweight(basename)

        Input:
            basename - file base name. We assume we can generate a full name
            with dataset.getName(basename, 'output')

        Output:
            -NONE- but file will be saved in getName 'output2' format.
        """
    
        import Image
        import numpy as np

        input  = self.getName(basename, type = 'output', subtype = subtype)
        output = self.getName(basename, type = 'output2', subtype = subtype)
        output_img = self.getName(basename, type = 'output_img', subtype = subtype)

        JS = self.loadData(basename, 'output', subtype = subtype)

        JS.update()
        JS.import_data(JS.CImg, JS.PCloud)

        # Best-performing feature weights
        opts = JS.opts
        w = opts['features']['weights']
        w['concavity'] = 30 
        w['color_hist'] = 15 
        w['alignment'] = 5 
        w['continuity_grid'] = 30
        w['gray_hist'] = 0
        w['pca_feat'] = 1 
        w['perim_length'] = 10 
        w['projection'] = 5 
        w['verticality'] = 15 
        w['DefaultValue'] = 0.5 

        JS.init_features(opts)

        # Remove regions with <200 USEFUL 3D points
        discard_regs = list()
        for r in JS.Regions:
            if r.getPts3D().shape[-1] < 200:
                discard_regs.append(r)
        for r in discard_regs:
            JS.Regions.remove(r)

        # Compute Feature Scores
        JS.Regions.getScore(weights = w, force = 3) 
        
        # Compute score summary (compact regions)
        JS.CompactReg = JS.Regions.compact_iter(weights = w)
       
        # Save output
        utils.save(output, {'JS': JS}, zipped=True)
        
        # Save output image
        segm_img = JS.CompactReg.show_2d( show = False )
        Image.fromarray(segm_img.astype(np.uint8)).save(output_img)

    # ------------------------------------------------------------------------- #
    def summarize(self, basename, subtype = ''):
        """ Strip JS of most of its fields so that we have shorter load times
        on JS structures.
        
        Usage: dataset.summarize(basename)

        Input:
            basename - file base name. We assume we can generate a full name
            with dataset.getName(basename, 'output')

        Output:
            -NONE- but file will be saved in getName 'summary' format.
        """
     
        input  = self.getName(basename, type = 'output', subtype = subtype)
        output = self.getName(basename, type = 'summary', subtype = subtype)

        JS = self.loadData(basename, 'output', subtype = subtype)
        
        # Strip JS off of all unnecessary information
        JS.update()
        JS.Regions = None
        JS.Segmentations = None
        JS.Superpix = None
        JS.NmsRegions = None

        for r in JS.CompactReg:
            r.Children = None
            r.Siblings = None
            r.VisIdxImg = None
            r.Better_R = None
            r.Scores = None

        # Save output
        utils.save(output, {'JS': JS}, zipped=False)
