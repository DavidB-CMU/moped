################################################################################
#                                                                              
# PCloud.py: Python module containing a Point Cloud and Depth Image classes,
#   and their parent class BaseRangeData.
#
# Copyright: Carnegie Mellon University & Intel Corp.
# Author: Alvaro Collet (acollet@cs.cmu.edu)
#
################################################################################
""" BaseRangeData - parent class for generic use of point clouds (needs to be
        subclassed).
    DepthImage - Depth Image class for python.
    PCloud - Point Cloud class for python.
    """

import numpy as np
import utils

################################################################################
#
# BaseRangeData class
#
################################################################################
class BaseRangeData(np.ndarray):
    """BaseRangeData - A basic class to work with range data in python, built 
        over numpy.
    """

    M = None
    """ 3-by-4 transformation matrix (world to sensor)."""

    type = 'depth'
    """ Sensor type. For now, either 'pcloud' (unstructured 3D data) or 'depth'
    (depth image)."""

    id = ''
    """ Cloud ID/name (in case it's needed)."""

    # ------------------------------------------------------------------------ #
    def __new__(subtype, data=[], M=None, type='pcloud', id=''):
        """ Usage: BaseRangeData(array) 

           returns: instance of BaseRangeData (default type is 'pcloud') 
        """

        if hasattr(data, 'float_data'): 
            # ROS message
            subarr = np.array(data.float_data).reshape((data.height, \
                                                        data.width)).copy()
        else:
            # Make sure we are working with an array, and copy the data if requested
            subarr = data.copy()

        # Transform 'subarr' from an ndarray to our new subclass.
        subarr = subarr.view(subtype)

        # Use the specified M if given
        if M is not None:
            subarr.M = M.copy()
        
        if type:
            subarr.type = type

        if id:
            subarr.id = id

        # Finally, we must return the newly created object:
        return subarr

    # ------------------------------------------------------------------------ #
    def __array_finalize__(self, obj):
        """Initialize extra values in class. Necessary to subclass ndarray"""

        self.M = getattr(obj, 'M', np.eye(3,4))
        self.type = ''
        self.id = ''

    # ------------------------------------------------------------------------ #
    def __repr__(self):
        desc="""BaseRangeData(%(data)s, type=%(tag)s)"""
        return desc % {'data': str(self), 'tag':self.type }

    # ------------------------------------------------------------------------ #
    # ADDED TO MAKE OBJECT PICKLABLE
    def __reduce__(self):
        """ Create object state (necessary to pickle a subclassed ndarray)"""

        object_state = list(np.ndarray.__reduce__(self))

        # Store extra variables, but not XY
        vals = list(self.__dict__.values())

        subclass_state = tuple(self.__dict__.values()) 
        object_state[2] = (object_state[2],subclass_state)
        return tuple(object_state)
 
    # ----------------------------------------------------------------------- #
    def __setstate__(self, state):
        """ Recover state from pickled data """

        np.ndarray.__setstate__(self,state[0])
        
        for i, key in enumerate(self.__dict__.keys()):
            self.__dict__[key] = state[1][i]
    
    # ------------------------------------------------------------------------ #
    def copy(self):
        """copy - Generate copy of self object.
        
        Usage: obj2 = obj.copy()

            """
        return BaseRangeData(self.view(np.ndarray), M = self.M,\
                             type = self.type, id = self.id)

    # ------------------------------------------------------------------------ #
    def getNumPts(self):
        """Return number of 3D points in cloud
        
        Usage: nPts = obj.getNumPts()

        Input:
            -NONE-

        Output:
            nPts - Total number of 3D points in point cloud.
        """
        # YOU NEED TO IMPLEMENT THIS FUNCTION IN YOUR SUBCLASS
        pass

    # ----------------------------------------------------------------------- #
    def getPts3D(self, idx = None):
        """Transform from depth image to 3D point cloud.
        
        Usage: pts3d = obj.getPts3D(idx)
        
        Input:
            idx - List of indexes 

        Output:
            pts - np.ndarray - 3-by-N ndarray of 3D points

        """
        # YOU NEED TO IMPLEMENT THIS FUNCTION IN YOUR SUBCLASS
        pass

################################################################################
#
# DepthImage class
#
################################################################################
class DepthImage(BaseRangeData):
    """DepthImage - A basic DepthImage class for python, built over numpy.
    """
    height = None
    """Image height"""

    width = None
    """Image width"""

    K = None
    """ 3-by-3 intrinsic transformation matrix."""
    
    XY = None
    """ M-by-N-by-2 temp array containing X, Y positions for each pixel.
        Jointly, [XY self] is a matrix of [X Y Z] values. This array won't be
        stored if pickled."""

    # ------------------------------------------------------------------------ #
    def __new__(subtype, data=[], K=None, M=None, type='depth', id=''):
        """ Usage: DepthImage(array, K) 
                   DepthImage(DepthMap) --> ROS Message of type DepthMap

           returns: instance of DepthImage
        """

        if hasattr(data, 'float_data'): 
            # ROS message
            subarr = np.array(data.float_data).reshape((data.height, \
                                                        data.width)).copy()
        else:
            # Make sure we are working with an array, and copy the data if 
            # requested
            subarr = data.copy()

        # Transform 'subarr' from an ndarray to our new subclass.
        subarr = subarr.view(subtype)

        if hasattr(data, 'width'):
            subarr.height = data.height
            subarr.width = data.width
        else:
            try:
                subarr.height = data.shape[0]
                subarr.width = data.shape[1]
            except IndexError, AttributeError:
                subarr.height = 0
                subarr.width = 0

        # Use the specified K if given
        if hasattr(data, 'focal_distance'):
            subarr.K = np.eye(3,3)
            subarr.K[0,0] = data.focal_distance
            subarr.K[1,1] = data.focal_distance
            subarr.K[0,2] = subarr.height/2.
            subarr.K[1,2] = subarr.width/2.
        if K is not None:
            subarr.K = K.copy()

        # Use the specified M if given
        if M is not None:
            subarr.M = M.copy()
        
        if type:
            subarr.type = type

        if id:
            subarr.id = id

        # Finally, we must return the newly created object:
        return subarr

    # ------------------------------------------------------------------------ #
    def __array_finalize__(self, obj):
        """Necessary to subclass ndarray"""

        super(BaseRangeData, self).__init__()
        self.K = getattr(obj, 'K', np.eye(3,3))
        self.height = 0
        self.width = 0
        self.XY = None

    # ------------------------------------------------------------------------ #
    def __repr__(self):
        desc="""DepthImage(%(data)s, type=%(tag)s)"""
        return desc % {'data': str(self), 'tag':self.type }

    # ------------------------------------------------------------------------ #
    # ADDED TO MAKE OBJECT PICKLABLE
    def __reduce__(self):
        object_state = list(np.ndarray.__reduce__(self))

        # Store extra variables, but not XY
        XY = self.XY
        self.XY = None
        vals = list(self.__dict__.values())
        self.XY = XY

        subclass_state = tuple(self.__dict__.values()) 
        object_state[2] = (object_state[2],subclass_state)
        return tuple(object_state)
 
    # ----------------------------------------------------------------------- #
    def __setstate__(self,state):
        # Recover state from pickled data

        np.ndarray.__setstate__(self,state[0])
        
        for i, key in enumerate(self.__dict__.keys()):
            self.__dict__[key] = state[1][i]
    
        self.XY = self.getXYZ()[:,:,0:1]

    # ------------------------------------------------------------------------ #
    def copy(self):
        """copy - Generate copy of self object"""
        return DepthImage(self.view(np.ndarray), K = self.K, M = self.M, \
                          type = self.type, id = self.id)

    # ------------------------------------------------------------------------ #
    def show_2d(self, display=True):
        """show_2d - Use PIL to visualize depth image
        
        Usage: d_img.show_2d(display)

        """
        normImg = self * 255. / np.max(self)
        normImg[normImg < 0] = 0 
        normImg = normImg.astype(np.uint8)

        if display:
            import Image
            Image.fromarray(normImg).show()
        
        return normImg

    # ------------------------------------------------------------------------ #
    def show_3d(self):
        """SHOW - Use mayavi2 to visualize point cloud.
        
        Usage: DepthImage.show()

        """
       
        from enthought.mayavi import mlab
       
        # Get 3D points
        pts = self.tocloud()

        # I want at most 50K points
        stride = 1 + pts.shape[1]/50000

        pts = np.c_[pts[:,::stride], pts[:,::stride]]
        colors = np.ones(pts.shape[1], dtype=np.uint8)

        # Draw clusters in point cloud
        fig = mlab.figure()
        mlab.points3d(pts[0,:], pts[1,:], pts[2,:], \
                      colors, colormap='spectral', figure=fig, \
                      scale_mode='none', scale_factor=0.02)

        mlab.view(180,180)
        mlab.show() 

    # ----------------------------------------------------------------------- #
    def tocloud(self, use_M=False):
        """Transform from depth image to 3D point cloud.
        
        Usage: pts3d = d_img.tocloud()
        
        Input:
            -NONE-

        Output:
            pts - np.ndarray - 3-by-N ndarray of 3D points

        """

        P = np.ones([3, self.size])

        x = np.arange(0, self.width)
        y = np.arange(0, self.height)
        xx, yy = np.meshgrid(x, y)

        # From depth map to Point Cloud --> use focal distance
        P[0,:] = (xx.ravel() - self.K[0,2]) / self.K[0,0]
        P[1,:] = (yy.ravel() - self.K[1,2]) / self.K[1,1]
        P = P * self.ravel()
    
        valid_pts = P[2,:] > 0

        return P[:, valid_pts]
    
    # ----------------------------------------------------------------------- #
    def tocolor(self, maxR = 1.5, maxG = 4, maxB = 10):
        """Transform depth map into color-coded image.

        Usage: img = d_img.tocolor()

        Input:
            maxR, maxG, maxB - Distance at which each of the channels
                will achieve its maximum value. If all values are equal,
                the colored depth image is a greyscale image.

        Output:
            img - M-by-N-by-3 color-coded image with red as shortest
            distance and blue as furthest distance.
        """

        MIN_RED = 0.25
        MAX_RED = 1.5 
        MIN_GREEN = 0.25
        MAX_GREEN = 4 
        MIN_BLUE = 0.25
        MAX_BLUE = 10

        invalid_pts = self <= 0
        # Red Channel: from 0 to 2 meters
        d = self.copy()
        d[d > MAX_RED] = MAX_RED
        d[d < MIN_RED] = MIN_RED
        red = np.uint8((d-MIN_RED) * 255./(MAX_RED-MIN_RED))
        red[invalid_pts] = 0

        # Green Channel: from 1 to 5 meters
        d = self.copy()
        d[d > MAX_GREEN] = MAX_GREEN
        d[d < MIN_GREEN] = MIN_GREEN
        green = np.uint8((d-MIN_GREEN) * 255./(MAX_GREEN-MIN_GREEN))
        green[invalid_pts] = 0

        # Green Channel: from 1 to 5 meters
        d = self.copy()
        d[d > MAX_BLUE] = MAX_BLUE
        d[d < MIN_BLUE] = MIN_BLUE
        blue = np.uint8((d-MIN_BLUE) * 255./(MAX_BLUE-MIN_BLUE))
        blue[invalid_pts] = 0
       
        # Mix the three channels together 
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:,:,0] = red
        img[:,:,1] = green
        img[:,:,2] = blue

        return img

    # ----------------------------------------------------------------------- #
    def fix_K(self, newK=None, new_fdist=None):
        """ Modify depth image to adjust to a new intrinsics/focal
        distance.
        
        Usage: d_img.fix_K(newK = K)
               d_img.fix_K(new_fdist = f)
        
        Input:
            newK = 3-by-3 intrinsics camera matrix
            new_fdist = new focal distance (you should give either K or
               the focal distance. If both are given, K takes preference.

        Output:
            -NONE- The depth image and intrinsics matrix are modified in place.

        """

        if newK is not None:
            KK = np.array([newK[0,0], newK[1,1], newK[0,2], newK[1,2]], \
                          dtype=float) 
        elif new_fdist is not None:
            KK = np.array([new_fdist, new_fdist, self.height/2., \
                           self.width/2.], dtype=float)
        else:
            print 'Must input K or focal distance'
            return

        pts3D = self.tocloud()
        # Order is important for Z-buffering
        # We want points further away to be processed first
        idx = pts3D[2,:].argsort()
        pts3D = pts3D[:, idx[::-1]]
        pts2D = utils.C_ProjectPts(pts3D, np.eye(4,4, dtype=float), KK)
        
        # Now transfer points back to image
        bounds = np.array([ [0, self.width-1], \
                            [0, self.height-1] ])
        invalid_pts = utils.out_of_bounds(pts2D, bounds)
        pts2D[:, invalid_pts] = 0

        # Remember: IdxImg is in format [row col] 
        pts2D_int = np.array(np.round(pts2D), dtype=int)
        self[:] = 0
        self[pts2D_int[1,:], pts2D_int[0,:]] = pts3D[2,:]

        self.K = np.eye(3,3)
        self.K[0,0] = KK[0]
        self.K[1,1] = KK[1]
        self.K[0,2] = KK[2]
        self.K[1,2] = KK[3]

        return

    # ------------------------------------------------------------------------ #
    def getNumPts(self):
        """Return number of 3D points in cloud
        
        Usage: nPts = PCloud.getNumPts()

        Input:
            -NONE-

        Output:
            nPts - Total number of 3D points in point cloud.
        """

        return np.sum(self > 0)

    # ----------------------------------------------------------------------- #
    def getPts3D(self, idx = None):
        """Transform from depth image to 3D point cloud.
        
        Usage: pts3d = d_img.getPts3D(idx)
        
        Input:
            idx - Either 1d list of indexes to keep or 2d list of indexes. If 2d
            indexes, we assume it is a list of [x y] pixel positions. Given that
            python uses [row col], the indexed array is pts3D[idx[1,:],
            idx[0,:]].

        Output:
            pts - np.ndarray - 3-by-N ndarray of 3D points

        """

        # We need to distinguish between 1d indexes and 2d indexes
        # 1d-indexes
        if idx is None:
            return self.getXYZ().reshape(3, self.width*self.height)

        if len(idx.shape) == 1:
            pts3D = self.getXYZ().reshape(3, self.width*self.height)
            return pts3D[:, idx]

        elif len(idx.shape) == 2:
            pts3D = self.getXYZ()
            return pts3D[idx[1,:], idx[0,:]]

    # ------------------------------------------------------------------------ #
    def getXYZ(self):
        """ Get XYZ values in world coordinates for each pixel.

        Usage: XYZ = self.getXYZ()

        Input:
            -NONE-

        Output:
            XYZ - M-by-N-by-3 matrix of [X Y Z] world coordinates for each pixel
        """

        if self.XY is not None:
            return np.c_[np.atleast_3d(self.XY), np.atleast_3d(self)]
        else:
            x = np.arange(0, self.width)
            y = np.arange(0, self.height)
            xx, yy = np.meshgrid(x, y)

            XY = np.zeros((self.height, self.width, 2))

            # From depth map to Point Cloud --> use focal distance
            XY[:,:,0] = (xx - self.K[0,2]) / self.K[0,0]
            XY[:,:,1] = (yy - self.K[1,2]) / self.K[1,1]
            XY = XY * np.atleast_3d(self)
            return np.c_[np.atleast_3d(self.XY), np.atleast_3d(self)]


################################################################################
#
# PCloud class
#
################################################################################
class PCloud(BaseRangeData):
    """PCloud - A basic Point Cloud class for python, built over numpy.
    """

    # ------------------------------------------------------------------------ #
    def __new__(subtype, data=[], M=None, type='pcloud', id=''):
        """ Usage: PCloud(array, M, type, id) --> 3-by-N array of 3D points
                   PCloud(array)

           returns: instance of PCloud
        """
        if hasattr(data, 'positions'): 
            # Old PCloud class
            subarr = np.array(data.positions).copy()
        else:
            # Make sure we are working with an array, and copy the data if 
            # requested
            subarr = data.copy()

        # Transform 'subarr' from an ndarray to our new subclass.
        subarr = subarr.view(subtype)

        # Use the specified M if given
        if M is not None:
            subarr.M = M.copy()
        elif hasattr(data, 'M'):
            subarr.M = data.M
       
        if type:
            subarr.type = type
        elif hasattr(data, 'type'):
            subarr.type = data.type

        if id:
            subarr.id = id
        elif hasattr(data, 'id'):
            subarr.id = data.id

        # Finally, we must return the newly created object
        return subarr

    # ------------------------------------------------------------------------ #
    def __repr__(self):
        desc="""PCloud(%(data)s, type=%(tag)s)"""
        return desc % {'data': str(self), 'tag':self.type }

    # ------------------------------------------------------------------------ #
    def copy(self):
        """copy - Generate copy of self object"""

        return PCloud(self.view(np.ndarray), M = self.M, \
                      type = self.type, id = self.id)

    # ------------------------------------------------------------------------ #
    def show_2d(self, display=True):
        """show_2d - Use PIL to visualize depth image
        
        Usage: obj.show_2d(display)

        """
        # Will do this someday
        pass

    # ------------------------------------------------------------------------ #
    def __len__(self):
        """ Return number of 3d points in point cloud."""

        return self.shape[1]

    # ------------------------------------------------------------------------ #
    def show_3d(self):
        """SHOW_3D - Use mayavi2 to visualize point cloud.
        
        Usage: obj.show_3d()

        """
       
        from enthought.mayavi import mlab
        
        # I want at most 50K points
        stride = 1 + len(self)/50000

        pts = self[:,::stride]
        colors = np.ones(pts.shape[1], dtype=np.uint8)
        # Draw clusters in point cloud
        fig = mlab.figure()
        mlab.points3d(pts[0,:], pts[1,:], pts[2,:], \
                      colors, colormap='spectral', figure=fig, \
                      scale_mode='none', scale_factor=0.02)

        mlab.view(180,180)
        mlab.show() 

    # ------------------------------------------------------------------------ #
    def getNumPts(self):
        """Return number of 3D points in point cloud.
        
        Usage: nPts = obj.getNumPts()

        Input:
            -NONE-

        Output:
            nPts - Total number of 3D points in point cloud.
        """

        return len(self) 

    # ----------------------------------------------------------------------- #
    def getPts3D(self, idx = None):
        """Get a set of points from 3D point cloud.
        
        Usage: pts3d = obj.getPts3D(idx)
        
        Input:
            idx - 1-by-N 1d list of indexes to output (if None, returns all
                points in the point cloud)

        Output:
            pts - np.ndarray - 3-by-N ndarray of 3D points

        """

        # We need to distinguish between 1d indexes and 2d indexes
        # 1d-indexes
        if idx is None:
            return self
        else:
            return self[:, idx]

