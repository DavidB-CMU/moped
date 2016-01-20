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
from scipy import ndimage
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
        self.type = getattr(obj, 'type', 'pcloud')
        self.id = getattr(obj, 'id', '')
        return

    # ------------------------------------------------------------------------ #
    def __repr__(self):
        desc="""BaseRangeData(%(data)s, type=%(tag)s)"""
        return desc % {'data': str(self), 'tag': self.type }

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

    # ----------------------------------------------------------------------- #
    # Reclass 
    @classmethod
    def reclass(cls, obj):
        """ Update the object class type. Useful if pickled object is outdated
            and want to refresh the class methods without copying data.

            Usage: New_Class.reclass(obj)

            Input:
                obj - Object to have its classed renewed to 'New_Class'
            Output:
                -NONE- The change is done in place
        """
        obj.__class__ = cls 

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
    def __new__(subtype, data=None, K=None, M=None, type='depth', id=''):
        """ Usage: DepthImage(array, K) 
                   DepthImage(DepthMap) --> ROS Message of type DepthMap

           returns: instance of DepthImage
        """

        if data is None:
            data = np.array([])

        if hasattr(data, 'float_data'): 
            # ROS message
            subarr = np.array(data.float_data).reshape((data.height, \
                                                        data.width)).copy()
        else:
            # Make sure we are working with an array, and copy the data if 
            # requested
            subarr = data.view(np.ndarray).copy()

        # Transform 'subarr' from an ndarray to our new subclass.
        subarr = subarr.view(subtype)

        if hasattr(data, 'width'):
            subarr.height = data.height
            subarr.width = data.width
        else:
            try:
                subarr.height = data.shape[0]
                subarr.width = data.shape[1]
            except (IndexError), (AttributeError):
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
        elif hasattr(data, 'K'):
            subarr.K = data.K.copy()
        else:
            subarr.K = np.eye(3,3)

        # Use the specified M if given
        if M is not None:
            subarr.M = M.copy()
        elif hasattr(data, 'M'):
            subarr.M = data.M.copy()
        else:
            subarr.M = np.eye(3,4)
        
        if type:
            subarr.type = type
        elif hasattr(data, 'type'):
            subarr.type = data.type

        if id:
            subarr.id = id
        elif hasattr(data, 'id'):
            subarr.id = data.id      

        # Finally, we must return the newly created object:
        return subarr

    # ------------------------------------------------------------------------ #
    def __array_finalize__(self, obj):
        """Necessary to subclass ndarray"""

        super(DepthImage, self).__array_finalize__(obj)
        self.K = getattr(obj, 'K', np.eye(3,3))
        self.height = getattr(obj, 'height', 0)
        self.width = getattr(obj, 'width', 0)
        self.XY = getattr(obj, 'XY', None)
        return

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
    
        # self.XY = self.getXYZ()[:,:,0:1]

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
    def tocolor2(self, maxR = 1.5, maxG = 4, maxB = 10):
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
        MAX_GREEN = 5 

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

        # Blue Channel: gradients info
        grad = np.gradient(self)
        d = np.zeros(self.shape)
        d[grad[0] > 0] = 127
        blue = ndimage.gaussian_filter(d, sigma=8) 
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

    # ----------------------------------------------------------------------- #
    def new_from_K(self, width=None, height=None, K=None):
        """ New depth image changing size (interpolating) and modifying
        intrinsics K.
        
        Usage: new_d_img = d_img.new_from_K(width, height, K)
        
        Input:
            width - Width of output image
            height - Height of output image
            newK - 4-by-1 intrinsics camera matrix

        Output:
            new_d_img - Output depth image with different intrinsics and size.
        """

        if K is None:
            K = self.K

        if len(K) == 9:
            KK = np.r_[K[0,0], K[1,1], K[0,2], K[1,2]]
        elif len(K) == 4:
            KK = K
            
        #TODO: Finish func

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
    def getPts3D(self, idx = None, filter=True):
        """Transform from depth image to 3D point cloud.
        
        Usage: pts3d = d_img.getPts3D(idx)
        
        Input:
            idx - Either 1d or 2d list of indexes to keep. If 2d
            indexes, we assume it is a list of [x y] pixel positions. Given that
            python uses [row col], the indexed array is pts3D[idx[1,:],
            idx[0,:]].

        Output:
            pts - np.ndarray - 3-by-N ndarray of 3D points

        """
        # We need to distinguish between 1d indexes and 2d indexes
        # 1d-indexes
        if idx is None:
            p3 = self.getXYZ().reshape(self.width*self.height, \
                                       3).T.view(np.ndarray)
        elif idx.size == 0:
            p3 = np.zeros((3,0))

        elif len(idx.shape) == 1:
            pts3D = self.getXYZ().reshape(self.width*self.height, \
                                          3).view(np.ndarray)
            p3 = pts3D[:, idx]

        elif len(idx.shape) == 2:
            # Is it a list of indexes, or a True/False boolean array?
            if idx.shape == self.shape:
                idx = utils.C_ind2sub(utils.find(idx), self.shape)
                idx = idx[::-1,:] # We need [x y], we have [r c]

            idx = idx.astype(int) # Ensure it's integers
            pts3D = self.getXYZ().view(np.ndarray)
            p3 = pts3D[idx[1,:], idx[0,:], :].T

        if filter:
            # Delete points with negative Z
            valid_pts = p3[2,:] > 0
            p3[:, valid_pts]

        return p3

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
            XY = XY * np.atleast_3d(self).view(np.ndarray)
            self.XY = XY
            return np.c_[np.atleast_3d(self.XY), \
                         np.atleast_3d(self).view(np.ndarray)]

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
        try:
            if hasattr(data, 'positions'): 
                # Old PCloud class
                subarr = np.array(data.positions).copy()
            else:
                # Make sure we are working with an array, and copy the data if 
                # requested
                subarr = data.view(np.ndarray).copy()
        except:
            # Return object as is, we have no idea what's in here.
            return data 

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

        return self.shape[-1]

    # ------------------------------------------------------------------------ #
    def show_3d(self, colors=None, colormap='gray', scale_factor=0.02, 
                display=True):
        """SHOW_3D - Use mayavi2 to visualize point cloud.
        
        Usage: obj.show_3d(colors, colormap, scale_factor, display)

        Input:
            colors{None} - If none, consider Z as depth and paint points
                accordingly. Otherwise, N-array of scalars.
            colormap{'gray'} - Any colormap that mayavi accepts ('gray',
                'spectral', 'bone', ...)
            scale_factor{0.02} - Define scale of 3D points.
            display{True} - If True, show figure and lock environment. If False, 
                return figure handler.
        """
       
        from enthought.mayavi import mlab
        
        if colors is None:
            colors = self[2,:]

        # I want at most 50K points
        stride = 1 + len(self)/50000

        pts = self[:,::stride]
        colors = colors[::stride]

        # Draw clusters in point cloud
        fig = mlab.figure()
        mlab.points3d(pts[0,:], pts[1,:], pts[2,:], \
                      colors, colormap=colormap, figure=fig, \
                      scale_mode='none', scale_factor=scale_factor)

        mlab.view(180,180)
        
        if show:
            mlab.show() 
        else:
            return fig

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
            return self.view(np.ndarray)
        else:
            return self.view(np.ndarray)[:, idx]

    # ----------------------------------------------------------------------- #
    def toDepthImage(self, width, height, K, viewpoint = None, zbuffer=True):
        """ Compute Depth Image from point cloud.
        
        Usage: dImg = cloud.toDepthImage(width, height, K, viewpoint)
        
        Input:
            width - Width of output image
            height - Height of output image
            K - numpy array, 3-by-3 intrinsics camera matrix 
            viewpoint - numpy array 4x4 extrinsic camera matrix, image to world
            zbuffer{True} - Fill out missing values with Z-buffering
        Output:
            dImg - Output depth image from given viewpoint
        """
        # Parse input arguments
        if K is None:
            K = self.K
        if K.size == 9:
            KK = np.r_[K[0,0], K[1,1], K[0,2], K[1,2]]
        elif K.size == 4:
            KK = K

        if viewpoint is None:
            viewpoint = np.eye(4,4, dtype=float)

        pts3D = self.copy()

        # Order is important for Z-buffering
        # We want points further away to be processed first
        idx_dist = pts3D[2,:].argsort()
        pts3D = pts3D[:, idx_dist[::-1]]
        pts2D = utils.C_ProjectPts(pts3D, viewpoint, KK)
        
        # Now transfer points back to image
        bounds = np.array([ [0, width-1], [0, height-1] ])
        invalid_pts = utils.out_of_bounds(pts2D, bounds)
        pts2D[:, invalid_pts] = 0

        # Remember: IdxImg is in format [row col] 
        pts2D_int = np.array(np.round(pts2D), dtype=int)

        # Get new depth image
        dImg = DepthImage(np.zeros((height, width)), K = K)

        if not zbuffer:
            dImg[pts2D_int[1,:], pts2D_int[0,:]] = pts3D[2,:]
        else:
            # Z-Buffering --------------------------------------------------- #
            # Create some circle windows according to distance
            # Points that are further away from the camera --> smaller windows
            num_windows = 7 
            winX = list()
            winY = list()
            minX = list(); maxX = list()
            minY = list(); maxY = list()
            for i in range(num_windows):
                x, y = utils.circle( radius = i*0.75 + 2)
                winX.append(x)
                winY.append(y)
                minX.append(np.zeros(x.shape))
                maxX.append(np.ones(x.shape) * width - 1)
                minY.append(np.zeros(y.shape))
                maxY.append(np.ones(y.shape) * height - 1)
            
            # Paint pixels, first the ones further away
            for pt_idx in idx_dist[::-1]:
                winsize = np.int(np.fmin(np.ceil(2./pts3D[2, pt_idx]),
                                         num_windows))-2
                winX_idx = np.array(np.fmax(np.fmin(winX[winsize] + \
                                   pts2D_int[0, pt_idx], maxX[winsize]), \
                                   minX[winsize]), dtype=int)
                winY_idx = np.array(np.fmax(np.fmin(winY[winsize] + \
                                   pts2D_int[1, pt_idx], maxY[winsize]), \
                                   minY[winsize]), dtype=int)
                dImg[winY_idx, winX_idx] = pts3D[2, pt_idx]

        return dImg

