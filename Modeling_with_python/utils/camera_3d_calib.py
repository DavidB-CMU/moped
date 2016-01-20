#!/usr/bin/env python
################################################################################
#                                                                              
# camera_3d_calib.py - Calibrate camera and kinect/3d map by clicking on stuff.
#
# Copyright: Carnegie Mellon University
# Author: Alvaro Collet (acollet@cs.cmu.edu)
#
#################################################################################

import pcd_reader
import clicker as ck
import numpy as np
import tempfile
import tf_format
import utils
import Image

# --------------------------------------------------------------------------- #
class Calib2d3d(object):
    """ Calib2d3d - Class to calibrate image to 3D data. """

    img = None
    """ Numpy array with image data"""

    cloud = None
    """ 3-by-N (or 4-by-N) array of 3D values."""

    img_filename = None
    """ image filename """

    pcd_filename = None
    """ Cloud filename (pcd format) """

    img_clicks = None
    """ list of tuples containing (x,y) coordinates for each click."""

    pcd_clicks = None
    """ list of tuples containing (x,y) coordinates for each click."""

    pts2D = None
    """ List of 2D coordinates (x,y) to optimize"""

    pts3D = None
    """ List of 3D coordinates (x,y,z) to optimize"""

    # ----------------------------------------------------------------------- #
    def __init__(self, img_filename, pcd_filename):
        """ Initialization.
        
        Usage: Calib2d3d( image_filename, pcd_filename )

        """
        self.img_filename = img_filename
        self.pcd_filename = pcd_filename
        self.img = np.asarray(Image.open(img_filename))
        self.cloud = pcd_reader.read_pcd(pcd_filename)

    # ----------------------------------------------------------------------- #
    def getClicks2D(self):
        """ Display image and get click positions.
        
        Usage: pts2D = getClicks2D()

        Input:
            -NONE-
        Output:
            pts2D - 2d points (x,y) in the image corresponding to the clicks        
        """
        clicker = ck.Clicker()
        clicker.clicker(img_filename)
        self.img_clicks = np.array(clicker.coords, dtype=np.float).T
        return self.img_clicks

    # ----------------------------------------------------------------------- #
    def getClicks3D(self):
        """ Display depth image and get click positions.
        
        Usage: pts3D = getClicks3D()

        Input:
            -NONE-
        Output:
            pts3D - 3-by-N array of 3d points in 3d map corresponding to clicks
        """
        
        # Get just the depth values
        depth = self.cloud[2,:]
        depth = depth.reshape(self.img.shape)
        
        temp_ = tempfile.NamedTemporaryFile(suffix='.png', delete = False)

        Image.fromarray( (depth*500).astype(np.uint8) ).save(temp_.name)

        clicker = ck.Clicker()
        clicker.clicker(temp_.name)
        self.pcd_clicks = np.array(clicker.coords)
        
        idx = utils.ravel_index(self.pcd_clicks[:, ::-1], self.img.shape)
        
        # Find points that have actual depth and not nans
        pts = list()
        for i, val in enumerate(idx):
            count_x = 0
            count_y = 0
            counter = 0
            while any(np.isnan(self.cloud[:,val+count_x + \
                                          count_y*self.img.shape[1]])):
                counter += 1
                if counter % 2:
                    count_x += 1
                else:
                    count_y += 1
            idx[i] = val + count_x + count_y * self.img.shape[1]

        pts3D = np.zeros((3, idx.size))
        for i in range(3):
            pts3D[i,:] = self.cloud[i, idx]

        return pts3D


    # ----------------------------------------------------------------------- #
    def getTransform(self, pts2D, pts3D, KK):
        """ Get transformation between 2D coordinates and 3D coordinates.

        R, T = getTransform()
        """
        from scipy.optimize import fmin
        
        def ReprojectionErr(cam_pose, pts2D, pts3D, KK):
            RT = tf_format.tf_format('3x4', cam_pose, 'rpy')
            pts2D_proj = utils.C_ProjectPts(pts3D, RT, KK, False)
            val = np.sqrt(np.sum( (pts2D - pts2D_proj)**2 )) / pts2D.shape[1]
            print (val)
            return val

        optim = fmin(ReprojectionErr, np.zeros((6,1)), (pts2D, pts3D, KK))
        return optim

