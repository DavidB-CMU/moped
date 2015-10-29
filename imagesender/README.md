### ImageSender

A simulator for a camera that can be used to publish images from a dataset.

This package contains a ROS node to read images from disk and publish the Image and CameraInfo data.
  
Author: Alvaro Collet

Package contents:  
imagesender.py - Read image name in images list, send image through the network, and wait until images list is updated.  
imagesenderlist.py - Read list of images and send them all through the network.


