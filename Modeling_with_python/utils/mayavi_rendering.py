################################################################################
#                                                                              
# mayavi_rendering.py: examples on how to use mayavi to render animations/video 
#
# Copyright: Carnegie Mellon University & Intel Corp.
# Author: Alvaro Collet (acollet@cs.cmu.edu)
#
################################################################################


# ---------------------------------------------------------------------------- #
def example():
    """ Example to run offscreen rendering with mayavi. If you see an 'int64'
    error, you will need to patch 'figure.py' to '... = int(magnification)'.
    This example saves a file called 'example.png' in the current folder, with
    some shapes.

    Usage: example()
    """
    from enthought.mayavi import mlab
    mlab.options.offscreen = True
    mlab.test_contour3d()
    mlab.savefig('example.png')
    
# ---------------------------------------------------------------------------- #
def move_cam(outpath):
    """ Example to record a stream of images moving the camera around 
    (using offscreen rendering)

    Usage: move_cam(out_path)

    Input:
        outpath - (String) output path to save images
       
    Output:
        -NONE- but stream of images is saved in outpath
    """

    import os.path
    import tf_format
    from enthought.mayavi import mlab
    mlab.options.offscreen = True
    mlab.test_contour3d()

    # Example motion: rotate clockwise around an object
    step = 15 # degrees of change between views

    for i, angle in enumerate(range(0, 360, step)):
        view = mlab.view(azimuth=angle)
        imgname = os.path.join(outpath, 'example' + str(i) + '.png')
        mlab.savefig(imgname)


