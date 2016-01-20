## MOPED (Multiple Object Pose Estimation and Detection)

This is a framework for the detection and pose estimation of multiple objects using a monocular camera.

**Authors:**  
Alvaro Collet  
Manuel Martinez  
Siddhartha Srinivasa  

**Package maintainer:**  
David Butterworth

&nbsp;

**Update Oct 2015:**  
MOPED has been verified working with ROS Fuerte on Ubuntu 11.10 Oneiric Ocelot.  
You can read the installation guide below.

&nbsp;
 
### Repository contents:

The packages in this repository are integrated with ROS and were made to be compiled using rosbuild (not catkin).

**BundlerPy**  

**imagesender**  
A node to publish images from a dataset.

**meanshiftpy**  

**moped_example**  
A self-contained program to test MOPED using a sequence of image frames.

**moped-modeling-py**  

**moped2**  
Original MOPED framework using a single, monocular 2D camera.

**moped3d**  

**moped_models**
Sample object models for use with the example program.

**moped_object_pose_publisher**
A node to publish Rviz Markers for the detected objects. 

**pr_msgs**  
Custom ROS messages and services used by the MOPED framework.

**pyublas**  

&nbsp;

### Related publications:

#### moped2:

*"Lifelong Robotic Object Perception"*  
Alvaro Collet Romea  
Doctoral dissertation, tech. report CMU-RI-TR-12-22, Robotics Institute, Carnegie Mellon University, August, 2012.  
[Details](https://www.ri.cmu.edu/publication_view.html?pub_id=7326&menu_code=0307) | [PDF](https://www.ri.cmu.edu/pub_files/2012/8/phd_thesis.pdf) (19MB)

*"The MOPED framework: Object Recognition and Pose Estimation for Manipulation"*  
Alvaro Collet Romea, Manuel Martinez Torres, and Siddhartha Srinivasa  
International Journal of Robotics Research, Vol. 30, No. 10, pp. 1284 - 1306, September, 2011.  
[Details](https://www.ri.cmu.edu/publication_view.html?pub_id=6856&menu_code=0307) | [PDF](https://www.ri.cmu.edu/pub_files/2011/9/moped.pdf) (3MB)

*"Efficient Multi-View Object Recognition and Full Pose Estimation"*  
Alvaro Collet Romea and Siddhartha Srinivasa  
2010 IEEE International Conference on Robotics and Automation (ICRA 2010), May, 2010.  
[Details](https://www.ri.cmu.edu/publication_view.html?pub_id=6564&menu_code=0307) | [PDF](https://www.ri.cmu.edu/pub_files/2010/5/Collet2010.pdf) (1MB)

*"MOPED: A Scalable and Low Latency Object Recognition and Pose Estimation System"*  
Manuel Martinez Torres, Alvaro Collet Romea, and Siddhartha Srinivasa  
Proceedings of ICRA 2010, May, 2010.  
[Details](https://www.ri.cmu.edu/publication_view.html?pub_id=6543&menu_code=0307) | [PDF](https://www.ri.cmu.edu/pub_files/2010/5/icra10.pdf) (1MB)

*"Object Recognition and Full Pose Registration from a Single Image for Robotic Manipulation"*  
Alvaro Collet Romea, Dmitry Berenson, Siddhartha Srinivasa, and David Ferguson  
IEEE International Conference on Robotics and Automation (ICRA 2009), May, 2009.  
[Details](https://www.ri.cmu.edu/publication_view.html?pub_id=6301&menu_code=0307) | [PDF](https://www.ri.cmu.edu/pub_files/2009/5/icra09_final.pdf) (1MB)

#### moped3d:

*"Object Recognition Robust to Imperfect Depth Data"*  
David Fouhey, Alvaro Collet Romea, Martial Hebert , and Siddhartha Srinivasa  
2nd Workshop on Consumer Depth Cameras for Computer Vision in conjunction with ECCV 2012, October, 2012.  
[Details](https://www.ri.cmu.edu/publication_view.html?pub_id=7252&menu_code=0307) | [PDF](https://www.ri.cmu.edu/pub_files/2012/10/cdc4cv.pdf) (3MB)

&nbsp;

### Documentation:

The available documentation is:  
 - This README file.  
 - doxygen  
Run 'doxygen mainpage.dox' in the `moped2` directory

&nbsp;

### How to install MOPED with ROS Fuerte:

This tutorial will explain how to install and test `moped2`.

The following computer setup was used:  
 - Intel quad-core i7 CPU
 - Dual Intel/Nvidia graphics card
 - Ubuntu 11.10 Oneiric Ocelot
 - ROS Fuerte
 - rosbuild (not catkin)
 - python 2.7.2
 - cmake 2.8.5

#### Install the required dependencies:

We will assume that you already have ROS Fuerte installed correctly.

First we need to install some external dependencies.

Update your package information:  
`$ sudo apt-get update`

You should install the following libraries using apt-get or aptitude:  
`libcv2.1`  
`libopencv2.3`  
`libopencv2.3-dev`  
`libgomp1`  
`freeglut3`  
`freeglut3-dev`  

For GPU support you should also install these libraries:  
`libglew-dev`  
`libdevil-dev`  

If you can't find any of these packages, make sure you have all the Ubuntu repositories enabled and you have run `apt-get update`.

You should install the following ROS libraries:  
`ros-fuerte-actionlib`  
`ros-fuerte-opencv2`  
`ros-fuerte-vision-opencv`  
`ros-fuerte-image-transport-plugins`  

#### Download the source code:

We will assume you already have a rosbuild workspace  
e.g.  
`~/ros_workspace/`

Download the entire `moped` repository to your computer.

Extract the files.  
We only need to use some of the included packages.

Move the required package directories into your ROS workspace.   
You should end up with the following structure:   
`~/ros_workspace/imagesender`  
`~/ros_workspace/moped2`  
`~/ros_workspace/moped_models`  
`~/ros_workspace/moped_example`  
`~/ros_workspace/pr_msgs`  

Don't forget to source the appropriate ROS `setup.bash` file  
and add the above package directories to your `ROS_PACKAGE_PATH`:  
`$ source /opt/ros/fuerte/setup.bash`  
`$ export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/ros_workspace/pr_msgs`  

#### Compile the packages:

We will build each package separately to make sure there are no problems.

**pr_msgs:**

Check dependencies:  
`$ rosdep check pr_msgs`

`$ rosmake pr_msgs`

If this is successful, the following directory structure will have been created in the pr_msgs package:  
> ~/ros_workspace/pr_msgs/bin/  
> ~/ros_workspace/pr_msgs/build/  
> ~/ros_workspace/pr_msgs/msg_gen/  
> ~/ros_workspace/pr_msgs/src/  
> ~/ros_workspace/pr_msgs/srv_gen/  

**moped2:**

Check dependencies:  
`$ rosdep check moped2`

You should see a message like the following:  
> All system dependencies have been satisified  
> ERROR[moped2]: Cannot locate rosdep definition for [glew]  

You can ignore the error about GLEW.

Compile libmoped:  
`$ cd ~/ros_workspace/moped2/libmoped/`  
`$ make clean`  
`$ make`  

You should see no error messages.

Compile the moped2 package:  
`$ cd ~/ros_workspace/`  
`$ rosmake moped2`  

You should see no error messages.

**imagesender:**

There is nothing to compile.

Check that imagesender is installed correctly.  
Start ROS:  
`$ roscore`  
Then in a separate terminal:  
`$ roslaunch imagesender imagesender_test.launch`  
If working correctly, imagesender will continuously publish a test image.  
You can view the ROS Msg:  
`$ rostopic echo /Image`

**moped_example:**

There is nothing to compile.  

&nbsp;

### Check that MOPED is installed correctly:

We will run an example program to check that MOPED is installed correctly.  
This program publishes a sequence of image frames as ROS messages and MOPED will detect the poses of objects in each image frame.

Start ROS:  
`$ roscore`  
In a separate terminal:  
`$ roslaunch moped_example moped_example.launch`  

It will take a short time to load MOPED, then you should see some output like this:  
```
Found 10 objects
 Found rice_tuscan at [-0.255514 -0.0222469 0.684658] [0.408402 0.260994 0.680898 0.549062] with score 39.1931
 Found tomatosoup at [-0.00130296 -0.0444741 0.66978] [0.817685 0.228818 -0.125879 0.513018] with score 39.511
 Found juicebox_front at [0.157076 0.10395 0.615698] [0.41425 0.661457 -0.520787 0.345907] with score 48.6485
```

You can view the sequence of images in Rviz:  
`$ rosrun rviz rviz`  
then add a display of type `"Image"` for the topic `"/moped/Image"`

You can view the detected poses:  
`$ rostopic echo /camera_poses`  

&nbsp;

### Modify MOPED's configuration:

You can modify the configuration to change which algorithms are used in each part of MOPED's processing pipeline.

**Enabling GPU support:**

Edit this config file:  
`~/ros_workspace/moped2/libmoped/src/config.hpp`  

By default, MOPED is detecting SIFT features using the CPU.  
In the config file you can see these options:  
`//#define USE_GPU 1`  
`#define USE_CPU 1`  

To use SIFT_GPU instead, just switch the parameter that is commented-out.

Re-compile libmoped:  
`$ cd ~/ros_workspace/moped2/libmoped/`  
`$ make clean`  
`$ make`  

Re-compile the moped2 package:  
`$ rosmake moped2`  

Launch the MOPED example again:  
`$ roslaunch moped_example moped_example.launch`  

If everything is working correctly, you will see the object poses printed to the terminal and a pop-up window displaying the features detected by SIFT_GPU.

*Note:* When testing MOPED using a laptop with a dual Intel-NVidia graphics card, it fails to render the SIFT_GPU pop-up window, however SIFT_GPU does work correctly.  

If MOPED no longer detects any objects with SIFT_GPU enabled  
*"Found 0 objects"*  
then MOPED may not be utilizing the GPU correctly.  
On laptops with dual Intel-NVidia graphics you need to execute a special command before running a program that requires the GPU.

**Enabling graphical displays:**

The configuration file  
`~/ros_workspace/moped2/libmoped/src/config.hpp`  
also includes various ***_DISPLAY modules which are commented out.  
These are described in the section below.

To enable these graphical displays, you need to un-comment them in the config file and then re-compile MOPED:  
`cd ~/ros_workspace/moped2/libmoped/`  
`make clean`  
`make`  
`rosmake moped2`  

&nbsp;

### Graphical displays in MOPED

MOPED allows you to enable various graphical display windows for debugging each step in the pipeline.

**FEAT_DISPLAY**  
Display the detected image features (SIFT or SURF)  
![FEAT_DISPLAY](https://raw.github.com/DavidB-CMU/moped/master/screenshots/moped2/FEAT_DISPLAY_1.png)

**MATCH_DISPLAY**  
Display the matched feature points  
![MATCH_DISPLAY](https://raw.github.com/DavidB-CMU/moped/master/screenshots/moped2/MATCH_DISPLAY_1.png)

**CLUSTER_DISPLAY**  
Display the detected cluster regions  
![CLUSTER_DISPLAY](https://raw.github.com/DavidB-CMU/moped/master/screenshots/moped2/CLUSTER_DISPLAY_1.png)

**POSE_DISPLAY**  
Display the multiple pose hypotheses  
![POSE_DISPLAY](https://raw.github.com/DavidB-CMU/moped/master/screenshots/moped2/POSE_DISPLAY_1.png)

**POSE2_DISPLAY**  
Display the final pose of each detected object  
![POSE2_DISPLAY](https://raw.github.com/DavidB-CMU/moped/master/screenshots/moped2/POSE2_DISPLAY_1.png)

**GLOBAL_DISPLAY**  
Display a window containing views of various steps of the pipeline and a graph of the processing load at each step  
![GLOBAL_DISPLAY](https://raw.github.com/DavidB-CMU/moped/master/screenshots/moped2/GLOBAL_DISPLAY_1.png)



