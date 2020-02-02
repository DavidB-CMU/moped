## MOPED (Multiple Object Pose Estimation and Detection)

This is a framework for the detection and pose estimation of multiple objects using a monocular camera.

<a href="https://raw.githubusercontent.com/DavidB-CMU/moped/master/moped2/screenshots/09_detected_objects_bounding_boxes.png"><img width="400" align="right" src="https://raw.githubusercontent.com/DavidB-CMU/moped/master/moped2/screenshots/09_detected_objects_bounding_boxes.png"></a>

**Authors:**  
Alvaro Collet  
Manuel Martinez  
Siddhartha Srinivasa  

**Package maintainer:**  
David Butterworth

&nbsp;

**Update Oct 2015:**  
MOPED has been verified working with ROS Fuerte  
on Ubuntu 11.10 Oneiric Ocelot.  
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

### Documentation & installation instructions:

See the README files in the `moped2` and `moped3d` packages.

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


