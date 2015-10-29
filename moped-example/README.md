### MOPED example

A self-contained example program to test MOPED with ROS.

This program will publish a sequence of 10 test images and MOPED will process them.

  
**Author:** Alvaro Collet

&nbsp;  

**Usage:**

Start ROS:  
`
$ roscore
`

In a separate terminal:  
`
$ roslaunch moped-example moped-example.launch
`

It will take a short time to load the program, then you should see some output like:  
```
Found 10 objects
 Found rice_tuscan at [-0.255514 -0.0222469 0.684658] [0.408402 0.260994 0.680898 0.549062] with score 39.1931
 Found tomatosoup at [-0.00130296 -0.0444741 0.66978] [0.817685 0.228818 -0.125879 0.513018] with score 39.511
 Found juicebox_front at [0.157076 0.10395 0.615698] [0.41425 0.661457 -0.520787 0.345907] with score 48.6485
```

You can view the sequence of images in Rviz:  
`
$ rosrun rviz rviz
`  
then add a display of type `"Image"` for the topic `"/moped/Image"`

You can view the detected poses:  
`
$ rostopic echo /camera_poses
`


