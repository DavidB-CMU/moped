// Copyright (c) 2015, Carnegie Mellon University
// All rights reserved.
// Authors: David Butterworth <dbworth@cmu.edu>
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - Neither the name of Carnegie Mellon University nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <ros/ros.h>

#include <pr_msgs/ObjectPose.h>
#include <pr_msgs/ObjectPoseList.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/ColorRGBA.h>

// Conversions
#include <eigen_conversions/eigen_msg.h>
#include <cmath> // M_PI

// Globals
ros::Publisher g_pub_rviz_markers;
visualization_msgs::MarkerArray g_markers;
unsigned int g_marker_id;
std::string g_camera_frame;

// Add a cylinder marker
void addCylinderMarker(const geometry_msgs::Pose &pose,
                       const std_msgs::ColorRGBA &color,
                       double height,
                       double radius,
                       const std::string &ns)
{
    visualization_msgs::Marker cylinder_marker;
    cylinder_marker.header.frame_id = g_camera_frame;
    cylinder_marker.action = visualization_msgs::Marker::ADD;
    cylinder_marker.type = visualization_msgs::Marker::CYLINDER;
    cylinder_marker.lifetime = ros::Duration(0.1); // 0 = Marker never expires
    cylinder_marker.header.stamp = ros::Time::now();
    cylinder_marker.ns = ns;
    cylinder_marker.id = g_marker_id;
    g_marker_id++;
    cylinder_marker.pose = pose;
    cylinder_marker.scale.x = radius;
    cylinder_marker.scale.y = radius;
    cylinder_marker.scale.z = height;
    cylinder_marker.color = color;

    g_markers.markers.push_back(cylinder_marker);
}

// Add an axis marker
void addAxisMarker(const geometry_msgs::Pose &pose,
                   double length,
                   double radius,
                   const std::string &ns)
{
    Eigen::Affine3d pose_a3d;
    tf::poseMsgToEigen(pose, pose_a3d);

    // Publish x axis
    Eigen::Affine3d x_pose_a3d = Eigen::Translation3d(length / 2.0, 0, 0)
                               * Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitY());
    x_pose_a3d = pose_a3d * x_pose_a3d;
    geometry_msgs::Pose x_pose;
    tf::poseEigenToMsg(x_pose_a3d, x_pose);
    std_msgs::ColorRGBA red;
    red.r = 1.0;
    red.g = 0.0;
    red.b = 0.0;
    red.a = 1.0;
    addCylinderMarker(x_pose, red, length, radius, ns);

    // Publish y axis
    Eigen::Affine3d y_pose_a3d = Eigen::Translation3d(0, length / 2.0, 0)
                               * Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitX());
    y_pose_a3d = pose_a3d * y_pose_a3d;
    std_msgs::ColorRGBA green;
    green.r = 0.0;
    green.g = 1.0;
    green.b = 0.0;
    green.a = 1.0;
    geometry_msgs::Pose y_pose;
    tf::poseEigenToMsg(y_pose_a3d, y_pose);
    addCylinderMarker(y_pose, green, length, radius, ns);

    // Publish z axis
    Eigen::Affine3d z_pose_a3d = Eigen::Translation3d(0, 0, length / 2.0)
                               * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ());
    z_pose_a3d = pose_a3d * z_pose_a3d;
    std_msgs::ColorRGBA blue;
    blue.r = 0.0;
    blue.g = 0.0;
    blue.b = 1.0;
    blue.a = 1.0;
    geometry_msgs::Pose z_pose;
    tf::poseEigenToMsg(z_pose_a3d, z_pose);
    addCylinderMarker(z_pose, blue, length, radius, ns);
}

// Add a bounding box marker
void addBoundingBoxMarker(const geometry_msgs::Pose &pose,
                          const geometry_msgs::Point &min_point,
                          const geometry_msgs::Point &max_point,
                          const std_msgs::ColorRGBA &color,
                          const std::string &ns)
{
    visualization_msgs::Marker bounding_box_marker;
    bounding_box_marker.header.frame_id = g_camera_frame;
    bounding_box_marker.action = visualization_msgs::Marker::ADD;
    bounding_box_marker.type = visualization_msgs::Marker::LINE_STRIP;
    bounding_box_marker.lifetime = ros::Duration(0.1); // 0 = Marker never expires
    bounding_box_marker.header.stamp = ros::Time::now();
    bounding_box_marker.ns = ns;
    bounding_box_marker.id = g_marker_id;
    g_marker_id++;
    bounding_box_marker.scale.x = 0.003; // line width
    bounding_box_marker.color = color;
    bounding_box_marker.pose = pose;

    bounding_box_marker.points.clear();
    geometry_msgs::Point p1;
    geometry_msgs::Point p2;

    // Draw bottom square
    p1.x = min_point.x;
    p1.y = min_point.y;
    p1.z = min_point.z;
    p2.x = max_point.x;
    p2.y = min_point.y;
    p2.z = min_point.z;
    bounding_box_marker.points.push_back(p1);
    bounding_box_marker.points.push_back(p2);
    p1.x = max_point.x;
    p1.y = max_point.y;
    p1.z = min_point.z;
    p2.x = min_point.x;
    p2.y = max_point.y;
    p2.z = min_point.z;
    bounding_box_marker.points.push_back(p1);
    bounding_box_marker.points.push_back(p2);

    // Vertical line up
    p1.x = min_point.x;
    p1.y = min_point.y;
    p1.z = min_point.z;
    p2.x = min_point.x;
    p2.y = min_point.y;
    p2.z = max_point.z;
    bounding_box_marker.points.push_back(p1);
    bounding_box_marker.points.push_back(p2);

    // Top square
    p1.x = min_point.x;
    p1.y = max_point.y;
    p1.z = max_point.z;
    p2.x = max_point.x;
    p2.y = max_point.y;
    p2.z = max_point.z;
    bounding_box_marker.points.push_back(p1);
    bounding_box_marker.points.push_back(p2);

    p1.x = max_point.x;
    p1.y = min_point.y;
    p1.z = max_point.z;
    p2.x = min_point.x;
    p2.y = min_point.y;
    p2.z = max_point.z;
    bounding_box_marker.points.push_back(p1);
    bounding_box_marker.points.push_back(p2);

    p1.x = min_point.x;
    p1.y = min_point.y;
    p1.z = max_point.z;
    p2.x = min_point.x;
    p2.y = max_point.y;
    p2.z = max_point.z;
    bounding_box_marker.points.push_back(p1);
    bounding_box_marker.points.push_back(p2);

    // Vertical line down
    p1.x = min_point.x;
    p1.y = max_point.y;
    p1.z = max_point.z;
    p2.x = min_point.x;
    p2.y = max_point.y;
    p2.z = min_point.z;
    bounding_box_marker.points.push_back(p1);
    bounding_box_marker.points.push_back(p2);

    // Bottom side line
    p1.x = min_point.x;
    p1.y = max_point.y;
    p1.z = min_point.z;
    p2.x = min_point.x;
    p2.y = min_point.y;
    p2.z = min_point.z;
    bounding_box_marker.points.push_back(p1);
    bounding_box_marker.points.push_back(p2);

    // Vertical line up
    p1.x = max_point.x;
    p1.y = min_point.y;
    p1.z = min_point.z;
    p2.x = max_point.x;
    p2.y = min_point.y;
    p2.z = max_point.z;
    bounding_box_marker.points.push_back(p1);
    bounding_box_marker.points.push_back(p2);

    // Top side line
    p1.x = max_point.x;
    p1.y = min_point.y;
    p1.z = max_point.z;
    p2.x = max_point.x;
    p2.y = max_point.y;
    p2.z = max_point.z;
    bounding_box_marker.points.push_back(p1);
    bounding_box_marker.points.push_back(p2);

    // Vertical line down
    p1.x = max_point.x;
    p1.y = max_point.y;
    p1.z = max_point.z;
    p2.x = max_point.x;
    p2.y = max_point.y;
    p2.z = min_point.z;
    bounding_box_marker.points.push_back(p1);
    bounding_box_marker.points.push_back(p2);

    // Bottom side line
    p1.x = max_point.x;
    p1.y = max_point.y;
    p1.z = min_point.z;
    p2.x = max_point.x;
    p2.y = min_point.y;
    p2.z = min_point.z;
    bounding_box_marker.points.push_back(p1);
    bounding_box_marker.points.push_back(p2);

    g_markers.markers.push_back(bounding_box_marker);
}

void objectPosesCallback(const pr_msgs::ObjectPoseList::ConstPtr& msg)
{
    g_markers.markers.clear();
    g_marker_id = 0;

    // Iterate over the detected objects
    for (int i=0; i < msg->object_list.size(); i++)
    {
        std::string object_name = msg->object_list[i].name;
        geometry_msgs::Pose object_pose = msg->object_list[i].pose;
        ROS_INFO("Detected object %s \n", object_name.c_str() );

        // Add an axis marker for this object
        std::string ns = object_name;
        addAxisMarker(object_pose, 0.12, 0.002, ns);

        // Add a bounding box marker for this object
        std_msgs::ColorRGBA light_green;
        light_green.r = 0.0;
        light_green.g = 0.8;
        light_green.b = 0.0;
        light_green.a = 1.0;
        addBoundingBoxMarker(object_pose, msg->object_list[i].bounding_box_min, msg->object_list[i].bounding_box_max, light_green, ns);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "moped_object_pose_publisher");
    ros::NodeHandle nh;

    // Get the TF frame of the camera
    ros::NodeHandle pnh("~");
    pnh.param("camera_frame", g_camera_frame, std::string("/map"));

    ros::Subscriber sub = nh.subscribe("/camera_poses", 10, objectPosesCallback);

    // Rviz marker publisher
    g_pub_rviz_markers = nh.advertise<visualization_msgs::MarkerArray>("/visualization_marker_array", 10);

    g_marker_id = 0;

    ros::Rate loop_rate(10);
    while (ros::ok())
    {
        // Publish Rviz Markers
        g_pub_rviz_markers.publish(g_markers);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}

