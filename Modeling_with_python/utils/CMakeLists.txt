cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
MESSAGE("SOURCE DIR: ${PROJECT_SOURCE_DIR}")

ADD_DEFINITIONS( -fPIC )
FIND_PACKAGE ( PythonLibs)
FILE(REMOVE ${PROJECT_SOURCE_DIR}/libutils.c)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
ADD_CUSTOM_COMMAND(OUTPUT ${PROJECT_SOURCE_DIR}/libutils.c COMMAND cython ${PROJECT_SOURCE_DIR}/libutils.pyx)
rosbuild_add_library(utils ${PROJECT_SOURCE_DIR}/libutils.c)

