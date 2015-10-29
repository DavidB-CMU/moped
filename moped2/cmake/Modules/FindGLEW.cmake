# FindGLEW.cmake
# This file was not included with CMake until version 2.8.10
# 
# Try to find GLEW library and include path.
# Once done, this will define:
#   GLEW_FOUND
#   GLEW_INCLUDE_DIR
#   GLEW_LIBRARIES
# 

IF (WIN32)
    FIND_PATH( GLEW_INCLUDE_DIR GL/glew.h
        $ENV{PROGRAMFILES}/GLEW/include
        ${PROJECT_SOURCE_DIR}/src/nvgl/glew/include
        DOC "The directory where GL/glew.h resides")
    FIND_LIBRARY( GLEW_LIBRARIES
        NAMES glew GLEW glew32 glew32s
        PATHS
        $ENV{PROGRAMFILES}/GLEW/lib
        ${PROJECT_SOURCE_DIR}/src/nvgl/glew/bin
        ${PROJECT_SOURCE_DIR}/src/nvgl/glew/lib
        DOC "The GLEW library")
ELSE (WIN32)
    FIND_PATH( GLEW_INCLUDE_DIR GL/glew.h
        /usr/include
        /usr/local/include
        /sw/include
        /opt/local/include
        DOC "The directory where GL/glew.h resides")
    FIND_LIBRARY( GLEW_LIBRARIES
        NAMES GLEW glew
        PATHS
        /usr/lib64
        /usr/lib
        /usr/local/lib64
        /usr/local/lib
        /sw/lib
        /opt/local/lib
        DOC "The GLEW library")
ENDIF (WIN32)

IF (GLEW_INCLUDE_DIR)
    SET( GLEW_FOUND 1 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
    MESSAGE("Using GLEW_INCLUDE_DIR = " ${GLEW_INCLUDE_DIR})
    MESSAGE("Using GLEW_LIBRARIES = " ${GLEW_LIBRARIES}) 
ELSE (GLEW_INCLUDE_DIR)
    SET( GLEW_FOUND 0 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ENDIF (GLEW_INCLUDE_DIR)

MARK_AS_ADVANCED( GLEW_FOUND ) 

