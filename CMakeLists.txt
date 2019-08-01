cmake_minimum_required(VERSION 2.8.3)
project(goggles)

## Add support for C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11)

#set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
  message_generation
)

find_package(Boost REQUIRED )

find_package(Threads REQUIRED)

find_package(PCL 1.8 REQUIRED)

find_package(Ceres REQUIRED)


generate_messages(
   DEPENDENCIES
   roscpp
   std_msgs
   sensor_msgs
   geometry_msgs
 )

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs
)

list(APPEND LINK_LIBS
  ${CERES_LIBRARIES}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

###########
## Build ##
###########
set( INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set( SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set( HEADERS
  ${INC_DIR}/CeresCostFunction.h
  )

set( RAD_VEL_SRCS
  ${SRC_DIR}/radar_velocity.cpp
  )

set( RAD_IN_VEL_SRCS
  ${SRC_DIR}/radar_inertial_velocity.cpp
  )

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${pthread_INCLUDE_DIRS}
  ${rt_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${INC_DIR}
)

link_directories( ${Boost_LIBRARIES_DIRS} )
link_directories( ${PCL_LIBRARY_DIRS} )

add_definitions(${PCL_DEFINITIONS})

add_executable(radar_velocity ${RAD_VEL_SRCS} ${HEADERS})
target_link_libraries(radar_velocity
   ${LINK_LIBS}
)

add_executable(radar_inertial_velocity ${RAD_IN_VEL_SRCS} ${HEADERS})
target_link_libraries(radar_inertial_velocity
   ${LINK_LIBS}
)
