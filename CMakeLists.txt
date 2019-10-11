cmake_minimum_required(VERSION 2.8.3)
project(goggles)
set(CMAKE_CXX_STANDARD 11)

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
	tf
)

enable_testing()
find_package(Boost REQUIRED )
find_package(Threads REQUIRED)
find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)
find_package(GTest REQUIRED)
find_package(yaml-cpp REQUIRED)

generate_messages(
   DEPENDENCIES
   roscpp
   std_msgs
   sensor_msgs
   geometry_msgs
 )

catkin_package(
  CATKIN_DEPENDS 
  roscpp std_msgs 
  sensor_msgs 
  tf 
  pcl_ros
  geometry_msgs
)

list(APPEND LINK_LIBS
  ${CERES_LIBRARIES}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${GTEST_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  pthread
)

###########
## Build ##
###########
set( INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set( SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set( APP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/apps)
set( HEADERS
  ${INC_DIR}/ErrorInterface.h
  ${INC_DIR}/BodyVelocityCostFunction.h
  ${INC_DIR}/VelocityChangeCostFunction.h
  ${INC_DIR}/VelocityMeasCostFunction.h
  ${INC_DIR}/ImuVelocityCostFunction.h
  ${INC_DIR}/ImuIntegrator.h
  ${INC_DIR}/MarginalizationError.h
	${INC_DIR}/QuaternionParameterization.h
	${INC_DIR}/DataTypes.h
)

set( SOURCES 
  ${SRC_DIR}/BodyVelocityCostFunction.cpp
  ${SRC_DIR}/VelocityChangeCostFunction.cpp
  ${SRC_DIR}/VelocityMeasCostFunction.cpp
  ${SRC_DIR}/ImuVelocityCostFunction.cpp
  ${SRC_DIR}/ImuIntegrator.cpp
  ${SRC_DIR}/MarginalizationError.cpp
  ${SRC_DIR}/QuaternionParameterization.cpp
  )

set( RAD_VEL_SRCS
  ${APP_DIR}/radar_velocity.cpp
  )

set( RAD_IN_VEL_SRCS
  ${APP_DIR}/radar_inertial_velocity.cpp
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
  ${GTEST_INCLUDE_DIRS}
	${YAML_CPP_INCLUDE_DIRS}
  ${INC_DIR}
)

link_directories( ${Boost_LIBRARIES_DIRS} )
link_directories( ${PCL_LIBRARY_DIRS} )

add_definitions(${PCL_DEFINITIONS})

#add_subdirectory(test)

catkin_add_gtest(testImuIntegration test/testImuIntegration.cpp ${SOURCES} ${HEADERS})
target_link_libraries(testImuIntegration ${LINK_LIBS})
catkin_add_gtest(testMarginalization test/testMarginalization.cpp ${SOURCES} ${HEADERS})
target_link_libraries(testMarginalization ${LINK_LIBS})

add_executable(radar_velocity ${RAD_VEL_SRCS} ${SOURCES} ${HEADERS})
target_link_libraries(radar_velocity
   ${LINK_LIBS}
)

add_executable(radar_inertial_velocity ${RAD_IN_VEL_SRCS} ${SOURCES} ${HEADERS})
target_link_libraries(radar_inertial_velocity
   ${LINK_LIBS}
)
