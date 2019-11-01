project(GogglesEval)
cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_CXX_STANDARD 11)

find_package(Ceres REQUIRED)
find_package(GFlags REQUIRED)

list(APPEND LINK_LIBS
  ${CERES_LIBRARIES}
  ${GTEST_LIBRARIES}
)

set( INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include)
set( SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

set(HEADERS
  ${INC_DIR}/DataTypes.h
  ${INC_DIR}/ErrorInterface.h
  ${INC_DIR}/GlobalImuVelocityCostFunction.h
  ${INC_DIR}/GlobalVelocityMeasCostFunction.h
  ${INC_DIR}/QuaternionParameterization.h)

set(SRCS
  ${SRC_DIR}/GlobalImuVelocityCostFunction.cpp
  ${SRC_DIR}/GlobalVelocityMeasCostFunction.cpp
  ${SRC_DIR}/QuaternionParameterization.cpp)

include_directories(
  ${CERES_INCLUDE_DIRS},
  ${GFLAGS_INCLUDE_DIRS},
  ${INC_DIR})

add_executable(alignByVel alignByVel.cpp ${SRCS} ${HEADERS})
target_link_libraries(alignByVel ${LINK_LIBS})

#add_executable(alignByImu alignByImu.cpp ${SRCS} ${HEADERS})
#target_link_libraries(alignByImu ${LINK_LIBS})