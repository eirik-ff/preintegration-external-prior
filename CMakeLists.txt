cmake_minimum_required(VERSION 3.0.0)
project(preint_extpos_test)

#set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS roscpp cv_bridge)

find_package(GTSAMCMakeTools)
include(GtsamBuildTypes)
find_package(GTSAM REQUIRED)
find_package(GTSAM_UNSTABLE REQUIRED)

add_definitions("-pthread -g -Wall")

catkin_package(
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
    ${GTSAM_UNSTABLE_INCLUDE_DIR}
)

add_executable(test_node
    src/main.cpp
)

target_link_libraries(test_node
    ${catkin_LIBRARIES}
    gtsam 
    gtsam_unstable 
)
