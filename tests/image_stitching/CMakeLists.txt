cmake_minimum_required(VERSION 2.8)
project(image_stitching)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(image_stitching image_stitching.cpp)

target_link_libraries(image_stitching ${OpenCV_LIBS})

install(TARGETS image_stitching RUNTIME DESTINATION bin)
