cmake_minimum_required(VERSION 2.8)
project(histogram_equalization)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(histogram_equalization histogram_equalization.cpp)

target_link_libraries(histogram_equalization ${OpenCV_LIBS})

install(TARGETS histogram_equalization RUNTIME DESTINATION bin)
