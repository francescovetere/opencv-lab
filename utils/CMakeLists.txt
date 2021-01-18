cmake_minimum_required(VERSION 2.8)
project(general)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(general general.cpp)

target_link_libraries(general ${OpenCV_LIBS})

install(TARGETS general RUNTIME DESTINATION bin)
