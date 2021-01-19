cmake_minimum_required(VERSION 2.8)
project(SAD)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(SAD SAD.cpp)

target_link_libraries(SAD ${OpenCV_LIBS})

install(TARGETS SAD RUNTIME DESTINATION bin)
