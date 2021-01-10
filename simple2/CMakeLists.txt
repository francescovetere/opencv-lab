cmake_minimum_required(VERSION 2.8)
project(simple2)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(simple2 simple2.cpp)

target_link_libraries(simple2 ${OpenCV_LIBS})

install(TARGETS simple2 RUNTIME DESTINATION bin)
