cmake_minimum_required(VERSION 2.8)
project(simple)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(simple main.cpp)

target_link_libraries(simple ${OpenCV_LIBS} )
install(TARGETS simple RUNTIME DESTINATION bin)
