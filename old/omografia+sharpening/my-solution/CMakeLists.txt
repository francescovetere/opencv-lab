cmake_minimum_required(VERSION 2.8)
project(main)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(main main.cpp)

target_link_libraries( main ${OpenCV_LIBS} )
install(TARGETS main RUNTIME DESTINATION bin)
