cmake_minimum_required(VERSION 2.8)
project(floodfill2)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(floodfill2 floodfill2.cpp)

target_link_libraries(floodfill2 ${OpenCV_LIBS})

install(TARGETS floodfill2 RUNTIME DESTINATION bin)
