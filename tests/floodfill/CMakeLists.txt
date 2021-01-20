cmake_minimum_required(VERSION 2.8)
project(floodfill)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(floodfill floodfill.cpp)

target_link_libraries(floodfill ${OpenCV_LIBS})

install(TARGETS floodfill RUNTIME DESTINATION bin)
