cmake_minimum_required(VERSION 2.8)
project(kmeans)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(kmeans kmeans.cpp)

target_link_libraries(kmeans ${OpenCV_LIBS})

install(TARGETS kmeans RUNTIME DESTINATION bin)
