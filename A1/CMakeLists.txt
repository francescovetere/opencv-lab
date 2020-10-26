cmake_minimum_required(VERSION 2.8)
project(A1_313336)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(A1_313336 A1_313336.cpp)

target_link_libraries(A1_313336 ${OpenCV_LIBS})

install(TARGETS A1_313336 RUNTIME DESTINATION bin)
