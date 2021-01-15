cmake_minimum_required(VERSION 2.8)
project(ottagono)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(ottagono ottagono.cpp)

target_link_libraries(ottagono ${OpenCV_LIBS})

install(TARGETS ottagono RUNTIME DESTINATION bin)
